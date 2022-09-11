import cvxpy as cp
import numpy as np
import pandas as pd
import scipy as sp

import datetime as dt
import itertools
from typing import List, Dict, Union, Callable


class Objective:

    def generate_objective(self,
                           date: dt.date,
                           holdings: pd.DataFrame,
                           variables: Dict,
                           port_info: Dict):
        pass


class MinTaxObjective(Objective):

    def __init__(self,
                 lt_gains_rate: float,
                 income_rate: float,
                 lt_cutoff_days=365):
        self.lt_gains_rate = lt_gains_rate
        self.income_rate = income_rate
        self.lt_cutoff_days = lt_cutoff_days

    def generate_objective(self,
                           date: dt.date,
                           holdings: pd.DataFrame,
                           variables: Dict,
                           port_info: Dict):

        current_date = date
        lt_cutoff_days = self.lt_cutoff_days
        st_rate, lt_rate = self.income_rate, self.lt_gains_rate

        sells = variables['sells']
        tax = 0
        for i in holdings.index:
            lot_info = holdings.loc[i]
            asset, date = lot_info['ticker'], lot_info['purchase_date']
            purchase_date = dt.date.fromisoformat(date)
            holding_period = (current_date - purchase_date).days
            if holding_period <= lt_cutoff_days:
                lot_rate = st_rate
            else:
                lot_rate = lt_rate
            gain = lot_info['current_price'] / lot_info['purchase_price']
            effective_rate = (gain - 1) * lot_rate
            tax += sells[asset][date] * effective_rate

        objective = cp.Minimize(tax)

        return objective


class MinTrackingErrorObjective(Objective):

    def __init__(self,
                 target_weights: pd.Series,
                 sigma: pd.DataFrame):
        self.target_weights = target_weights
        self.sigma = sigma

    def generate_objective(self,
                           date: dt.date,
                           holdings: pd.DataFrame,
                           variables: Dict,
                           port_info: Dict):
        target_weights = self.target_weights
        assets = target_weights.index
        weights = pd.Series(variables['positions'])[assets] / \
            port_info['investment_value']
        sigma = self.sigma.loc[assets][assets]
        diffs = weights - target_weights

        objective = cp.Minimize(sum((sp.linalg.sqrtm(sigma) @ diffs) ** 2))

        return objective


class Constraint:

    def generate_constraint(self,
                            date: dt.date,
                            holdings: pd.DataFrame,
                            variables: Dict,
                            port_info: Dict) -> List:
        pass


class FullInvestmentConstraint(Constraint):

    def __init__(self):
        """ Constraint to enforce full investment """
        pass

    def generate_constraint(self, date, holdings, variables, port_info):
        positions = variables['positions']
        total_invested = sum(list(positions.values()))

        return [total_invested == port_info['investment_value']]


class LongOnlyConstraint(Constraint):

    def __init__(self):
        """ Constraint to enforce all portfolio holdings are non-negative
        """
        pass

    def generate_constraint(self, date, holdings, variables, port_info):
        return [v >= 0 for v in variables['positions'].values()]


class DoNotIncreaseDeviationConstraint(Constraint):

    def __init__(self,
                 target_weights: pd.Series):
        """ Constraint that prohibits buying in currently overweight assets
        and selling in currently underweight assets

        :param target_weights: target portfolio weights
        """
        self.target_weights = target_weights

    def generate_constraint(self, date, holdings, variables, port_info):

        all_assets = variables['buys'].keys()
        current_port = holdings[['ticker', 'value']]. \
            groupby(['ticker']). \
            sum()['value']. \
            reindex(list(all_assets)). \
            fillna(0.0)

        target_port = self.target_weights * port_info['investment_value']
        cons = []
        for asset in all_assets:
            if current_port[asset] >= target_port[asset]:
                cons.append(variables['buys'][asset] == 0)

            if asset not in variables['sells']:
                continue

            if current_port[asset] <= target_port[asset]:
                for sell in variables['sells'][asset].values():
                    cons.append(sell == 0)

        return cons


class DoNotTradePastTargetConstraint(Constraint):

    def __init__(self,
                 target_weights: pd.Series):
        """ Prevent trading past the target weight.
        Constrain positions of currently overweight assets to not be less
        than the target, and positions of currently underweight assets
        to not be more than the target.

        :param target_weights: Weights of the target portfolio
        """
        self.target_weights = target_weights

    def generate_constraint(self, date, holdings, variables, port_info):

        positions = variables['positions']
        all_assets = variables['buys'].keys()
        current_port = holdings[['ticker', 'value']]. \
            groupby(['ticker']). \
            sum()['value']. \
            reindex(list(all_assets)). \
            fillna(0.0)

        target_port = self.target_weights * port_info['investment_value']
        cons = []
        for asset in all_assets:
            target_position = target_port[asset]
            if current_port[asset] >= target_position:
                cons.append(positions[asset] >= target_position)

            if current_port[asset] <= target_position:
                cons.append(positions[asset] <= target_position)

        return cons


class VolBasedDeviationConstraint(Constraint):

    def __init__(self,
                 target_weights: pd.Series,
                 asset_vols: pd.Series,
                 bounds: Union[float, pd.Series]):
        """ Set deviation constraints on a per-asset basis, based on each
        asset's volatility. Constraints are of the form

            |h_i - t_i| <= vol_i * bounds_i

        :param target_weights: Weights of the target portfolio
        :param asset_vols: Volatility of each asset
        :param bounds: Tolerances for each asset. If a single number is
            passed, that value is used for all assets
        """

        self.target_weights = target_weights
        self.asset_vols = asset_vols
        self.bounds = bounds

    def generate_constraint(self, date, holdings, variables, port_info):

        positions = variables['positions']
        all_assets = variables['buys'].keys()
        investment_value = port_info['investment_value']
        target_port = self.target_weights * investment_value
        bounds = self.bounds
        if not isinstance(bounds, pd.Series):
            bounds = pd.Series(bounds, list(all_assets))

        cons = []
        for asset in all_assets:
            lhs = cp.abs(positions[asset] - target_port[asset])
            rhs = bounds[asset] * self.asset_vols[asset] * investment_value
            cons.append(lhs <= rhs)

        return cons


class MaxDeviationConstraint(Constraint):

    def __init__(self,
                 target_weights: pd.Series,
                 bounds: Union[float, pd.Series]):
        """ Constrain each asset to be within a given tolerance of the
        target

        :param target_weights: Weights of the target portfolio
        :param bounds: Amount of tolerance to allow in each asset's weight.
            If a single number is passed, that value is used for all assets.
        """

        self.target_weights = target_weights
        self.bounds = bounds

    def generate_constraint(self, date, holdings, variables, port_info):

        positions = variables['positions']
        all_assets = variables['buys'].keys()
        investment_value = port_info['investment_value']
        target_port = self.target_weights * investment_value
        bounds = self.bounds
        if not isinstance(bounds, pd.Series):
            bounds = pd.Series(bounds, list(all_assets))

        cons = []
        for asset in all_assets:
            lhs = cp.abs(positions[asset] - target_port[asset])
            rhs = bounds[asset] * investment_value
            cons.append(lhs <= rhs)

        return cons


class RebalancingOpt:

    def __init__(self,
                 date: dt.date,
                 target_port: pd.Series,
                 holdings: pd.DataFrame,
                 constraints: List[Constraint],
                 objective: Objective):
        """ Create an instance of an optimization problem to rebalance a
        portfolio

        :param date: current date
        :param target_port: target portfolio, in dollars
        :param holdings: holdings information, including share quantity,
            price, ticker
        :param constraints: constraints to apply in the problem
        :param objective: objective to use in the problem
        """

        self.date = date
        self.target_port = target_port
        all_assets = target_port.index.values
        if holdings.shape[0]:
            all_assets = np.concatenate((all_assets,
                                         holdings['ticker'].values))

        self.assets = np.unique(all_assets)
        self.holdings = holdings
        self.variables = self._generate_variables(holdings)
        cons = self._generate_constraints(constraints)
        obj = self._generate_objective(objective)
        self.prob = cp.Problem(obj, cons)

    def _generate_variables(self, holdings):
        all_assets = self.assets
        variables = {'buys': {}, 'sells': {}, 'positions': {}}

        asset_holdings = holdings[['ticker', 'value']]. \
            groupby(['ticker']). \
            sum()['value']. \
            reindex(all_assets). \
            fillna(0.0)

        for asset in all_assets:
            variables['buys'][asset] = cp.Variable(nonneg=True)

        for i in holdings.index:
            lot_info = holdings.loc[i]
            asset, date = lot_info['ticker'], lot_info['purchase_date']
            if asset not in variables['sells']:
                variables['sells'][asset] = {}
            variables['sells'][asset][date] = cp.Variable(nonneg=True)

        for asset in all_assets:
            variables['positions'][asset] = asset_holdings[asset] + \
                                            variables['buys'][asset]
            if asset in variables['sells']:
                asset_sell = \
                    sum([x for x in variables['sells'][asset].values()])
                variables['positions'][asset] -= asset_sell

        return variables

    def _generate_constraints(self, constraints):
        target_port = self.target_port
        port_info = {'investment_value': target_port.sum()}
        cons = [c.generate_constraint(self.date, self.holdings,
                                      self.variables, port_info)
                for c in constraints]

        sell_size_cons = []
        sells = self.variables['sells']
        holdings = self.holdings
        for i in holdings.index:
            lot_info = holdings.loc[i]
            asset, date = lot_info['ticker'], lot_info['purchase_date']
            sell_size_cons.append(sells[asset][date] <= lot_info['value'])

        cons = list(itertools.chain.from_iterable(cons))
        cons.extend(sell_size_cons)

        return cons

    def _generate_objective(self, objective):
        target_port = self.target_port
        port_info = {'investment_value': target_port.sum()}
        return objective.generate_objective(self.date, self.holdings,
                                            self.variables, port_info)

    def solve(self):
        self.prob.solve()

    def get_trades(self):
        variables = self.variables

        buys = {a: v.value for a, v in variables['buys'].items()}
        buys = np.round(pd.Series(buys), 2)
        sells = variables['sells']
        sell_values = {}
        for asset, asset_sells in sells.items():
            asset_sells = {d: v.value for d, v in asset_sells.items()}
            asset_sells = np.round(pd.Series(asset_sells), 2)
            sell_values[asset] = asset_sells

        return {'buys': buys, 'sells': sell_values}


class Rebalancer:

    def __init__(self, target_weights: pd.Series):
        self.target_weights = target_weights

    def rebalance(self,
                  date: dt.date,
                  holdings: pd.DataFrame,
                  investment_value: float):
        pass


class SimpleRebalancer(Rebalancer):

    def __init__(self,
                 target_weights: pd.Series,
                 tax_params: Dict):
        super().__init__(target_weights)
        self.tax_params = tax_params

    def generate_complete_trades(self,
                                 date: dt.date,
                                 holdings: pd.DataFrame,
                                 investment_value: float):
        """ Calculate trades that would take the invested portfolio all
        the way back to the target weights, then select tax-optimized
        lots for sells. Trades are returned as dollar values.

        :param date: current date
        :param holdings: current holdings
        :param investment_value: dollar amount to invest
        :return: dictionary with buys and sells
        """
        asset_holdings = holdings[['ticker', 'value']]. \
            groupby(['ticker']). \
            sum()['value']. \
            fillna(0.0)
        target_values = self.target_weights * investment_value
        full_index = asset_holdings.index.union(target_values.index)
        trade_values = target_values.reindex(full_index).fillna(0) - \
            asset_holdings.reindex(full_index).fillna(0)
        buys = trade_values.where(trade_values > 0).dropna()
        sells = trade_values.where(trade_values < 0).dropna().to_dict()

        holdings = self.add_tax_info(holdings, date, self.tax_params)
        for asset, asset_sale in sells.items():
            asset_holdings = holdings[holdings['ticker'] == asset]
            shares_to_sell = -1 * asset_sale / \
                asset_holdings['current_price'].values[0]
            sells_by_lot = self.select_lots_for_sale(shares_to_sell,
                                                     asset_holdings)
            sells[asset] = sells_by_lot

        return {'buys': buys, 'sells': sells}

    @staticmethod
    def _empty_trades():
        """ Gives empty trades in the right format
        :return: Dictionary with buys and sells, but empty values
        """
        return {'buys': pd.Series(), 'sells': {}}

    @staticmethod
    def add_tax_info(lots: pd.DataFrame,
                     current_date: dt.date,
                     tax_params: Dict) -> pd.DataFrame:

        tax_info = {}
        for i in lots.index:
            lot_info = lots.loc[i]
            purchase_date = dt.date.fromisoformat(lot_info['purchase_date'])
            holding_period = (current_date - purchase_date).days
            if holding_period <= tax_params['lt_cutoff']:
                lot_rate = tax_params['income_rate']
            else:
                lot_rate = tax_params['lt_gains_rate']

            purchase_price = lot_info['purchase_price']
            gain = (lot_info['current_price'] / purchase_price - 1)
            effective_rate = gain * lot_rate

            tax_info[i] = pd.Series({'holding_period': holding_period,
                                     'applicable_rate': lot_rate,
                                     'pct_gain': gain,
                                     'effective_rate': effective_rate})

        tax_info = pd.DataFrame(tax_info).T

        return lots.join(tax_info)

    @staticmethod
    def select_lots_for_sale(shares_to_sell: float,
                             holdings: pd.DataFrame) -> Dict:
        """ Choose which lots to sell from based on tax burden

        :param shares_to_sell: number of shares of the asset to sell
        :param holdings: holdings for this asset only
        :return: dictionary keyed by lot date. values are dollar amounts to
        sell from each lot
        """

        holdings = holdings.reset_index(drop=True)
        order = holdings['effective_rate'].argsort().values
        shares_available = holdings['quantity'].copy()

        sells = {}
        while shares_to_sell > 0:
            current_best_lot = order[0]
            best_lot_date = holdings['purchase_date'][current_best_lot]
            if shares_available.iloc[current_best_lot] < shares_to_sell:
                sell_value = shares_available[current_best_lot] \
                             * holdings['current_price'][current_best_lot]
                shares_to_sell -= shares_available[current_best_lot]
                shares_available[current_best_lot] = 0
                order = order[1:]
            else:
                shares_available[current_best_lot] -= shares_to_sell
                sell_value = shares_to_sell * \
                    holdings['current_price'][current_best_lot]
                shares_to_sell = 0
            sells[best_lot_date] = sell_value

        return sells


class IntervalBasedRebalancer(SimpleRebalancer):

    def __init__(self,
                 target_weights: pd.Series,
                 rebalance_dates: List[dt.date],
                 tax_params: Dict):
        """

        :param target_weights: weights of the target portfolio
        :param rebalance_dates: list of dates on which the portfolio should
            be rebalanced
        :param tax_params: tax rates and long-term gains cutoff
        """
        super().__init__(target_weights, tax_params)
        self.rebalance_dates = set(rebalance_dates)

    def rebalance(self, date, holdings, investment_value):
        if date not in self.rebalance_dates:
            return self._empty_trades()

        return self.generate_complete_trades(date, holdings,
                                             investment_value)


class ThresholdBasedRebalancer(SimpleRebalancer):

    def __init__(self,
                 target_weights: pd.Series,
                 threshold_function: Callable,
                 tax_params: Dict):
        """ Rebalancer that trades all the way to the target when a
        trigger is satisfied

        :param target_weights: weights of the target portfolio
        :param threshold_function: callable object that takes the current
            and target weights of the portfolio, and returns a True or False
            value indicating whether the portfolio should be rebalanced
        :param tax_params: tax rates and long-term gains cutoff
        """
        super().__init__(target_weights, tax_params)
        self.threshold_function = threshold_function

    def rebalance(self, date, holdings, investment_value):
        current_weights = holdings[['ticker', 'value']] \
                              .groupby(['ticker']) \
                              .sum()['value'] \
                              / investment_value

        if not self.threshold_function(current_weights,
                                       self.target_weights):
            return self._empty_trades()

        return self.generate_complete_trades(date, holdings,
                                             investment_value)


class OptimizationBasedRebalancer(Rebalancer):

    def __init__(self,
                 target_weights: pd.Series,
                 objective: Objective,
                 constraints: List[Constraint]):
        """ Rebalancer that generates rebalancing trades by solving
        an optimization problem

        :param target_weights: weights of the target portfolio
        :param objective: objective for the optimization problem
        :param constraints: constraints for the optimization problem
        """
        super().__init__(target_weights)
        self.objective = objective
        self.constraints = constraints

    def rebalance(self, date, holdings, investment_value):
        target_port = self.target_weights * investment_value
        opt = RebalancingOpt(date,
                             target_port,
                             holdings,
                             self.constraints,
                             self.objective)
        opt.solve()
        trades = opt.get_trades()

        return trades
