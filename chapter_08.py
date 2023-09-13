from typing import List, Dict, Union

import cvxpy as cp
import numpy as np
import pandas as pd
import yfinance as yf


class Constraint:

    def generate_constraint(self, variables: Dict):
        """ Create the cvxpy Constraint

        :param variables: dictionary containing the cvxpy Variables for the
          problem
        :return: A cvxpy Constraint object representing the constraint
        """
        pass


class TrackingErrorConstraint(Constraint):

    def __init__(self,
                 asset_names: Union[List[str], pd.Index],
                 reference_weights: pd.Series,
                 sigma: pd.DataFrame,
                 upper_bound: float):
        """ Constraint on the tracking error between a subset of the
        portfolio and a set of target weights

        :param asset_names: Names of all assets in the problem
        :param reference_weights: Vector of target weights. Index should be
          a subset of asset_names
        :param sigma: Covariance matrix, indexed by asset_names
        :param upper_bound: Upper bound for the constraint, in units of
          volatility (standard deviation)
        """
        self.reference_weights = \
            reference_weights.reindex(asset_names).fillna(0)
        self.sigma = sigma
        self.upper_bound = upper_bound ** 2

    def generate_constraint(self, variables: Dict):
        w = variables['w']
        tv = cp.quad_form(w - self.reference_weights, self.sigma)
        return tv <= self.upper_bound


class VolatilityConstraint(TrackingErrorConstraint):

    def __init__(self,
                 asset_names: Union[List[str], pd.Index],
                 sigma: pd.DataFrame,
                 upper_bound: float):
        """ Constraint on the overall volatility of the portfolio

        :param asset_names: Names of all assets in the problem
        :param sigma: Covariance matrix, indexed by asset_names
        :param upper_bound: Upper bound for the constraint, in units of
          volatility (standard deviation)
        """

        zeros = pd.Series(np.zeros(len(asset_names)), asset_names)
        super(VolatilityConstraint, self).__init__(asset_names, zeros,
                                                   sigma, upper_bound)


class LinearConstraint(Constraint):

    def __init__(self,
                 asset_names: List[str],
                 coefs: pd.Series,
                 rhs: float,
                 direction: str):
        """
        Generic linear constraint, of the form

            coefs * w [vs] rhs
        |
        where [vs] can be <=, >=, or ==

        :param asset_names: Names of all assets in the problem
        :param coefs: Vector of coefficients, indexed by asset names. Can
          be a subset of all assets
        :param rhs: Right-hand side of the constraint
        :param direction: String starting with "<", ">", or "="
        """
        self.coefs = coefs.reindex(asset_names).fillna(0).values
        self.rhs = rhs
        self.direction = direction

    def generate_constraint(self, variables: Dict):
        w = variables['w']
        direction = self.direction
        if direction[0] == '<':
            return self.coefs.T @ w <= self.rhs
        elif direction[0] == '>':
            return self.coefs.T @ w >= self.rhs
        elif direction[0] == '=':
            return self.coefs.T @ w == self.rhs


class LongOnlyConstraint(Constraint):

    def __init__(self):
        """ Constraint to enforce all portfolio weights are non-negative
        """
        pass

    def generate_constraint(self, variables: Dict):
        return variables['w'] >= 0


class FullInvestmentConstraint(Constraint):

    def __init__(self):
        """ Constraint to enforce the sum of the portfolio weights is one
        """
        pass

    def generate_constraint(self, variables: Dict):
        return cp.sum(variables['w']) == 1.0


class GlobalMaxWeightConstraint(Constraint):
    def __init__(self, upper_bound: float):
        """ Constraint to enforce an upper bound on the magnitude of every
        asset in the portfolio

        :param upper_bound: Magnitude of every position will be constrained
          to be at most this value
        """
        self.upper_bound = upper_bound

    def generate_constraint(self, variables: Dict):
        return cp.norm_inf(variables['w']) <= self.upper_bound


class SubsetWeightConstraint(LinearConstraint):

    def __init__(self,
                 target_asset_name: str,
                 asset_names: List[str],
                 asset_subset_names: List[str],
                 rhs: float,
                 direction: str):
        """ Create a constraint of the form

          w_k [vs] b * sum_I(w_i)

          where [vs] can be >=, <=, or ==.
          This constraints the weight of asset k as a fraction of the total
          weight of assets in the set I.

        :param target_asset_name: Name of asset whose weight will be
          constrained
        :param asset_names: All asset names in the problem
        :param asset_subset_names: Target asset's weight will be constrained
          as a fraction of the total weigh in this set
        :param rhs: Bound for the constraint
        :param direction: String starting with "<", ">", or "="
        """
        coefs = pd.Series(-rhs, asset_subset_names)
        coefs[target_asset_name] += 1
        super(SubsetWeightConstraint, self).__init__(asset_names, coefs,
                                                     0, direction)


class MeanVarianceOpt:

    def __init__(self):
        self.asset_names = []
        self.variables = None
        self.prob = None

    @staticmethod
    def _generate_constraints(variables: Dict,
                              constraints: List[Constraint]):
        return [c.generate_constraint(variables) for c in constraints]

    def solve(self):
        self.prob.solve()

    def get_var(self, var_name: str):
        return pd.Series(self.variables[var_name].value, self.asset_names)


class MaxExpectedReturnOpt(MeanVarianceOpt):

    def __init__(self,
                 asset_names: Union[List[str], pd.Index],
                 constraints: List[Constraint],
                 ers: pd.Series):
        super().__init__()
        self.asset_names = asset_names
        variables = dict({'w': cp.Variable(len(ers))})

        cons = MeanVarianceOpt._generate_constraints(variables,
                                                     constraints)
        obj = cp.Maximize(ers.values.T @ variables['w'])
        self.variables = variables
        self.prob = cp.Problem(obj, cons)


def generate_subset_weight_constraints(asset_subset_names,
                                       all_asset_names,
                                       ref_weights,
                                       tolerance):

    ref_weights = ref_weights[asset_subset_names]
    ref_weights /= ref_weights.sum()
    cons = []
    for target_asset_name in asset_subset_names:
        ub = ref_weights[target_asset_name] + tolerance
        ub_con = SubsetWeightConstraint(target_asset_name,
                                        all_asset_names,
                                        asset_subset_names,
                                        ub,
                                        '<')
        lb = ref_weights[target_asset_name] - tolerance
        lb_con = SubsetWeightConstraint(target_asset_name,
                                        all_asset_names,
                                        asset_subset_names,
                                        lb,
                                        '>')
        cons.extend([ub_con, lb_con])

    return cons


def generate_frontier(ers: pd.Series, sigma: pd.DataFrame):

    asset_vols = np.sqrt(np.diag(sigma))
    target_vols = np.arange(np.min(np.floor(asset_vols * 100)) / 100,
                            np.max(asset_vols) + 0.005, 0.005)

    result = []
    for target_vol in target_vols:
        cons = [LongOnlyConstraint(),
                FullInvestmentConstraint(),
                VolatilityConstraint(ers.index, sigma, target_vol)]

        eq_bmk = pd.Series([.6, .3, .1], ['VTI', 'VEA', 'VWO'])
        subset_cons_eq = generate_subset_weight_constraints(eq_bmk.index,
                                                            ers.index,
                                                            eq_bmk, .20)
        cons.extend(subset_cons_eq)

        fi_bmk = pd.Series([.4, .4, .2], ['AGG', 'BNDX', 'EMB'])
        subset_cons_fi = generate_subset_weight_constraints(fi_bmk.index,
                                                            ers.index,
                                                            fi_bmk, .20)
        cons.extend(subset_cons_fi)

        o = MaxExpectedReturnOpt(ers.index, cons, ers)
        o.solve()
        weights = np.round(o.get_var('w'), 6)
        if np.any(np.isnan(weights)):
            continue

        risk = np.sqrt(weights @ sigma @ weights)
        er = weights @ ers
        if risk < (target_vol - .005):
            continue
        info = pd.Series([risk, er], ['Risk', 'ER'])
        result.append(pd.concat((info, weights)))

    return pd.concat(result, axis=1).T


def pull_etf_returns(tickers: List[str],
                     period: str = 'max') -> pd.DataFrame:
    rets = yf.download(tickers, period=period)['Adj Close'].pct_change()
    rets = rets.dropna(axis=0, how='any')[tickers]

    return rets


def get_default_inputs():
    tickers = ['VTI', 'VEA', 'VWO', 'AGG', 'BNDX', 'EMB']
    ers = pd.Series([.05, .05, .07, .03, .02, .04], tickers)
    sigma = np.array(
        [[0.0287, 0.0250, 0.0267, 0.0000, 0.0002, 0.0084],
         [0.0250, 0.0281, 0.0288, 0.0003, 0.0002, 0.0092],
         [0.0267, 0.0288, 0.0414, 0.0005, 0.0004, 0.0112],
         [0.0000, 0.0003, 0.0005, 0.0017, 0.0008, 0.0019],
         [0.0002, 0.0002, 0.0004, 0.0008, 0.0010, 0.0011],
         [0.0084, 0.0092, 0.0112, 0.0019, 0.0011, 0.0083]])
    sigma = pd.DataFrame(sigma, tickers, tickers)

    return ers, sigma
