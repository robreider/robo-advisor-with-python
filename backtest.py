import datetime as dt
from typing import Dict, List, Union

import pandas as pd
import yfinance as yf

from rebalancing import Rebalancer


def get_dividends(assets: List[str]) -> Dict:
    """ Get all the historical dividends for a set of assets

    :param assets: list of tickers
    :return: dictionary keyed by ticker, with a series of dividend values
    """
    div_dict = {}
    for ticker in assets:
        t = yf.Ticker(ticker)
        divs = t.dividends
        divs.index = pd.Index(map(lambda x: x.date(), divs.index))
        div_dict[ticker] = divs

    return div_dict


def get_prices(assets: List[str],
               start_date: str,
               end_date: str) -> pd.DataFrame:
    """ Retrieve historical prices for given assets

    :param assets: list of tickers
    :param start_date: first date to get prices for
    :param end_date: last date to get prices for
    :return: DataFrame of prices - one asset per column, one day per row
    """
    prices = yf.download(assets, start_date, end_date)['Close']
    prices.index = pd.Index(map(lambda x: x.date(), prices.index))
    if isinstance(prices, pd.Series):
        prices = pd.DataFrame(prices)
        prices.columns = assets

    return prices


class BacktestParams:

    def __init__(self,
                 target_weights: pd.Series,
                 start_date: str,
                 end_date: str,
                 starting_investment: float,
                 cash_buffer: float,
                 tax_params: Dict,
                 spreads: Union[pd.Series, float],
                 rebalancer: Rebalancer):
        self.target_weights = target_weights
        self.start_date = start_date
        self.end_date = end_date
        self.starting_investment = starting_investment
        self.cash_buffer = cash_buffer
        self.rebalancer = rebalancer
        self.spreads = spreads
        self.tax_params = tax_params


class Backtest:

    def __init__(self,
                 params: BacktestParams,
                 prices: pd.DataFrame,
                 dividends: Dict):
        self.assets = list(params.target_weights.index.values)
        self.prices = prices
        self.dividends = dividends
        self.params = params

    def run(self):

        params = self.params
        cash = params.starting_investment
        holdings = pd.DataFrame({'ticker': [],
                                 'value': [],
                                 'quantity': []})
        rebalancer = params.rebalancer
        prices = self.prices
        dividends = self.dividends
        daily_info, weights_df, in_weights_df = {}, {}, {}

        for date in self.prices.index:
            print(date)
            current_prices = prices.loc[date]
            holdings = self.mark_to_market(holdings, current_prices)
            divs = self.calc_dividend_income(date, holdings, dividends)
            cash += divs
            portfolio_value = holdings['value'].sum() + cash
            in_weights_df[date] = \
                self.weights_from_holdings(holdings,
                                           portfolio_value,
                                           self.assets)

            investment_value = portfolio_value * (1 - params.cash_buffer)
            trades = rebalancer.rebalance(date, holdings,
                                          investment_value)
            trade_prices = self.calc_trade_prices(current_prices,
                                                  params.spreads)
            holdings, weights, info = \
                self.get_current_data(date, holdings, cash, current_prices,
                                      trades, trade_prices,
                                      params.tax_params)
            info['dividends'] = divs / info['portfolio_value']
            daily_info[date] = info
            weights_df[date] = weights
            cash = info['cash']

        weights_df = pd.DataFrame(weights_df).fillna(0.0).T
        in_weights_df = pd.DataFrame(in_weights_df).fillna(0.0).T
        daily_info = pd.DataFrame(daily_info).T

        return daily_info, weights_df, in_weights_df, holdings

    @staticmethod
    def mark_to_market(holdings: pd.DataFrame,
                       current_prices: pd.Series) -> pd.DataFrame:
        """ Update holdings values with current prices

        :param holdings: holdings information, including share quantity,
            price, ticker
        :param current_prices: current asset prices
        :return: data frame of the same shape as the input, with the
            price per share and total value updated to reflect the current
            asset prices
        """
        holdings['current_price'] = \
            current_prices[holdings['ticker']].values
        holdings['value'] = holdings['current_price'] * holdings['quantity']

        return holdings

    @staticmethod
    def weights_from_holdings(holdings: pd.DataFrame,
                              portfolio_value: float,
                              assets: List[str]) -> pd.Series:
        """ Calculate weights of a portfolio

        :param holdings: holdings information, including share quantity,
            price, ticker
        :param portfolio_value: value of holdings and cash
        :param assets: all assets to calculate weights for
        :return: Series containing current portfolio weights
        """
        weights = holdings[['ticker', 'value']]. \
            groupby(['ticker']). \
            sum()['value']. \
            reindex(assets). \
            fillna(0.0) / \
            portfolio_value

        return weights

    @staticmethod
    def calc_dividend_income(date: dt.date,
                             holdings: pd.DataFrame,
                             dividends: Dict) -> float:
        """ Calculate how much dividend cash the portfolio generated today

        :param date: current data
        :param holdings: current portfolio holdings
        :param dividends: full historical dividend information
        :return: total dividend income for the day
        """

        if not holdings.shape[0]:
            return 0.0

        shares_by_asset = holdings[['ticker', 'quantity']]. \
            groupby(['ticker']). \
            sum()['quantity']

        div_income = 0.0
        assets = set(dividends.keys()). \
            intersection(set(shares_by_asset.index))
        for asset in assets:
            try:
                asset_div = dividends[asset][date]
            except KeyError:
                asset_div = 0.0
            div_income += shares_by_asset[asset] * asset_div

        return div_income

    @staticmethod
    def calculate_tax(purchase_price: float,
                      sell_price: float,
                      quantity: float,
                      purchase_date: dt.date,
                      sell_date: dt.date,
                      tax_params: Dict) -> float:
        """ Calculate tax due to a sale

        :param purchase_price: per-share purchase price of the asset
        :param sell_price: per-share sale price
        :param quantity: number of shares old
        :param purchase_date: date shares were purchased
        :param sell_date: date shares are being sold
        :param tax_params: tax rates and cutoff for long-term gains
        :return: tax owed due to the sale. positive value means paying tax.
        """
        holding_period = (sell_date - purchase_date).days
        if holding_period <= tax_params['lt_cutoff']:
            tax_rate = tax_params['lt_gains_rate']
        else:
            tax_rate = tax_params['income_rate']

        return (sell_price - purchase_price) * quantity * tax_rate

    @staticmethod
    def calc_trade_prices(current_prices: pd.Series,
                          spreads: Union[float, pd.Series]):
        """ Calculate prices for buys and sells, accounting for bid/ask
        spreads

        :param current_prices: asset prices for the day
        :param spreads: assumed bid/ask spreads, expressed as percentages
            of the prices
        :return: dictionary with assumed prices for buys and sells
        """
        assets = current_prices.index
        if not isinstance(spreads, pd.Series):
            spreads = pd.Series(spreads, assets)
        spreads = (spreads[assets] * current_prices).clip(lower=0.01)
        buy_prices = current_prices + spreads / 2
        sell_prices = current_prices - spreads / 2

        return {'buy': buy_prices, 'sell': sell_prices}

    @staticmethod
    def get_current_data(date: dt.date,
                         holdings: pd.DataFrame,
                         cash: float,
                         prices: pd.Series,
                         trades: Dict,
                         trade_prices: Dict,
                         tax_params: Dict) -> tuple:
        """ Current portfolio information after applying trades

        :param date: current date
        :param holdings: holdings information, including share quantity,
            price, ticker
        :param cash: amount of uninvested cash before any trading
        :param prices: asset prices
        :param trades: details on buy and sell trades
        :param trade_prices: assumed transaction prices for buys and sells
        :param tax_params: tax rates and holding period for long-term gains
        :return: tuple with the following items:
            - DataFrame of updated (after trading) holdings
            - Series containing the current portfolio weights
            - Series with some current information on the portfolio and
            trading
        """

        buys = trades['buys'][trades['buys'] > 0]
        buy_shares = (buys / prices[buys.index]).round(2)
        buy_prices = trade_prices['buy'][buys.index]

        buys = pd.DataFrame({'ticker': buys.index,
                             'purchase_price': buy_prices,
                             'current_price': prices[buys.index],
                             'quantity': buy_shares.values})
        buys = buys[buys['quantity'] > 0]
        buys['purchase_date'] = date.isoformat()
        buys['value'] = buys['quantity'] * buys['current_price']
        spread_costs = (buy_prices * buy_shares).sum() - buys['value'].sum()
        total_buy = (buys['quantity'] * buys['purchase_price']).sum()

        sells = trades['sells']
        total_sell, total_tax = 0, 0
        for i in holdings.index:
            asset = holdings['ticker'][i]
            purchase_date = holdings['purchase_date'][i]
            asset_sells = sells.get(asset, {})
            lot_sale = asset_sells.get(purchase_date, 0)
            if lot_sale == 0:
                continue
            shares_sold = lot_sale / prices[asset]
            holdings.loc[i, 'quantity'] -= shares_sold
            purchase_date = dt.date.fromisoformat(purchase_date)
            sell_price = trade_prices['sell'][asset]
            spread_costs += shares_sold * (prices[asset] - sell_price)
            tax = Backtest.calculate_tax(holdings['purchase_price'][i],
                                         sell_price, shares_sold,
                                         purchase_date, date,
                                         tax_params)
            total_sell += sell_price * shares_sold
            total_tax += tax

        holdings = pd.concat([holdings, buys], ignore_index=True)
        holdings = holdings[holdings['quantity'] > 0]
        holdings['value'] = holdings['quantity'] * holdings['current_price']
        print(holdings.shape)

        cash += (total_sell - total_buy)
        portfolio_value = holdings['value'].sum() + cash
        assets = list(prices.index)
        current_weights = Backtest.weights_from_holdings(holdings,
                                                         portfolio_value,
                                                         assets)
        turnover = (total_sell + total_buy) / portfolio_value
        current_info = {'portfolio_value': portfolio_value,
                        'cash': cash,
                        'turnover': turnover,
                        'tax': total_tax / portfolio_value,
                        'spread_costs': spread_costs / portfolio_value}

        return holdings, current_weights, pd.Series(current_info)
