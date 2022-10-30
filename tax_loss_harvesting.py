import datetime as dt
from typing import Tuple, Dict, List

import numpy as np
from optionprice import Option
import pandas as pd


def _sellable(lot: pd.Series, new_lots: pd.Series) -> pd.Series:
    """ Check whether a lot can be sold without creating a wash sale

    :param lot: Information about the tax lot
    :param new_lots: Series containing purchase dates of newly-purchased
        lots
    :return: List of two elements. First element is True if the lot can be
        sold without creating a wash sale. Second element is an array of
        the purchase dates of the lots (if any) that prevent selling.
    """

    idx = ['sellable', 'blocking_lots']
    if not lot['still_held']:
        return pd.Series([False, []], idx)

    if not lot['is_at_loss']:
        return pd.Series([True, []], idx)

    if lot['is_new'] and len(new_lots) > 1:
        blocking_lots = new_lots[new_lots != lot.purchase_date]
        return pd.Series([False, blocking_lots], idx)

    if not lot['is_new'] and len(new_lots) > 0:
        return pd.Series([False, new_lots], idx)

    return pd.Series([True, []], idx)


def _blocks_buying(lot: pd.Series, current_date: dt.date) -> bool:
    """ Check whether a lot prevents us from buying an asset, because it
    was recently sold at a loss

    :param lot: Information about the tax lot
    :param current_date: Current date
    :return: True if the lot has been recently sold at a loss, False if not
    """

    if lot['still_held']:
        return False
    how_long = (current_date - dt.date.fromisoformat(lot['sell_date'])).days

    return how_long <= 30 and lot['sell_price'] < lot['purchase_price']


def check_asset_for_restrictions(lots: pd.DataFrame,
                                 current_price: float,
                                 current_date: dt.date) -> pd.DataFrame:
    """ Check buying and selling eligibility for an asset

    :param lots: tax lots
    :param current_price: current asset price
    :param current_date: current date
    :return: Original lots with information about wash sales appended
    """

    lots = lots.copy()
    ws_start = (current_date - dt.timedelta(days=30)).strftime('%Y-%m-%d')
    lots['is_new'] = list(map(lambda x: x >= ws_start,
                              lots['purchase_date']))
    lots['is_at_loss'] = list(map(lambda x: current_price < x,
                                  lots['purchase_price']))
    lots['still_held'] = list(map(lambda x: x != x, lots['sell_date']))
    new_lots = lots[lots['is_new'] & lots['still_held']]['purchase_date']
    sellability = {i: _sellable(lots.loc[i], new_lots) for i in lots.index}
    lots = lots.join(pd.DataFrame(sellability).T)
    buy_blocks = list(map(lambda x: _blocks_buying(x[1], current_date),
                          lots.iterrows()))
    buy_blocks = pd.Series(buy_blocks, lots.index)
    lots['blocks_buy'] = buy_blocks
    lots.drop(['is_new', 'is_at_loss', 'still_held'], axis=1, inplace=True)

    return lots


def check_all_assets_for_restrictions(lots: pd.DataFrame,
                                      current_prices: pd.Series,
                                      current_date: dt.date):
    """ Check buying and selling eligibility for an asset

    :param lots: tax lots
    :param current_prices: current asset price
    :param current_date: current date
    :return: Tuple of two values:
        - First is a dataframe of lots with the 'can_sell' flag added
        - Second is a Dict, with an entry for each asset indicating blocks
    """
    tickers = current_prices.index
    results = []
    for ticker in tickers:
        asset_lots = lots[lots['ticker'] == ticker].copy()
        asset_res = \
            check_asset_for_restrictions(asset_lots,
                                         current_prices[ticker],
                                         current_date)
        results.append(asset_res)

    results = pd.concat(results)
    return results


def create_new_lots(buys, current_date, current_prices):
    new_lots = []
    for ticker, buy_amount in buys.items():
        idx = ['ticker', 'purchase_price', 'quantity', 'purchase_date',
               'sell_price', 'sell_date', 'wash_sale']
        new_lot = pd.Series([ticker, current_prices[ticker],
                             buy_amount / current_prices[ticker],
                             current_date.strftime('%Y-%m-%d'),
                             np.NaN, np.NaN, False],
                            index=idx)
        new_lots.append(new_lot)

    return pd.DataFrame(new_lots)


def adjust_lots_with_buys(lots: pd.DataFrame,
                          buys: Dict,
                          current_date: dt.date,
                          current_prices: pd.Series) -> Tuple:
    """

    :param lots: Current holdings, with information about wash sale
        restrictions
    :param buys: Dictionary of buy trades, in dollars
    :param current_date: The current date
    :param current_prices: Current asset prices
    :return: Three dataframes.
        First is lots corresponding to new buys.
        Second is closed lots from which we had to wash some but not all
        losses.
        Third is existing lots with adjustments for wash sales.
    """

    new_lots = create_new_lots(buys, current_date, current_prices)
    leftover_lots = []
    for new_idx, new_lot in new_lots.iterrows():
        ticker_lots = lots[lots['ticker'] == new_lot['ticker']]
        blocking_lots = ticker_lots[ticker_lots['blocks_buy']] \
            .sort_values(by='purchase_date')
        shares_to_wash = min(new_lot['quantity'],
                             blocking_lots['quantity'].sum())
        for i in blocking_lots.index:
            lot = lots.loc[i].copy()
            lot['wash_sale'] = True
            if shares_to_wash < lot['quantity']:
                leftover_lot = lots.loc[i].copy()
                leftover_lot['quantity'] -= shares_to_wash
                lot['quantity'] = shares_to_wash
                shares_to_wash = 0
                leftover_lots.append(leftover_lot)
            else:
                shares_to_wash -= lot['quantity']

            washed_loss = lot['quantity'] * (lot['purchase_price'] -
                                             lot['sell_price'])
            new_lots.loc[new_idx, 'purchase_price'] += washed_loss / \
                                                       new_lot['quantity']
            lots.loc[i] = lot

            if shares_to_wash == 0:
                break
    leftover_lots = pd.DataFrame(leftover_lots) \
        .drop(['sellable', 'blocking_lots', 'blocks_buy'],
              axis=1, errors='ignore')

    return new_lots, leftover_lots, lots


def close_lots(lots, sells, current_date, current_prices):
    """ Adjust lots for sales (no wash sale adjustments)

    :param lots: Current holdings, with information about wash sale
        restrictions
    :param sells: Dictionary of asset sales (in dollars)
    :param current_date: The current date
    :param current_prices: Current asset prices
    :return: Two data frames. First has still-held lots, and second has
        newly-closed lots.
    """
    lots = lots.copy()
    closed_lots = []
    for ticker, ticker_sells in sells.items():
        ticker_lots = lots[np.isnan(lots['sell_price']) *
                           lots['ticker'] == ticker]
        for purchase_date, sell_value in ticker_sells.items():
            i = ticker_lots['purchase_date'].tolist().index(purchase_date)
            i = ticker_lots.index[i]
            sold_lot = lots.loc[i]
            sold_shares = sell_value / current_prices[ticker]
            closed_lot = sold_lot.copy()
            idx = ['quantity', 'sell_date', 'sell_price', 'wash_sale']
            closed_lot[idx] = [sold_shares,
                               current_date.strftime('%Y-%m-%d'),
                               current_prices[ticker],
                               not closed_lot['sellable']]
            closed_lots.append(closed_lot)

            remaining_shares = sold_lot['quantity'] - sold_shares
            if remaining_shares > 0:
                remainder_lot = sold_lot.copy()
                remainder_lot['quantity'] = remaining_shares
                lots.loc[i] = remainder_lot
            else:
                lots = lots.drop(i)

    return pd.DataFrame(closed_lots, columns=lots.columns), lots


def adjust_for_wash_sales(closed_lots, lots):
    """ Adjust lots for wash sales caused by selling

    :param closed_lots: Lots that have just been sold
    :param lots: Remaining (still-open) lots
    :return: Lots with cost basis adjusted for any wash sales in the
        recent sales
    """
    wash_sales = closed_lots[closed_lots['wash_sale']]
    for _, sold_lot in wash_sales.iterrows():
        blocking_lots = sold_lot['blocking_lots'].sort_values()
        held_lots = blocking_lots.index.intersection(lots.index)
        washed_loss = sold_lot['quantity'] * (sold_lot['purchase_price'] -
                                              sold_lot['sell_price'])
        if len(held_lots):
            lots.loc[held_lots[0], 'purchase_price'] += washed_loss / \
                                                        lots.loc[held_lots[0], 'quantity']

    return lots


def adjust_lots_for_sells(lots: pd.DataFrame,
                          sells: Dict,
                          current_date: dt.date,
                          current_prices: pd.Series) -> Tuple:
    """ Adjust lots for sells

    :param lots: Current holdings, with information about wash sale
        restrictions
    :param sells: Dictionary of asset sales (in dollars)
    :param current_date: The current date
    :param current_prices: Current asset prices
    :return: Two data frames. First has still-held lots with any adjustments
        for wash sales. Second has newly-closed lots.
    """
    closed_lots, lots = close_lots(lots, sells, current_date,
                                   current_prices)
    lots = adjust_for_wash_sales(closed_lots, lots)

    return lots, closed_lots


def update_lots_with_trades(lots: pd.DataFrame,
                            buys: Dict,
                            sells: Dict,
                            current_date: dt.date,
                            current_prices: pd.Series) -> pd.DataFrame:
    """ Modify tax lots to reflect buys and sells, with any necessary
        adjustments for wash sales

    :param lots: Current holdings, with information about wash sale
        restrictions
    :param buys: Dictionary of asset buys (in dollars)
    :param sells: Dictionary of asset sales (in dollars)
    :param current_date: The current date
    :param current_prices: Current asset prices
    :return: Two data frames. First has still-held lots with any adjustments
        for wash sales. Second has newly-closed lots.
    """
    lots, closed_lots = adjust_lots_for_sells(lots, sells, current_date,
                                              current_prices)
    new_lots, leftover_lots, lots = adjust_lots_with_buys(lots, buys,
                                                          current_date,
                                                          current_prices)
    updated_lots = pd.concat((closed_lots, lots, leftover_lots, new_lots),
                             ignore_index=True)
    updated_lots.drop(['sellable', 'blocking_lots', 'blocks_buy'],
                      axis=1, inplace=True)

    return updated_lots


def evaluate_tcost(lot: pd.Series, current_price: float,
                   sell_spread: float, buy_spread: float,
                   commission_rate: float = 0):
    """ Evaluate a loss harvest based on the cost to execute it

    :param lot: A single tax lot
    :param current_price: Current price of the asset
    :param sell_spread: Estimated bid/ask spread on the asset being sold
    :param buy_spread: Estimated bid/ask spread on the asset being bought
    :param commission_rate: Commission rate for trades
    :return: Tax benefit less the cost of trading
    """
    trade_size = lot['quantity'] * current_price

    spread_cost = (buy_spread + sell_spread) * trade_size
    commission_cost = 2 * commission_rate * trade_size

    return lot['tax_benefit'] - (spread_cost + commission_cost)


def evaluate_opp_cost(lot: pd.Series, price: float, current_date: dt.date,
                      tax_params: Dict, sigma: float, risk_free: float,
                      div_yield: float):
    """ Evaluate a loss harvest based on the opportunity cost vs the tax
    benefit

    :param lot: A single tax lot
    :param price: Current asset price
    :param current_date: Current date
    :param tax_params: Dict with tax rates and cutoff
    :param sigma: Volatility of the asset
    :param risk_free: Risk-free rate
    :param div_yield: Dividend yield of the asset
    :return: Tax benefit less the opportunity cost of harvesting
    """
    purchase_date = dt.date.fromisoformat(lot['purchase_date'])
    holding_period = (current_date - purchase_date).days
    if holding_period >= tax_params['lt_cutoff']:
        rate_now = rate_later = tax_params['lt_rate']
    else:
        rate_now = tax_params['st_rate']
        rate_later = tax_params['lt_rate']
    price = float(price)

    o_full = Option(european=False, kind='put', s0=price, k=price,
                    t=30, sigma=sigma, r=risk_free, dv=div_yield)
    p_full = o_full.getPrice('BT')

    days_to_switch = tax_params['lt_cutoff'] - holding_period
    if 0 < days_to_switch <= 30:
        o_pre_switch = Option(european=False, kind='put', s0=price,
                              k=price, t=days_to_switch, sigma=sigma,
                              r=risk_free, dv=div_yield)
        p_til_switch = o_pre_switch.getPrice('BT')
        p_after_switch = p_full - p_til_switch

        opp_cost = rate_now * p_til_switch + rate_later * p_after_switch
    else:
        opp_cost = rate_now * p_full

    benefit = rate_now * (lot['purchase_price'] - price)

    return benefit - opp_cost


def find_eligible_replacements(etf_set, lots, current_date):
    """ Find all ETFs from a given set that for that would not create a
    wash sale if they were purchased today.

    :param etf_set: Set of potential "replacement" ETFs
    :param lots: Tax lots
    :param current_date: Current date
    :return: A set of tickers that are considered replacements for the
        provided ticker and have not been recently sold at a loss
    """
    lots = lots[lots['sell_date'] == lots['sell_date']]
    lots = lots[lots['ticker'].isin(etf_set)]
    lots = lots[lots['sell_price'] < lots['purchase_price']]
    hp = lots.apply(lambda x: (current_date -
                               dt.date.fromisoformat(x['sell_date'])).days,
                    axis=1)
    lots = lots.loc[hp[hp <= 30].index]

    return [x for x in etf_set if x not in lots['ticker'].values]


def evaluate_harvest(lot, lots, replacements, current_date, prices, tax_params,
                     sigma, risk_free, div_yield, spreads, tickers_selling):
    """ Given a lot trading at a loss, decide whether or not the lot
    ahould be harvested, and which ETF can replace the one currently held

    :param lot: Tax lot being considered for harvesting
    :param lots: All other tax lots
    :param replacements: Potential replacement ETFs for the one being
        considered for harvesting
    :param current_date: The current date
    :param prices: Prices of all ETFs
    :param tax_params: Tax rates and long-term/short-term cutoff
    :param sigma: volatility of the ETF considered for harvesting
    :param risk_free: Current risk-free rate
    :param div_yield: Dividend yield of the ETF considered for harvesting
    :param spreads: Assumed bid/ask spreads
    :param tickers_selling: Tickers being harvested already
    :return: The trade information, if possible and beneficial
    """
    replacements = \
        find_eligible_replacements(replacements, lots, current_date)
    replacements = [x for x in replacements if x not in tickers_selling]
    if len(replacements) == 0:
        return None
    ticker = lot['ticker']
    replacement = replacements[0]
    benefit_net_tcosts = evaluate_tcost(lot, prices[ticker],
                                        spreads[ticker],
                                        spreads[replacement])
    benefit_net_oppcost = evaluate_opp_cost(lot, prices[ticker],
                                            current_date,
                                            tax_params,
                                            sigma, risk_free, div_yield)

    if benefit_net_tcosts < 0 or benefit_net_oppcost < 0:
        return None

    value = lot['quantity'] * prices[ticker]
    res = pd.Series([ticker, lot['purchase_date'], value, replacement],
                    ['sell_ticker', 'purchase_date', 'value', 'buy_ticker'])
    return res


def get_replacement_set(ticker: str):
    """ Find potential replacement ETFs for a given ticker

    :param ticker: ETF being considered for a tax loss harvest
    :return: Set of tickers considered exchangeable with the input ETF
    """
    etf_sets = (('VTI', 'SCHB', 'ITOT', 'SPTM'),
                ('VEA', 'SCHF', 'IEFA'))

    for s in etf_sets:
        if ticker in s:
            s = list(s)
            s.remove(ticker)
            return s
    return []


def evaluate_all_harvests(lots, current_date, prices, tax_params, sigmas,
                          risk_free, div_yields, spreads):
    """

    :param lots: DataFrame of tax lots
    :param current_date: Date of harvest evaluation
    :param prices: ETF prices - should include all harvestable ETFs
    :param tax_params: Dict of tax parameters
    :param sigmas: Asset volatilities
    :param risk_free: Current risk-free rate
    :param div_yields: Asset dividend yields
    :param spreads: Bid/ask spreads
    :return: DataFrame with harvests to execute
    """
    all_lots = lots.copy()
    lots = lots[lots['sell_price'] != lots['sell_price']].copy()
    gains = (prices[lots['ticker']].values - lots['purchase_price'])
    lots['gain'] = lots['quantity'] * gains
    lots = lots[lots['gain'] < 0]
    lots['hp'] =\
        list(map(lambda d: (current_date - dt.date.fromisoformat(d)).days,
                 lots['purchase_date']))
    lots['tax_rate'] = tax_params['st_rate']
    idx = lots['hp'][lots['hp'] >= tax_params['lt_cutoff']].index
    lots.loc[idx, 'tax_rate'] = tax_params['lt_rate']
    lots['tax_benefit'] = -lots['gain'] * lots['tax_rate']
    lots = lots.sort_values(by='tax_benefit', ascending=False)

    harvests, buying, selling = [], [], []
    for i, lot in lots.iterrows():
        ticker = lot['ticker']
        replacements = get_replacement_set(ticker)
        result = evaluate_harvest(lot, all_lots, replacements, current_date,
                                  prices, tax_params, sigmas[ticker],
                                  risk_free, div_yields[ticker], spreads,
                                  selling)
        if result is not None and ticker not in buying:
            harvests.append(result)
            buying.append(result['buy_ticker'])
            selling.append(ticker)

    harvests = pd.DataFrame(harvests)
    return harvests

