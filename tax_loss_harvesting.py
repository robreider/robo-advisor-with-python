import datetime as dt
from typing import Tuple, Dict, List, Union

import numpy as np
from optionprice import Option
import pandas as pd


def sellable(lot: pd.Series, new_lots: pd.Series) -> pd.Series:
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
        return pd.Series([False, pd.Series(dtype='str')], idx)

    if not lot['is_at_loss']:
        return pd.Series([True, pd.Series(dtype='str')], idx)

    if lot['is_new'] and len(new_lots) > 1:
        blocking_lots = new_lots[new_lots != lot.purchase_date]
        return pd.Series([False, blocking_lots], idx)

    if not lot['is_new'] and len(new_lots) > 0:
        return pd.Series([False, new_lots], idx)

    return pd.Series([True, pd.Series(dtype='str')], idx)


def blocks_buying(lot: pd.Series, current_date: dt.date) -> bool:
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
    sellability = {i: sellable(lots.loc[i], new_lots) for i in lots.index}
    lots = lots.join(pd.DataFrame(sellability).T)
    buy_blocks = list(map(lambda x: blocks_buying(x[1], current_date),
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
    tickers = lots['ticker'].unique()
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


def create_new_lots(buys: Dict, current_date: dt.date,
                    current_prices: pd.Series):
    """ Create new tax lots from buy trades

    :param buys: New purchases
    :param current_date: The current date
    :param current_prices: Current asset prices
    :return: A data frame holding tax lot information for the purchases
    """
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


def update_lots_with_buys(lots: pd.DataFrame,
                          buys: Dict,
                          current_date: dt.date,
                          current_prices: pd.Series) -> Tuple:
    """ Update existing lots with buys, adjusting cost basis for wash sales
    where necessary

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


def update_lot(lot: pd.Series, idx: List[str], values: List):
    """ Helper function to create a new lot by updating an existing one

    :param lot: Series representing a tax lot
    :param idx: List of indices to put the replacement data
    :param values: replacement data
    :return: A new lot with the replacement data in the right places
    """
    new_lot = lot.copy()
    new_lot[idx] = values
    return new_lot


def close_unblocked_lots(lots: pd.DataFrame, sells: Dict,
                         current_date: dt.date, current_prices: pd.Series):
    """ Update tax lots for sales that don't create any wash sales

    :param lots: DataFrame of tax lots
    :param sells: Dictionary of sells, by asset and purchase date
    :param current_date: The current date
    :param current_prices: Current asset prices
    :return: Three DataFrames. First has closed tax lots. Second has
        still-open lots. Third has sells that have not been accounted for
        yet (because they could create washes)
    """
    lots = lots.copy()
    closed_lots = []
    remaining_sells = {k: {} for k in sells}
    for ticker, ticker_sells in sells.items():
        ticker_lots = lots[np.isnan(lots['sell_price']) *
                           lots['ticker'] == ticker]
        for purchase_date, sell_value in ticker_sells.items():
            i = ticker_lots['purchase_date'].tolist().index(purchase_date)
            i = ticker_lots.index[i]
            sold_lot = lots.loc[i]
            if len(sold_lot['blocking_lots']):
                remaining_sells[ticker][sold_lot['purchase_date']] = \
                    sell_value
                continue
            sold_shares = sell_value / current_prices[ticker]
            idx = ['quantity', 'sell_date', 'sell_price', 'wash_sale']
            val = [sold_shares, current_date.strftime('%Y-%m-%d'),
                   current_prices[ticker], not sold_lot['sellable']]
            closed_lot = update_lot(sold_lot, idx, val)
            closed_lots.append(closed_lot)

            remaining_shares = sold_lot['quantity'] - sold_shares
            if remaining_shares > 0:
                remainder_lot = sold_lot.copy()
                remainder_lot['quantity'] = remaining_shares
                lots.loc[i] = remainder_lot
            else:
                lots = lots.drop(i)
    closed_lots = pd.DataFrame(closed_lots, columns=lots.columns)

    return closed_lots, lots, remaining_sells


def calculate_remaining_quantities(lots: pd.DataFrame, sells: Dict,
                                   current_prices: pd.Series):
    """ Calculate the number of shares remaining in tax lots after any
    sales

    :param lots: DataFrame of tax lots
    :param sells: Dictionary of sells, by asset and purchase date
    :param current_prices: Current asset prices
    :return: Copy of the tax lots with a new column showing the number of
        shares that remain AFTER sales
    """
    lots = lots.copy()
    remaining_quantities = pd.Series(0, lots.index)
    for (i, lot) in lots.iterrows():
        if not np.isnan(lot['sell_date']):
            continue
        ticker = lot['ticker']
        ticker_sells = sells.get(ticker, {})
        purchase_date = lot['purchase_date']
        lot_sale = ticker_sells.get(purchase_date, 0)
        remaining_quantities[i] = \
            lot['quantity'] - lot_sale / current_prices[ticker]
    lots['remaining_quantity'] = remaining_quantities

    return lots


def update_ticker_lots_with_wash_sells(lots, sells, current_date,
                                       current_price):
    """ Adjust lots for a single ticker for sales that could create washes

    :param lots: Current holdings, with information about wash sale
        restrictions
    :param sells: Dictionary of asset sales (in dollars)
    :param current_date: The current date
    :param current_price: Current asset price
    :return: Two data frames. First has newly closed lots, and second has
        lots that are still open
    """
    lots = lots.copy()
    closed_lots, adjustments = [], []
    blocking_lots = lots['blocking_lots']
    idx = np.unique(np.concatenate([x.index.values for x in blocking_lots]))
    blocking_shares = lots.loc[pd.Index(idx), 'remaining_quantity']
    date_str = current_date.strftime('%Y-%m-%d')
    for purchase_date, sell_value in sells.items():
        i = lots.index[lots['purchase_date'].tolist().index(purchase_date)]
        sold_lot = lots.loc[i]
        shares_sold = sell_value / current_price
        blocking_lots = sold_lot['blocking_lots'].sort_values()
        blocks_idx = blocking_lots.index
        shares_washed = \
            min(shares_sold, blocking_shares[blocks_idx].sum())
        loss_to_wash = shares_washed * \
            (sold_lot['purchase_price'] - current_price)

        idx = ['quantity', 'sell_date', 'sell_price', 'wash_sale']
        if shares_washed > 0:
            val = [shares_washed, date_str, current_price, True]
            closed_lots.append(update_lot(sold_lot, idx, val))
            adjustments.append((blocking_lots, loss_to_wash))
            shares = blocking_shares[blocks_idx].cumsum() - shares_washed
            blocking_shares[blocks_idx] = np.clip(shares, 0, np.Inf)

        if shares_sold > shares_washed:
            val = [shares_sold - shares_washed, date_str, current_price,
                   False]
            closed_lots.append(update_lot(sold_lot, idx, val))

        if sold_lot['quantity'] > shares_sold:
            remainder_lot = sold_lot.copy()
            remainder_lot['quantity'] -= shares_sold
            lots.loc[i] = remainder_lot
        else:
            lots.drop(i, axis='index', inplace=True)

    for blocking_lots, wash_amount in adjustments:
        for purchase_date in blocking_lots.values:
            idx = lots[lots['purchase_date'] == purchase_date].index
            if len(idx) == 0:
                continue
            adj_per_share = wash_amount / lots.loc[idx[0], 'quantity']
            lots.loc[idx[0], 'purchase_price'] += adj_per_share

    return pd.DataFrame(closed_lots), lots


def update_with_wash_sells(lots, sells, current_date, current_prices):
    """ Adjust lots for sales (no wash sale adjustments)

    :param lots: Current holdings, with information about wash sale
        restrictions
    :param sells: Dictionary of asset sales (in dollars)
    :param current_date: The current date
    :param current_prices: Current asset prices
    :return: Two data frames. First has still-held lots, and second has
        newly-closed lots.
    """
    lots = calculate_remaining_quantities(lots, sells, current_prices)
    closed_lots, open_lots = [], []
    for ticker, ticker_sells in sells.items():
        ticker_lots = lots[lots['ticker'] == ticker]
        price = current_prices[ticker]
        ticker_res = \
            update_ticker_lots_with_wash_sells(ticker_lots, ticker_sells,
                                               current_date, price)
        closed_lots.append(ticker_res[0])
        open_lots.append(ticker_res[1])

    return pd.concat(closed_lots), pd.concat(open_lots)


def update_lots_for_sells(lots: pd.DataFrame,
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
    unblocked_res = \
        close_unblocked_lots(lots, sells, current_date, current_prices)
    updates = update_with_wash_sells(unblocked_res[1], unblocked_res[2],
                                     current_date, current_prices)

    closed_lots = pd.concat((unblocked_res[0], updates[0]))

    return updates[1], closed_lots


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
    lots, closed_lots = update_lots_for_sells(lots, sells, current_date,
                                              current_prices)
    new_lots, leftover_lots, lots = update_lots_with_buys(lots, buys,
                                                          current_date,
                                                          current_prices)
    updated_lots = pd.concat((closed_lots, lots, leftover_lots, new_lots),
                             ignore_index=True)
    updated_lots.drop(['sellable', 'blocking_lots', 'blocks_buy',
                       'remaining_quantity'], axis=1, inplace=True)

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


def evaluate_harvest(lot: pd.Series, replacement: str,
                     current_date: dt.date, prices: pd.Series,
                     tax_params: Dict, sigma: float, risk_free: float,
                     div_yield: float, spreads: pd.Series):
    """ Given a lot trading at a loss, decide whether or not the lot
    should be harvested

    :param lot: Tax lot being considered for harvesting
    :param replacement: ETF to buy as a replacement
    :param current_date: The current date
    :param prices: Prices of all ETFs
    :param tax_params: Tax rates and long-term/short-term cutoff
    :param sigma: volatility of the ETF considered for harvesting
    :param risk_free: Current risk-free rate
    :param div_yield: Dividend yield of the ETF considered for harvesting
    :param spreads: Assumed bid/ask spreads
    :return: Harvest trade information
    """
    ticker = lot['ticker']
    benefit_net_tcosts = evaluate_tcost(lot, prices[ticker],
                                        spreads[ticker],
                                        spreads[replacement])
    benefit_net_oppcost = evaluate_opp_cost(lot, prices[ticker],
                                            current_date,
                                            tax_params,
                                            sigma, risk_free, div_yield)
    if min(benefit_net_tcosts, benefit_net_oppcost) < 0:
        return None

    value = lot['quantity'] * prices[ticker]
    return pd.Series([ticker, lot['purchase_date'], value, replacement],
                     ['ticker', 'purchase_date', 'amount', 'replacement'])


def evaluate_harvests_for_etf_set(lots, current_date, prices, tax_params,
                                  sigmas, risk_free, div_yields,
                                  spreads, etf_set):
    """ Decide whether or not to harvest eligible lots from holdings
    belonging to one set of exchangeable ETFs

    :param lots: DataFrame of tax lots for ETFs from one TLH set
    :param current_date: Date of harvest evaluation
    :param prices: ETF prices - should include all harvestable ETFs
    :param tax_params: Dict of tax parameters
    :param sigmas: Asset volatilities
    :param risk_free: Current risk-free rate
    :param div_yields: Asset dividend yields
    :param spreads: Bid/ask spreads
    :param etf_set: All ETFs in this ETF set
    :return: DataFrame with harvests to execute
    """
    all_lots = check_all_assets_for_restrictions(lots, prices,
                                                 current_date)
    lots = all_lots[all_lots['sellable']]
    gains = (prices[lots['ticker']].values - lots['purchase_price'])
    lots['gain'] = lots['quantity'] * gains
    lots = lots[lots['gain'] < 0]
    lots['hp'] = \
        list(map(lambda d: (current_date - dt.date.fromisoformat(d)).days,
                 lots['purchase_date']))
    lots['tax_rate'] = tax_params['st_rate']
    lt_lots = lots['hp'][lots['hp'] >= tax_params['lt_cutoff']].index
    lots.loc[lt_lots, 'tax_rate'] = tax_params['lt_rate']
    lots['tax_benefit'] = -lots['gain'] * lots['tax_rate']
    lots = lots.sort_values(by='tax_benefit', ascending=False)

    tickers_by_benefit = lots.groupby('ticker')['tax_benefit'] \
        .sum().sort_values().index.values
    non_buyable = all_lots[all_lots['blocks_buy']]['ticker'].values
    replacements = [_ for _ in etf_set if _ not in tickers_by_benefit
                    and _ not in non_buyable]
    replacements.extend(tickers_by_benefit)

    harvests, buying, selling = {}, [], []
    for i, lot in lots.iterrows():
        ticker = lot['ticker']
        lot_replacements = [_ for _ in replacements if _ != ticker
                            and _ not in selling]
        if ticker in buying or len(lot_replacements) == 0:
            continue
        replacement = lot_replacements[0]
        result = evaluate_harvest(lot, replacement, current_date,
                                  prices, tax_params, sigmas[ticker],
                                  risk_free, div_yields[ticker], spreads)
        if result is not None:
            harvests[i] = result
            buying.append(replacement)
            selling.append(ticker)

    return pd.DataFrame(harvests).T


def evaluate_all_harvests(lots: pd.DataFrame, current_date: dt.date,
                          prices: pd.Series, tax_params: Dict,
                          sigmas: pd.Series, risk_free: float,
                          div_yields: pd.Series, spreads: pd.Series,
                          etf_sets: Union[Tuple, List]):
    """ Generate harvesting trades for all currently-held assets

    :param lots: DataFrame of tax lots
    :param current_date: Date of harvest evaluation
    :param prices: ETF prices - should include all harvestable ETFs
    :param tax_params: Dict of tax parameters
    :param sigmas: Asset volatilities
    :param risk_free: Current risk-free rate
    :param div_yields: Asset dividend yields
    :param spreads: Bid/ask spreads
    :param etf_sets: Sets of exchangeable ETFs
    :return: DataFrame with harvests to execute
    """
    harvests = []
    for etf_set in etf_sets:
        set_lots = lots[lots['ticker'].isin(etf_set)]
        if len(set_lots) == 0:
            continue
        set_harvests = \
            evaluate_harvests_for_etf_set(set_lots, current_date, prices,
                                          tax_params, sigmas, risk_free,
                                          div_yields, spreads, etf_set)
        if len(set_harvests) > 0:
            harvests.append(set_harvests)

    return pd.concat(harvests, ignore_index=True)
