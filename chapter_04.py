from typing import Iterable, Union

from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import yfinance as yf


def calculate_cost_comparison(tickers: Iterable[str],
                              holding_period: float) -> pd.DataFrame:
    """ Create a table showing the various costs of holding a selection of
    ETFs over a specified time period, as well as the total cost of
    ownership.

    :param tickers: Tickers of the ETFs to compute cost of ownership for
    :param holding_period: Expected holding period (in years)
    :return: A pandas DataFrame showing the various costs of holding an ETF
    over the specified time period
    """
    inputs = get_cost_inputs(tickers)
    return inputs.apply(calculate_costs_for_ticker, 0,
                        holding_period=holding_period)


def calculate_costs_for_ticker(inputs: pd.Series,
                               holding_period: float) -> pd.Series:
    """ Calculate the various costs of ownership for a single ETF

    :param inputs: Series containing the input data needed to compute the
      various costs. Should contain values for "fee" (annual expense ratio),
      "spread" (expected bid/ask spread), "min_prem", and "max_prem" (the
      range or premium/discount values over the previous year)
    :param holding_period: Expected holding period, in years
    :return: Series containing the various cost components as well as the
    total
    """
    position_size = 100_000
    holding_cost = inputs['fee'] * position_size * holding_period
    spread_cost = inputs['spread'] * position_size / inputs['price']

    total = spread_cost + holding_cost

    return pd.Series([holding_cost, spread_cost, total],
                     ['holding', 'spread', 'total'])


def get_cost_inputs(tickers: Iterable[str]) -> pd.DataFrame:
    """ Retrieve the inputs needed to calculate cost of ownership

    :param tickers: Ticker symbols of the ETFs to compute costs for
    :return: A DataFrame containing all the necessary inputs to compute
    costs
      of ownership
    """
    dr = webdriver.Safari()
    try:
        info = {t: get_ticker_cost_inputs(t, dr) for t in tickers}
    finally:
        dr.close()

    return pd.DataFrame(info)


def get_ticker_cost_inputs(ticker: str, dr: webdriver) -> pd.Series:
    """ Retrieve necessary inputs for computing cost of ownership for a
    single ETF

    :param ticker: Symbol of the ETF to get cost inputs for
    :param dr: A Selenium webdriver object
    :return: A Series with all necessary inputs
    """
    url = f'https://www.etf.com/{ticker}'
    dr.get(url)

    divs = BeautifulSoup(dr.page_source).find_all('div')
    fee = find_value("Expense Ratio", divs)
    spread = find_value("Average Spread ($)", divs)

    fee = parse_pct(fee)
    spread = parse_dollars(spread)

    t = yf.Ticker(ticker)
    price = t.info['regularMarketPrice']

    return pd.Series([fee, spread, price], ['fee', 'spread', 'price'])


def find_value(key_str: str, divs_list: Iterable) -> Union[str, None]:
    """ Search a list if divs for once with a specific label attribute, and
    return the data

    :param key_str: Label to match
    :param divs_list: A list of divs, as parsed by BeautifulSoup
    :return: The data from the first div with a label matching the one
    specified
    """
    for div in divs_list:
        lab = div.find('label')
        if lab is None:
            continue
        if lab.getText().strip("\n") == key_str:
            span = div.find('span')
            if span is not None:
                return span.contents[0]
    return None


def parse_pct(val: str) -> float:
    """ Convert a string representing a percentage value into a floating
    point decimal value

    :param val: String representing a percentage, e.g. "0.25%"
    :return: Value of input as a decimal
    """
    return float(val.strip("%")) / 100.0


def parse_dollars(val: str) -> float:
    """ Convert a string representing a dollar value into a a floating
    point decimal value

    :param val: String representing a dollar value, e.g. "$0.75"
    :return: Value of input as a decimal
    """
    return float(val.strip("$"))
