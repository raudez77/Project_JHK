import pandas as pd
import numpy as np
import yfinance as yf

def DownloadCleanData (tickers_:str, start_ :str, end_: str)-> pd.DataFrame:
    """ Download and Clean Data"""

    # Step 1 Download Data
    tmp_data = yf.download(tickers = tickers_, start=start_, end=end_, progress=False, interval = "1d")

    # Step 2 : Clean Data
    tmp_data.dropna(axis=0, inplace=True)

    return tmp_data