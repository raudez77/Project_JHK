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

import sys
sys.path.append("../")
import pandas as pd
import numpy as np 
import re
import json
import yfinance as yf 
import warnings
import plotly.graph_objects as go 
from typing import List, Optional, Union, Type
# from Config_.config import DATA_DIR, CREDENTIALS_DIR
from sklearn import covariance, manifold, cluster
import cmasher as cmr
import cmasher
import plotly.graph_objects as go
warnings.filterwarnings("ignore")




# def LoadCredentials (path:str) -> json:
#     with open(CREDENTIALS_DIR, "r") as json_file:
#         key_ = json.loads(json_file.read())
#         return key_


def SelectedTickers(group:str, tickers_:list[str], start:str, end:str, return_df:bool=False)-> pd.DataFrame:
    """Download given Tickers and saved Jsonl File
    
    Parameters:
    -----------------------------------------------

    Args:
        group: Name for Cluster, 
        ticker: List of Tickers,
        start : "yyyy-mm-dd" Starting day to check stocks
        end : "yyyy-mm-dd" Ending day to check stocks
        return_df: if True, DataFrame is return (for debugging purpose only)
        
    """

    # Save Information
    CRITERIA, SAVE_AT = "Adj Close" , str(DATA_DIR/ str(group + ".json"))

    # Downloading 
    stocks_ = yf.download(tickers = tickers_, start=start, end=end)[CRITERIA]

    if len(stocks_.shape) == 1: #yahoo finance return a pd.Series if only one ticker
        tmp_dict = {tickers_[0]:stocks_.to_dict()} # transform to dict

        # Saving 
        pd.DataFrame.from_dict(tmp_dict).to_json(SAVE_AT)
    
    else:
        # Checking Stocks
        stocks_.to_json(SAVE_AT)

    if return_df:
        return stocks_


def CreateGroups (group:str, tickers:any, start:str, end:str, return_df:bool=True) -> list:
    """ PreProcess the Text, Return group and Tickers
    
    Args:
        group: Group Name to be assigned to the given Tickers
        tickers: Ticker Symbols
        start : "yyyy-mm-dd" Starting day to check stocks
        end : "yyyy-mm-dd" Ending day to check stocks
        return_df: if True, DataFrame is return (for debugging purpose only)
    
    """
    # Validate input arguments
    tmp_group = group.split()[0] # Validating Groups
    tmp_ticker = re.findall(r"\w{3,4}", str(tickers)) # No matter the input it will extract only 4letters

    # Downloading Tickers
    tickers_ = SelectedTickers(
        group = tmp_group,
        tickers_ = tmp_ticker,
        start= start,
        end = end , 
        return_df = return_df)

    print (f"Cluster: {tmp_group} , Downloading : {tmp_ticker}")
    return tickers_

class CalculateRatios (pd.DataFrame):
    def __init__(self, data:pd.DataFrame, IndexFund :str)->None:
        """ Calculate Ratio , Disperision and ZSccore
        
        Parameters:
        ---------------------------------------------
        data: must be pandas data frame with index as date time
        IndexFund: string, Index fund to be compared with other tickers (must be in the dataframe)
        """
        super(CalculateRatios,self).__init__()
        self.data_ = data
        self.IndexFund_ = IndexFund
        self.WindowsRollUp = None

        # Create Dict
        self._check_data()
        self._create_dict()

    def _check_data(self)-> None:
        assert type(self.data_.index) is pd.DatetimeIndex , f"Data Frame index Must be DataTimeIndex Type, but it is {type(self.data_.index).__name__}"

    def _create_dict(self) -> None:
        self.Ratios_ = {k:None for k in self.data_.columns if k != self.IndexFund_}
        self.Csv_ = {k:None for k in self.data_.columns if k != self.IndexFund_}
        self.Dispersion_ = {k:None for k in self.data_.columns if k != self.IndexFund_}

    def _IndexFundRollUp(self, Windows:int) -> None:
        tmpDataIndex = self.data_.copy()
        self.IndexDispersion = tmpDataIndex.loc[:,self.IndexFund_].values / tmpDataIndex.loc[:, self.IndexFund_].shift(-Windows).values - 1
        return self.IndexDispersion

    def RelativeRatio (self, return_std:bool=True)-> pd.DataFrame:
        """ Calculate Relative Rate
        Parameters:
        ---------------------------------------------
        Args:
            return_std: if True 1 Standar Deviation is calculated are Upper and Lower Limit
            
        Return:
            Ratio: Pandas DataFrame with Ratio
        """

        tmpData = self.data_.copy()
        tickers_ = list(self.Ratios_.keys())
        
        for ticker_ in tickers_:
            self.Ratios_[ticker_] = tmpData.loc[:,ticker_].values / tmpData.loc[:, self.IndexFund_].values


            tmp_ticker_ = self.Ratios_[ticker_]

            
            # Calculating average, SD, zscore
            tmp_std, tmp_average = np.std(tmp_ticker_), np.average(tmp_ticker_)

            # Calculating zscore
            tmp_zscore = (tmp_ticker_ - tmp_average) / tmp_std

            if return_std:
                self.Ratios_[f"{ticker_}_+SD"] = np.repeat(tmp_average+tmp_std, len(tmp_ticker_))
                self.Ratios_[f"{ticker_}_-SD"] = np.repeat(tmp_average-tmp_std, len(tmp_ticker_))

        return pd.DataFrame(index = tmpData.index, data = self.Ratios_)

    def _calculate_csv (self, Windows:int):
        """ Internal calculating for CSV
        Args:
            Windows: rolloup time frame
        """

        # Index Fund Dispersion
        tmpIndexDispersion = self._IndexFundRollUp(Windows=Windows)
        tmpData = self.data_.copy()
    
        for col in self.Csv_.keys():
            tmpTickerDispersion = (tmpData.loc[:, col] / tmpData.loc[:, col].shift(-Windows) - 1)-tmpIndexDispersion
            tmpTickerDispersion.dropna(inplace=True)
            
            self.Csv_[col] = tmpTickerDispersion.values
            self.rollupindex = tmpTickerDispersion.index

        return self.Csv_

    def CalculateCSV(self, Windows_:int ) -> pd.DataFrame:
        """ Calculate Dispersion Rate
        Parameters:
        ---------------------------------------------
        Args:
            Windows_:Roll up timeframe 
            
        Return:
            tempCSV: Pandas DataFrame with Ratio and CSV as last Columns
        """

        self.WindowsRollUp = Windows_
        tmpCSV = self._calculate_csv(Windows=Windows_)
        tmpCSV = pd.DataFrame(index = self.rollupindex, data = tmpCSV)
        tmpCSV['CSV'] = tmpCSV.std(axis=1)
            
        return tmpCSV
   
    def Zscore(self,Windows_:int=None):
        """ Calculate ZScore
        Parameters:
        ---------------------------------------------
        Args:
            Windows_: Roll up timeframe
            
        Return:
            Dispersion: Pandas DataFrame with Dispersion
        """


        if self.WindowsRollUp is None:
            tmpCSV_ = self.CalculateCSV(Windows_ = Windows_).copy()

        elif (self.WindowsRollUp is not None) and (Windows_ is not None):
            print(f"Warning : a Dispersion Rates Windows Rollup  was given previously: {self.WindowsRollUp} days ")
            print(f"The program is Re-Calculating CSV Dispersion with the new RollUp windows : {Windows_} days ")
            tmpCSV_ = self.CalculateCSV(Windows_ = Windows_).copy()
        
        else:
            print(f"Warning : Dispersion Rates is calculated based on the Windows Rollup given at CaculateCsv: {self.WindowsRollUp} days ")
            tmpCSV_ = pd.DataFrame(index=self.rollupindex, data = self.Csv_).copy()


        for ticker_ in tmpCSV_.columns:
            if ticker_ != 'CSV':
                # Tmp dispersion
                tmp_dispersion = tmpCSV_[ticker_].values

                # Calculated Average and Std
                tmp_ave, tmp_std = np.average(tmp_dispersion), np.std(tmp_dispersion)
                self.Dispersion_[ticker_] = (tmp_dispersion - tmp_ave) / tmp_std

        self.df_dispersion =  pd.DataFrame(index = tmpCSV_.index, data = self.Dispersion_)
        return pd.DataFrame(index = tmpCSV_.index, data = self.Dispersion_)

    def Plot_Zscore(self, Windows_:int=None, plot_tickers_: List ="all", width:int = 1700, height:int = 900) -> go.Figure:
        """ 
        Plot Zscore 
        Parameters:
        ---------------------------------------------
        Args:
            Windows_: Roll up timeframe
            plot_tickers_: default "all" plot all the ticker other wise pass a list ['ticker_1',..,'ticker_n']
            width : int, width dimmension
            height : int, width dimmension
            
        Return:
            Go.figure
        """
        assert (type(plot_tickers_) == list or plot_tickers_ == "all"), "pass a list with ticker symbols ['ticker1',..,'ticker2'] or use 'all' to plot all of tickers"

         # Create Fig to
        fig = go.Figure()

        # Settings 
        index_ , text =   self.df_dispersion.index, [str(point)[:10] for point in  self.df_dispersion.index]

        # Settings 
        index_ , text =  self.df_dispersion.index, [str(point)[:10] for point in  self.df_dispersion.index]


        if plot_tickers_ == "all":
            # Adding plots
            for ticker_ in self.df_dispersion.columns:
                fig.add_scatter(x = index_, y = self.df_dispersion[ticker_], name=ticker_, text = text, hovertemplate='<b>Zcore: %{y} on %{text}</b>')

        elif type(plot_tickers_) == list:
            for ticker_ in plot_tickers_:
                fig.add_scatter(x = index_, y = self.df_dispersion[ticker_], name=ticker_, text = text, hovertemplate='<b>Zcore: %{y} on %{text}</b>')

        fig.update_layout(autosize=False,width=width,height=height)
        return fig.show()


def Graph3D (data:pd.DataFrame, title:str , variation_score:float = 0.02 ,  edge_color:str = 'GnBu') -> go.Figure :
    """ Return a 3D Graph """
    #https://matplotlib.org/stable/tutorials/colors/colormaps.html

    DATA = data.copy().T        # Transporting
    symbols = np.array(DATA.index)
    total_symbols = len(symbols)
    Dim_ = 3 

    # Calculating Edges
    alphas = np.logspace(-1.5, 1, num=10)
    edge_model = covariance.GraphicalLassoCV(alphas=alphas)

    # standardize the time series: using correlations rather than covariance
    # former is more efficient for structure recovery
    X = data.copy().copy()
    X /= X.std(axis=0)
    edge_model.fit(X)


    _clusters = {}
    _, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=0)
    n_labels = labels.max()

    for i in range(n_labels + 1):
        _clusters[f"Cluster_{i + 1}"] = [symbols[labels == i]]
        print(f"Cluster {i + 1}: {', '.join(symbols[labels == i])}")

    # Creating Lines 
    node_position_model = manifold.LocallyLinearEmbedding(n_components=Dim_, eigen_solver="dense", n_neighbors=n_labels+1)
    embedding = node_position_model.fit_transform(X.T).T

    # Plot the graph of partial correlations
    partial_correlations = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = np.abs(np.triu(partial_correlations, k=1)) > variation_score

    # Colleccting Lines
    start_idx, end_idx = np.where(non_zero) # a sequence of (*line0*, *line1*, *line2*), where:: linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[embedding[:, start], embedding[:, stop]] for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero])
    _color  = cmr.take_cmap_colors(edge_color, len(values), return_fmt='hex',cmap_range =(variation_score,1))

    
    fig = go.Figure()

    # Plotting Lines
    for i in range(len(segments)):
        fig.add_scatter3d(
            x = np.array(segments[i])[:,0], y = np.array(segments[i])[:,1] , z  = np.array(segments[i])[:,2],
            mode = 'lines', line = dict (width = values[i] * 15, color = _color[i])
            )

    # Plotting Points
    for clu, sym in _clusters.items():
        _group = [list(symbols).index(sym) for sym in sym[0]] # This extract the col Positoin at the index that mach with the group plot
        fig.add_scatter3d(
            x = embedding[0, _group], y=embedding[1, _group] , z =embedding[2, _group], text = sym[0], mode = 'markers+text'
        )

    fig.update_layout(autosize=False,width=1000,height=900, scene=dict(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        zaxis=dict(showticklabels=False)),
        margin=dict(l=0, r=0, b=0, t=0),
        meta = {'title':f"3D Cluster by {title}"})
    fig.update_layout(showlegend=False)

    return fig