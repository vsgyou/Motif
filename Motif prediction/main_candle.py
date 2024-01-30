#%%
import yfinance as yf
import torch
import torch.nn as nn
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
#%%
stock_code = '005930.KS'
start_date = '2020-07-30'
end_date = '2023-07-30'
samsung_data = yf.download(stock_code, start = start_date, end = end_date)
fig = go.Figure(data = [go.Candlestick(x = samsung_data.index,
                                       open = samsung_data['Open'],
                                       high = samsung_data['High'],
                                       low = samsung_data['Low'],
                                       close = samsung_data['Close'])])
fig.show()

samsung_data.index