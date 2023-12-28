import json
import sys
import yfinance as yf
import pandas as pd

tickers = yf.Tickers('msft aapl goog INFY.NS hdfc.ns')
TK=input('ENTER STOCK SYMBOL = ')
up= TK.upper()
msft = tickers.tickers[f"{up}"].info
print(type(msft))
with open('data.json','w') as f:
    js= json.dumps(msft,f)
print(js)
