from urllib import request
from django.shortcuts import render
from django.http import HttpResponse
from django.template import RequestContext
import csv
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import Scatter

import pandas as pd
import numpy as np
import json

import yfinance as yf
import datetime as dt
import qrcode

from .models import Project

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection, svm
from app.valid_tickers import Valid_Ticker



# The Home page when Server loads up
def index(request):
    # ================================================= Left Card Plot =========================================================
    # Here we use yf.download function
    data = yf.download(
        
        # passes the ticker
        tickers=['AAPL', 'AMZN', 'QCOM', 'META', 'NVDA', 'JPM','TCS.NS','INFY.NS','RELIANCE.NS'],
        # tickers= Valid_Ticker
        group_by = 'ticker',
        
        threads=True, # Set thread value to true
        # used for access data[ticker]
        period='1y', 
        interval='1d'
    
    )

    data.reset_index(level=0, inplace=True)
    # data.reset_index(level=0, inplace=True)
    


    fig_left = go.Figure()
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['AAPL']['Adj Close'], name="AAPL")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['AMZN']['Adj Close'], name="AMZN")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['QCOM']['Adj Close'], name="QCOM")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['META']['Adj Close'], name="META")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['NVDA']['Adj Close'], name="NVDA")
            )
    fig_left.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    
    fig_right = go.Figure()
    fig_right.add_trace(
                go.Scatter(x=data['Date'], y=data['TCS.NS']['Adj Close'], name="TCS")
            )
    fig_right.add_trace(
                go.Scatter(x=data['Date'], y=data['INFY.NS']['Adj Close'], name="INFY")
            )
    fig_right.add_trace(
                go.Scatter(x=data['Date'], y=data['RELIANCE.NS']['Adj Close'], name="RELIANCE")
            )
    
    fig_right.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")

    plot_div_left = plot(fig_left, auto_open=False, output_type='div')
    plot_div_right = plot(fig_right, auto_open=False, output_type='div')

    # ================================================ To show recent stocks ==============================================
    
    df1 = yf.download(tickers = 'AAPL', period='1d', interval='1d')
    df2 = yf.download(tickers = 'AMZN', period='1d', interval='1d')
    df3 = yf.download(tickers = 'GOOGL', period='1d', interval='1d')
    df4 = yf.download(tickers = 'UBER', period='1d', interval='1d')
    df5 = yf.download(tickers = 'TSLA', period='1d', interval='1d')
    df6 = yf.download(tickers = 'NVDA', period='1d', interval='1d')

    # indian stocks
    df7 = yf.download(tickers = 'INFY.NS', period='1d', interval='1d')
    df8 = yf.download(tickers = 'TCS.NS', period='1d', interval='1d')
    df9 = yf.download(tickers = 'RELIANCE.NS', period='1d', interval='1d')
    df10 = yf.download(tickers = 'MRF.NS', period='1d', interval='1d')


    # crypto
    df11 = yf.download(tickers = 'BTC-USD', period='1d', interval='1d')
    df12 = yf.download(tickers = 'ETH-USD', period='1d', interval='1d')
    df13 = yf.download(tickers = 'SOL-USD', period='1d', interval='1d')
    df14 = yf.download(tickers = 'XRP-USD', period='1d', interval='1d')
    
    # US STONKS
    df1.insert(0, "Ticker", "APPLE.INC")
    df2.insert(0, "Ticker", "AMAZON")
    df3.insert(0, "Ticker", "GOOGLE")
    df4.insert(0, "Ticker", "UBER")
    df5.insert(0, "Ticker", "TESLA")
    df6.insert(0, "Ticker", "NVEDIA")

# indian stonks
    df7.insert(0, "Ticker", "INFOSYS INDIA")
    df8.insert(0, "Ticker", "TATA CONSULTANCY SERVICES INDIA")
    df9.insert(0, "Ticker", "RELIANCE INDIA")
    df10.insert(0, "Ticker", "MRF INDIA")
#    CRYPTO
    df11.insert(0, "Ticker", "BTC-USD")
    df12.insert(0, "Ticker", "ETH-USD")
    df13.insert(0, "Ticker", "SOL-USD")
    df14.insert(0, "Ticker", "XRP-USD")

    df = pd.concat([df1, df2, df3, df4, df5, df6], axis=0)
    df.reset_index(level=0, inplace=True)
    df.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    convert_dict = {'Date': object}
    df = df.astype(convert_dict)
    df.drop('Date', axis=1, inplace=True)
    json_records = df.reset_index().to_json(orient='records')
    recent_stocks = []
    recent_stocks = json.loads(json_records)


    ind = pd.concat([df7, df8, df9, df10], axis=0)
    ind.reset_index(level=0, inplace=True)
    ind.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    convert_dict = {'Date': object}
    ind = ind.astype(convert_dict)
    ind.drop('Date', axis=1, inplace=True)

    json_records = ind.reset_index().to_json(orient='records')
    recent_indian_stocks = []
    recent_indian_stocks = json.loads(json_records)


    # crypto
    crypto = pd.concat([df11,df12,df13,df14], axis=0)
    crypto.reset_index(level=0, inplace=True)
    crypto.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    convert_dict = {'Date': object}
    crypto = crypto.astype(convert_dict)
    crypto.drop('Date', axis=1, inplace=True)

    json_records = crypto.reset_index().to_json(orient='records')
    crypto_symbols = []
    crypto_symbols = json.loads(json_records)

    # ========================================== Page Render section =====================================================

    return render(request, 'index.html', {
        'plot_div_left': plot_div_left,
        'plot_div_right': plot_div_right,
        'recent_stocks': recent_stocks,
        'recent_indian_stocks': recent_indian_stocks,
        'crypto_symbols': crypto_symbols
    })

# search page
def search(request):
    return render(request, 'search.html', {})


# search result page
def ticker(request):
    # ================================================= Load Ticker Table ================================================
    ticker_df = pd.read_csv('app/Data/new_tickers.csv') 
    json_ticker = ticker_df.reset_index().to_json(orient ='records')
    ticker_list = []
    ticker_list = json.loads(json_ticker)


    return render(request, 'ticker.html', {
        'ticker_list': ticker_list
    })





# The Predict Function to implement Machine Learning as well as Plotting
def predict(request, ticker_value, number_of_days):
    try:
        
        # ticker_value = request.POST.get('ticker')
        ticker_value = ticker_value.upper()
        df = yf.download(tickers = ticker_value, period='1d', interval='1m')
        print("Downloaded ticker = {} successfully".format(ticker_value))
    except:
        return render(request, 'API_Down.html', {})

    try:
        # number_of_days = request.POST.get('days')
        number_of_days = int(number_of_days)
    except:
        return render(request, 'Invalid_Days_Format.html', {})
    # from valid_tickers import stock_list as Valid_Ticker

  
    
    if ticker_value not in Valid_Ticker:
        return render(request, 'Invalid_Ticker.html', {})
    
    if number_of_days < 0:
        return render(request, 'Negative_Days.html', {})
    
    if number_of_days > 365:
        return render(request, 'Overflow_days.html', {})
    

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'], name = 'market data'))
    fig.update_layout(
                        title='{} live share price evolution'.format(ticker_value),
                        yaxis_title='Stock Price (USD per Shares)')
    fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
        )
    )
    fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="blue")
    plot_div = plot(fig, auto_open=False, output_type='div')



    # ========== Machine Learning ===============


    try:
        df_ml = yf.download(tickers = ticker_value, period='2y', interval='1h')
    except:
        ticker_value = 'AAPL'
        df_ml = yf.download(tickers = ticker_value, period='2y', interval='1m')

    # Fetching ticker values from Yahoo Finance API 
    df_ml = df_ml[['Adj Close']]
    forecast_out = int(number_of_days)
    df_ml['Prediction'] = df_ml[['Adj Close']].shift(-forecast_out)
    # Splitting data for Test and Train
    X = np.array(df_ml.drop(['Prediction'], axis=1))

    # X = np.array(df_ml.drop(['Prediction'],1))
    X = preprocessing.scale(X)
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]
    y = np.array(df_ml['Prediction'])
    y = y[:-forecast_out]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
    # Applying Linear Regression
    clf = LinearRegression()
    clf.fit(X_train,y_train)
    # Prediction Score
    confidence = clf.score(X_test, y_test)
    # Predicting for 'n' days stock data
    forecast_prediction = clf.predict(X_forecast)
    forecast = forecast_prediction.tolist()


    # ========================================== Plotting predicted data ======================================


    pred_dict = {"Date": [], "Prediction": []}
    for i in range(0, len(forecast)):
        pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        pred_dict["Prediction"].append(forecast[i])
    
    pred_df = pd.DataFrame(pred_dict)
    pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'])])
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')

    # ========================================== Display Ticker Info ==========================================
    # Symbol = 'INFY.NS'  # or any appropriate default value

    ticker = pd.read_csv('app/Data/Tickers.csv')
    to_search = ticker_value
    ticker.columns = ['Symbol', 'Name', 'Last_Sale', 'Net_Change', 'Percent_Change', 'Market_Cap',
                    'Country', 'IPO_Year', 'Volume', 'Sector', 'Industry']
    for i in range(0,ticker.shape[0]):
        if ticker.Symbol[i] == to_search:
            Symbol = ticker.Symbol[i]
            Name = ticker.Name[i]
            Last_Sale = ticker.Last_Sale[i]
            Net_Change = ticker.Net_Change[i]
            Percent_Change = ticker.Percent_Change[i]
            Market_Cap = ticker.Market_Cap[i]
            Country = ticker.Country[i]
            IPO_Year = ticker.IPO_Year[i]
            Volume = ticker.Volume[i]
            Sector = ticker.Sector[i]
            Industry = ticker.Industry[i]
            break

    # ========================================== Page Render section =========================================
    

    return render(request, "result.html", context={ 'plot_div': plot_div, 
                                                    'confidence' : confidence,
                                                    'forecast': forecast,
                                                    'ticker_value':ticker_value,
                                                    'number_of_days':number_of_days,
                                                    'plot_div_pred':plot_div_pred,
                                                    'Symbol':Symbol,
                                                    'Name':Name,
                                                    'Last_Sale':Last_Sale,
                                                    'Net_Change':Net_Change,
                                                    'Percent_Change':Percent_Change,
                                                    'Market_Cap':Market_Cap,
                                                    'Country':Country,
                                                    'IPO_Year':IPO_Year,
                                                    'Volume':Volume,
                                                    'Sector':Sector,
                                                    'Industry':Industry,
                                                    })
# need to implement later.....
# search feature activation is needed

def search_ticker(request,symbol):
    with open('app/Data/Tickers.csv ', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            a =row['Symbol'].find(symbol)
            if a == 0:
                break
        else:
            return HttpResponse('not avl')
        # print(row)
        return HttpResponse('found')

