import time
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
# .yf lib
import yfinance as yf
# yahoo fin lib
# from yahoo_fin.stock_info import *
import datetime as dt,datetime
# import datetime
import qrcode
import queue  # Importing the queue module
import concurrent.futures
from .models import Project

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection, svm
from app.valid_tickers import Valid_Ticker
import threading
from django.http import JsonResponse




def index(request):
    tickers=['AAPL', 'AMZN', 'QCOM', 'META', 'NVDA', 'JPM','TCS.NS','INFY.NS','RELIANCE.NS','BTC-USD','ETH-USD','SOL-USD','XRP-USD','^NSEI']
    data = yf.download(
        tickers,
        group_by='ticker',
        threads=True,
        period='5y', 
        interval='1d'
    )

    data.reset_index(level=0, inplace=True)

    fig_left = go.Figure()
    fig_right = go.Figure()
    fig_crypto = go.Figure()
    fig_index = go.Figure()

    for ticker in tickers:
        if ticker in data.columns.levels[0]:
            adj_close = data[ticker]['Adj Close']
            high = data[ticker]['High']
            low = data[ticker]['Low']
            open = data[ticker]['Open']
            # close = data[ticker]['Adj Close']
            if ticker in ['AAPL', 'AMZN', 'META', 'NVDA']:
                # fig_left.add_trace(go.Candlestick(open=open,
                # high=high,
                # low=low,
                # close=adj_close,))
                fig_left.add_trace(go.Scatter(x=data['Date'], y=adj_close, name=ticker))
            elif ticker in ['TCS.NS', 'INFY.NS', 'RELIANCE.NS']:
                fig_right.add_trace(go.Scatter(x=data['Date'], y=adj_close, name=ticker))
            elif ticker in ['BTC-USD','ETH-USD','SOL-USD','XRP-USD']:
                fig_crypto.add_trace(go.Scatter(x=data['Date'], y=adj_close, name=ticker))
            elif ticker in ['^NSEI']:
                print(ticker)
                fig_index.add_trace(go.Scatter(x=data['Date'], y=adj_close, name=ticker))

    fig_left.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    fig_right.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    fig_crypto.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    fig_index.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")

    plot_div_left = plot(fig_left, auto_open=False, output_type='div')
    plot_div_right = plot(fig_right, auto_open=False, output_type='div')
    plot_div_crypto = plot(fig_crypto, auto_open=False, output_type='div')
    plot_div_index = plot(fig_index, auto_open=False, output_type='div')
    return render(request, 'index.html',{ 'plot_div_left': plot_div_left,
        'plot_div_right': plot_div_right,'plot_div_crypto':plot_div_crypto,'plot_div_index':plot_div_index})
    # ================================================= Left Card Plot =========================================================
    # Here we use yf.download function

def fetch_data(request):    
    start = time.time()
    # tickers=['AAPL', 'AMZN', 'QCOM', 'META', 'NVDA', 'JPM','TCS.NS','INFY.NS','RELIANCE.NS']
    # data = yf.download(
    #     tickers,
    #     group_by='ticker',
    #     threads=True,
    #     period='1y', 
    #     interval='1d'
    # )

    # data.reset_index(level=0, inplace=True)

    # fig_left = go.Figure()
    # fig_right = go.Figure()

    # for ticker in tickers:
    #     if ticker in data.col umns.levels[0]:
    #         adj_close = data[ticker]['Adj Close']
    #         if ticker in ['AAPL', 'AMZN', 'META', 'NVDA']:
    #             fig_left.add_trace(go.Scatter(x=data['Date'], y=adj_close, name=ticker))
    #         elif ticker in ['TCS.NS', 'INFY.NS', 'RELIANCE.NS']:
    #             fig_right.add_trace(go.Scatter(x=data['Date'], y=adj_close, name=ticker))

    # fig_left.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    # fig_right.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")

    # plot_div_left = plot(fig_left, auto_open=False, output_type='div')
    # plot_div_right = plot(fig_right, auto_open=False, output_type='div')


    def download_and_process(tickers):
        dfs = []
        for ticker in tickers:
            df = yf.download(tickers=ticker, period='1d', threads=True, interval='1d')
            df['Change'] = df['Close'] - df['Open']  # Calculate change
            
            df.insert(0, "Ticker", ticker)
            dfs.append(df)
        df_concat = pd.concat(dfs, axis=0)
        df_concat.reset_index(level=0, inplace=True)
        df_concat.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume', 'Change']
        df_concat.drop('Date', axis=1, inplace=True)
        json_records = df_concat.reset_index().to_json(orient='records')
        # print(json_records)
        return json.loads(json_records)

# US Stocks
    us_stocks = ['AAPL', 'AMZN', 'GOOGL', 'UBER', 'TSLA', 'NVDA']
    recent_stocks = download_and_process(us_stocks)

    # Indian Stocks
    indian_stocks = ['INFY.NS', 'TCS.NS', 'RELIANCE.NS', 'MRF.NS']
    recent_indian_stocks = download_and_process(indian_stocks)

    # Crypto
    crypto_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD']
    crypto = download_and_process(crypto_symbols)

    

   
    print(time.time() - start)
    data= {
        # 'plot_div_left': plot_div_left,
        # 'plot_div_right': plot_div_right,
        'recent_stocks': recent_stocks,
        'recent_indian_stocks': recent_indian_stocks,
        'crypto': crypto
    }
    # ========================================== Page Render section =====================================================
    return JsonResponse(data)
    # return render(request, 'index.html', {
    #     'plot_div_left': plot_div_left,
    #     'plot_div_right': plot_div_right,
    #     'recent_stocks': recent_stocks,
    #     'recent_indian_stocks': recent_indian_stocks,
    #     'crypto': crypto
    # })



    # ================================================ To show recent stocks ==============================================
    
#     df1 = yf.download(tickers = 'AAPL', threads=True, period='1d', interval='1d')
#     df2 = yf.download(tickers = 'AMZN', threads=True, period='1d', interval='1d')
#     df3 = yf.download(tickers = 'GOOGL', period='1d',threads=True, interval='1d')
#     df4 = yf.download(tickers = 'UBER', period='1d',threads=True, interval='1d')
#     df5 = yf.download(tickers = 'TSLA', period='1d',threads=True, interval='1d')
#     df6 = yf.download(tickers = 'NVDA', period='1d',threads=True, interval='1d')

#     # indian stocks
#     df7 = yf.download(tickers = 'INFY.NS', period='1d',threads=True, interval='1d')
#     df8 = yf.download(tickers = 'TCS.NS', period='1d', threads=True,interval='1d')
#     df9 = yf.download(tickers = 'RELIANCE.NS',threads=True, period='1d', interval='1d')
#     df10 = yf.download(tickers = 'MRF.NS',threads=True, period='1d', interval='1d')


#     # crypto
#     df11 = yf.download(tickers = 'BTC-USD',threads=True, period='1d', interval='1d')
#     df12 = yf.download(tickers = 'ETH-USD',threads=True, period='1d', interval='1d')
#     df13 = yf.download(tickers = 'SOL-USD',threads=True, period='1d', interval='1d')
#     df14 = yf.download(tickers = 'XRP-USD',threads=True, period='1d', interval='1d')
    
#     # US STONKS
#     df1.insert(0, "Ticker", "APPLE.INC")
#     df2.insert(0, "Ticker", "AMAZON")
#     df3.insert(0, "Ticker", "GOOGLE")
#     df4.insert(0, "Ticker", "UBER")
#     df5.insert(0, "Ticker", "TESLA")
#     df6.insert(0, "Ticker", "NVEDIA")

# # indian stonks
#     df7.insert(0, "Ticker", "INFOSYS INDIA")
#     df8.insert(0, "Ticker", "TATA CONSULTANCY SERVICES INDIA")
#     df9.insert(0, "Ticker", "RELIANCE INDIA")
#     df10.insert(0, "Ticker", "MRF INDIA")
# #    CRYPTO
#     df11.insert(0, "Ticker", "BTC-USD")
#     df12.insert(0, "Ticker", "ETH-USD")
#     df13.insert(0, "Ticker", "SOL-USD")
#     df14.insert(0, "Ticker", "XRP-USD")

#     df = pd.concat([df1, df2, df3, df4, df5, df6], axis=0)
#     df.reset_index(level=0, inplace=True)
#     df.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
#     convert_dict = {'Date': object}
#     df = df.astype(convert_dict)
#     df.drop('Date', axis=1, inplace=True)
#     json_records = df.reset_index().to_json(orient='records')
#     recent_stocks = []
#     recent_stocks = json.loads(json_records)


#     ind = pd.concat([df7, df8, df9, df10], axis=0)
#     ind.reset_index(level=0, inplace=True)
#     ind.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
#     convert_dict = {'Date': object}
#     ind = ind.astype(convert_dict)
#     ind.drop('Date', axis=1, inplace=True)

#     json_records = ind.reset_index().to_json(orient='records')
#     recent_indian_stocks = []
#     recent_indian_stocks = json.loads(json_records)

#     # crypto
#     crypto = pd.concat([df11,df12,df13,df14], axis=0)
#     crypto.reset_index(level=0, inplace=True)
#     crypto.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
#     convert_dict = {'Date': object}
#     crypto = crypto.astype(convert_dict)
#     crypto.drop('Date', axis=1, inplace=True)

#     json_records = crypto.reset_index().to_json(orient='records')
#     crypto_symbols = []
#     crypto_symbols = json.loads(json_records)
#     print(time.time()-start)
    

# search page
def search(request):
    return render(request, 'search.html', {})






# The Predict Function to implement Machine Learning as well as Plotting
def predict(request, ticker_value, number_of_days):
    try:
        
        # ticker_value = request.POST.get('ticker')
        ticker_value = ticker_value.upper()
        df = yf.download(tickers = ticker_value, period='1w', interval='1m')
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
    ticker.columns = ['Symbol', 'Name', 'Last_Sale', 'Net_Change', 'Percent_Change', 'Market_Cap','Country', 'IPO_Year', 'Volume', 'Sector', 'Industry']
    for i in range(0,ticker.shape[0]):
        if ticker.Symbol[i] == to_search:
            Symbol = ticker.Symbol[i]
            Name = ticker.Name[i]
            data = yf.Ticker(Symbol).info
            # print(data)
            # market info
            currency = data.get('financialCurrency')
            dividends = data.get('lastDividendValue')
            yearly_change = "₹ {:,.2f} /-".format(data.get('52WeekChange'))
            enterpriseValue = "₹ {:,.2f} /-".format(data.get('enterpriseValue'))
            totalRevenue = "₹ {:,.2f} /-".format(data.get('totalRevenue'))
            Market_Cap = "₹ {:,.2f} /-".format(data.get('marketCap'))
            Country = data.get('country')
            debt ="₹ {:,.2f} /-".format(data.get('totalDebt'))
            IPO_Year = dt.datetime.utcfromtimestamp(data.get('firstTradeDateEpochUtc'))
            Volume = " {:,.2f} ".format(data.get('volume'))
            Sector = data.get('sector')
            Industry = data.get('industry')

            # market financials.

            currentPrice = data.get('currentPrice')            
            debtToEquity = data.get('debtToEquity')
            revenuePerShare = data.get('revenuePerShare')
            revenueGrowth = data.get('revenueGrowth')
            floatShares = data.get('floatShares')
            profitMargins = data.get('profitMargins')
            averageVolume = data.get('averageVolume')
            fiftyTwoWeekHigh = data.get('fiftyTwoWeekHigh')
            fiftyTwoWeekLow = data.get('fiftyTwoWeekLow')
            fiftyDayAverage = data.get('fiftyDayAverage')
            twoHundredDayAverage = data.get('twoHundredDayAverage')
            targetHighPrice = data.get('targetHighPrice')
            targetLowPrice = data.get('targetLowPrice')
            targetMeanPrice = data.get('targetMeanPrice')
            targetMedianPrice = data.get('targetMedianPrice')
            recommendationMean = data.get('recommendationMean')
            recommendationKey = data.get('recommendationKey')
            break
 
    tk = yf.Ticker(ticker_value).income_stmt
    tk2 = yf.Ticker(ticker_value).balance_sheet
    rev = tk.loc['Total Revenue']
    deb = tk2.loc['Total Debt']

    # Prepare data for rendering
    data = []
    for year in rev.index:
        year_data = {'Year': year.year, 'Total Revenue': rev[year], 'Total Debt': deb[year]}
        data.append(year_data)

    # Create Plotly figure
    fig = go.Figure()

    # Add traces for each year
    for year_data in data:
        fig.add_trace(go.Bar(
            x=['Total Revenue', 'Total Debt'],
            y=[year_data['Total Revenue'], year_data['Total Debt']],
            name=str(year_data['Year']),
        ))

    # Update layout
    fig.update_layout(
        barmode='group',    
        title='Revenue vs. Debt',
        xaxis=dict(title='year'),
        yaxis=dict(title='Amount'),
        autosize=True,
        paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white"
        # :True,  
        # margin=dict(l=50, r=50, t=50, b=50)
    )
    # Convert Plotly figure to HTML
    financials_div = fig.to_html(full_html=True,)
    


    return render(request, "result.html", context={ 'plot_div': plot_div, 
                                                    'confidence' : confidence,
                                                    'forecast': forecast,
                                                    'ticker_value':ticker_value,
                                                    'number_of_days':number_of_days,
                                                    'plot_div_pred':plot_div_pred,
                                                    'financials_div':financials_div,
                                                    'Symbol':Symbol,
                                                    'Name':Name,
                                                    'currency':currency,
                                                    'dividends':dividends,
                                                    'yearly_change':yearly_change,
                                                    'totalRevenue':totalRevenue,
                                                    'Market_Cap':Market_Cap,
                                                    'Country':Country,
                                                    'IPO_Year':IPO_Year,
                                                    'Volume':Volume,
                                                    'Sector':Sector,
                                                    'Industry':Industry,
                                                    'debt':debt,
                                                    'enterpriseValue':enterpriseValue,
                                                    # financials
                                                    'currentPrice' :currentPrice,          
                                                    'debtToEquity':debtToEquity,
                                                    'revenuePerShare':revenuePerShare,
                                                    'revenueGrowth':revenueGrowth ,
                                                    'floatShares':floatShares,
                                                    'profitMargins':profitMargins,
                                                    'averageVolume':averageVolume,
                                                    'fiftyTwoWeekHigh':fiftyTwoWeekHigh,
                                                    'fiftyTwoWeekLow':fiftyTwoWeekLow,
                                                    'fiftyDayAverage':fiftyDayAverage,
                                                    'twoHundredDayAverage':twoHundredDayAverage,
                                                    'targetHighPrice':targetHighPrice,
                                                    'targetLowPrice':targetLowPrice,
                                                    'targetMeanPrice':targetMeanPrice,
                                                    'targetMedianPrice':targetMedianPrice,
                                                    'recommendationMean':recommendationMean,
                                                    'recommendationKey':recommendationKey,



                                                    })
def get_last_n_years_data(ticker, n):
    try:
        # Fetch the historical market data
        stock = yf.Ticker(ticker)
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=n*365)
        data = stock.history(start=start_date, end=end_date)

        # Filter relevant columns (adjust as per your data availability)
        financials = data[['Total Revenue', 'Long Term Debt']]

        return financials
    except Exception as e:
        # Handle any errors gracefully
        print(f"Error fetching data for {ticker}: {e}")
        return None



# need to implement later.....
# search feature activation is needed


# def   search_ticker(request,symbol):
#     with open('app/Data/Tickers.csv ', newline='') as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             a =row['Symbol'].find(symbol)
#             if a == 0:
#                 break
#         else:
#             return HttpResponse('not avl')
#         # print(row)
#         return HttpResponse('found')
    # stock_search_app/views.py

# search result page
    # ================================================= Load Ticker Table ================================================
def ticker(request):
    # ticker_df = pd.read_csv('app/Data/new_tickers.csv') 
    # json_ticker = ticker_df.reset_index().to_json(orient ='records')
    # ticker_list = []
    # ticker_list = json.loads(json_ticker)
    # return render(request, 'ticker.html', {
    #     'ticker_list': ticker_list
    # })
# def search_tickers(request):
    if request.is_ajax():
        query = request.GET.get('query')
        if query:
            results = filter_stocks(query)   
            return JsonResponse({'results': results})
    else:
        # Load all tickers from the file
        ticker_df = pd.read_csv('app/Data/Tickers.csv') 
        json_ticker = ticker_df.reset_index().to_json(orient='records')
        results = json.loads(json_ticker)
        return render(request, 'ticker.html', {'ticker_list': results})

    return render(request, 'ticker.html')


def crypto_ticker(request):

    if request.is_ajax():
        query = request.GET.get('query')
        if query:
            results = filter_stocks(query)   
            return JsonResponse({'results': results})
    else:
        # Load all tickers from the file
        ticker_df = pd.read_csv('app/Data/crypto_list.csv') 
        json_ticker = ticker_df.reset_index().to_json(orient='records')
        results = json.loads(json_ticker)
        return render(request, 'crypto_tik.html', {'ticker_list': results})

    return render(request, 'crypto_tik.html')

def filter_stocks(query):
    # Load CSV file and filter stocks based on query
    results = []
    with open('app/Data/raw.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if query.lower() in row['Symbol'].lower() or query.lower() in row['Name'].lower():
                results.append({'number': row['number'], 'symbol': row['Symbol'], 'name': row['Name']})
    return results
