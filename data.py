
import pandas as pd
import yfinance as yf


def Get_raw_data(ticker, start, end):

    #Importar datos
    aux = yf.download(ticker, start=start, end=end, progress=False)
    prices = aux["Close"].squeeze() #esta funcion transforma la columna de fechas en un indice que puede usar el df de pandas

    #Definir el DataFrame
    df = pd.DataFrame({
        "Close": prices
    })

    return df

def Get_process_data(ticker, start, end):

    #Importar df con datos crudos
    df = Get_raw_data(ticker, start, end)

    #Feature engineering
    df['ret_1'] = df['Close'].pct_change()
    df['ret_5'] = df['Close'].pct_change(5)
    df['ma_5'] = df['Close'].rolling(5).mean()
    df['vol_5'] = df['ret_1'].rolling(5).std()

    df = df.dropna()

    X = df[['ret_1', 'ret_5', 'ma_5', 'vol_5']]

    return X

def Get_target(df):
    target = (df['ret_1'].shift(-1) > 0).astype(int)
    return target



#version para semanas
#    aux = yf.download(ticker, start=start, end=end, interval="1wk", progress=False)