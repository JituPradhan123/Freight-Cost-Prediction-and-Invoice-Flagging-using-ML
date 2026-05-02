import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from sklearn.model_selection import train_test_split

def load_invoice_data(db_path: str):
    con = sqlite3.connect(db_path)
    query = "select * from vendor_invoice"
    df = pd.read_sql_query(query,con)
    con.close()
    return df

def prepare_features(df:pd.DataFrame):
    x=df[["Dollars","Quantity"]]
    y=df[['Freight']]
    return x , y

def split_data(x,y):
    return train_test_split(x,y,test_size=0.2,random_state=42)
