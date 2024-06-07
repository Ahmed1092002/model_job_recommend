# from sqlalchemy import create_engine
# import pandas as pd
# import numpy as np
#
# # # Create a connection to your database
# # engine = create_engine('mysql://root:eGGEEcsiwuFlGntwhDSNEFlGILkCFIxY@monorail.proxy.rlwy.net:10257/railway')
# #
# # # Load your data into a pandas DataFrame
# df = pd.read_csv('jobs_dataset.csv')
# # List of columns to drop
# df.info()
# # List of columns to drop
# columns_to_drop = ['location', 'Country','longitude', 'latitude','Company Size','Contact Person','Contact','Job Portal','Sector','City','State','Zip','Website','Ticker','CEO']
#
# # Check if the columns exist in the DataFrame
# columns_to_drop = [col for col in columns_to_drop if col in df.columns]
#
# # Drop the columns
# df = df.drop(columns=columns_to_drop)
#
# df.info()
# df = df.dropna()
#
# # If your data uses a different indicator for missing values, such as 'N?A', you should first replace those values with NaNs
# df = df.replace('N?A', np.nan).dropna()# Drop the columns
# columns_to_drop = ['All']
# df = df.drop(columns=columns_to_drop)
#
# df.info()
# df['All'] = df.loc[:, ~df.columns.isin(['Job Id', 'Job Posting Date','Benefits'])].apply(lambda x: ' '.join(x.astype(str)), axis=1)
# df.info()
# print (df['All'])
#
# df.to_csv('jobs_dataset_new.csv', index=False)
#
# # df = df.drop(columns=columns_to_drop)
#
# # # # Write the data from the DataFrame to your SQL database
# # df.to_sql('jobs_dataset', con=engine, if_exists='replace', index=False)
# # print('Data loaded to SQL')

import pandas as pd
import numpy as np
import nltk
import PyPDF2
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from heapq import nlargest
from collections import defaultdict
import json
from nltk.collocations import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import urllib
from sqlalchemy import create_engine
import pandas as pd

params = urllib.parse.quote_plus(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=db5399.public.databaseasp.net;'
    'DATABASE=db5399;'
    'UID=db5399;'
    'PWD=5e@ZQ?2p8Wx!'
)

connection_string = f"mssql+pyodbc:///?odbc_connect={params}"
engine = create_engine(connection_string)

# SQL query to load data
query = "SELECT * FROM Jobs"

# Use pandas to load data from SQL query
df = pd.read_sql_query(query, engine)
print(df.columns)