import pandas as pd
import numpy as np
import nltk
import PyPDF2
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import pymssql
from joblib import Parallel, delayed
import urllib
import requests
from io import BytesIO
from flask import Flask, request, jsonify
from flask import Flask, request, jsonify

# Ensure NLTK dependencies are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class JobRecommender:
    def __init__(self):
        self.f2 = open('stopwords.txt', 'r', errors='ignore')
        self.text2 = self.f2.read()
        self.stopwords_additional = word_tokenize(self.text2.replace("\n", " "))
        self.df2 = pd.DataFrame()
        self.df2['JobID'] = ["I"]
        self.df2['JobName'] = ["I"]
        self.df2['YearsOfExperience'] = ["I"]
        self.df2['Description'] = ["I"]
        self.df2['Country'] = ["I"]
        self.df2['Location'] = ["I"]
        self.df2['PublishDate'] = ["I"]
        self.df2['Salary'] = ["I"]
        self.df2['IsView'] = ["I"]

        # Establish a connection to the SQL Server database
        server = 'db5399.public.databaseasp.net'
        user = 'db5399'
        password = '5e@ZQ?2p8Wx!'
        database = 'db5399'
        self.conn = pymssql.connect(server, user, password, database)

        # Load data from the database
        query = "SELECT * FROM Jobs"
        self.df = pd.read_sql(query, self.conn)
        self.df = self.df[self.df['IsView'] == True]
        self.df['All'] = self.df.loc[:, self.df.columns != 'JobID'].apply(lambda x: ' '.join(x.astype(str)), axis=1)

    def nlp(self, text):
        text = text.lower().replace("\n", "")
        word_sent = word_tokenize(text)
        _stopwords = set(stopwords.words('english') + list(punctuation) + list("●") + list('–') + list('’') + self.stopwords_additional)
        word_sent = [word for word in word_sent if word not in _stopwords]
        lemmatizer = WordNetLemmatizer()
        NLP_Processed_CV = [lemmatizer.lemmatize(word) for word in word_sent]
        return NLP_Processed_CV

    def get_recommendation(self, top, df, scores):
        recommendation = pd.DataFrame(columns=['JobID', 'JobName', 'Description', 'Salary', 'Country', 'Location',
                                                'PublishDate', 'YearsOfExperience', 'score'])
        count = 0
        for i in top:
            recommendation.at[count, 'JobID'] = df['JobID'][i]  # Get JobID from the database

            recommendation.at[count, 'JobName'] = df['JobName'][i]
            recommendation.at[count, 'IsView'] = df['IsView'][i]
            recommendation.at[count, 'YearsOfExperience'] = df['YearsOfExperience'][i]
            recommendation.at[count, 'Description'] = df['Description'][i]
            recommendation.at[count, 'Country'] = df['Country'][i]
            recommendation.at[count, 'Location'] = df['Location'][i]
            recommendation.at[count, 'Salary'] = df['Salary'][i]
            recommendation.at[count, 'PublishDate'] = df['PublishDate'][i]
            recommendation.at[count, 'score'] = scores[count]
            count += 1
        return recommendation

    def TFIDF(self, scraped_data, cv):
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_jobid = tfidf_vectorizer.fit_transform(scraped_data)
        user_tfidf = tfidf_vectorizer.transform(cv)
        cos_similarity_tfidf = Parallel(n_jobs=-1)(delayed(cosine_similarity)(user_tfidf, x) for x in tfidf_jobid)
        return list(cos_similarity_tfidf)

    def count_vectorize(self, scraped_data, cv):
        count_vectorizer = CountVectorizer(max_features=5000)
        count_jobid = count_vectorizer.fit_transform(scraped_data)
        user_count = count_vectorizer.transform(cv)
        cos_similarity_countv = Parallel(n_jobs=-1)(delayed(cosine_similarity)(user_count, x) for x in count_jobid)
        return list(cos_similarity_countv)

    def KNN(self, cv):
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        n_neighbors = 2
        KNN = NearestNeighbors(n_neighbors=n_neighbors, p=2, n_jobs=-1)
        KNN.fit(tfidf_vectorizer.fit_transform(self.df['All']))
        NNs = KNN.kneighbors(tfidf_vectorizer.transform(cv), return_distance=True)
        top = NNs[1][0][1:]
        index_score = NNs[0][0][1:]
        knn = self.get_recommendation(top, self.df, index_score)
        return knn

    def process_cv(self, cv_url):
        response = requests.get(cv_url)
        response.raise_for_status()

        file = BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(file)
        new_cv_text = ''
        for page_num in range(len(pdf_reader.pages)):
            new_cv_text += pdf_reader.pages[page_num].extract_text()
        processed_new_cv = self.nlp(new_cv_text)
        processed_new_cv_str = ' '.join(processed_new_cv)
        NLP_Processed_CV = self.nlp(processed_new_cv_str)

        self.df2['All'] = " ".join(NLP_Processed_CV)
        output2 = self.TFIDF(self.df['All'], self.df2['All'])
        df = self.df.reset_index(drop=True)
        top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:1000]
        list_scores = [output2[i][0][0] for i in top]
        TF = self.get_recommendation(top, df, list_scores)
        output3 = self.count_vectorize(df['All'], self.df2['All'])
        top = sorted(range(len(output3)), key=lambda i: output3[i], reverse=True)[:1000]
        list_scores = [output3[i][0][0] for i in top]
        cv = self.get_recommendation(top, df, list_scores)
        knn = self.KNN(self.df2['All'])
        merge1 = knn[['JobID', 'JobName', 'Description', 'Salary', 'Country', 'Location',
                      'IsView',
                      'PublishDate', 'YearsOfExperience', 'score']].merge(TF[['JobID', 'score']], on="JobID")
        final = merge1.merge(cv[['JobID', 'score']], on="JobID")
        final = final.rename(columns={"score_x": "KNN", "score_y": "TF-IDF", "score": "CV"})
        slr = MinMaxScaler()
        final[["KNN", "TF-IDF", 'CV']] = slr.fit_transform(final[["KNN", "TF-IDF", 'CV']])
        final['KNN'] = (1 - final['KNN']) / 3
        final['TF-IDF'] = final['TF-IDF'] / 3
        final['CV'] = final['CV'] / 3
        final['Final'] = final['KNN'] + final['TF-IDF'] + final['CV']
        final.sort_values(by="Final", ascending=False)
        result_jd = final
        final_jobrecomm = result_jd.head(10)
        final_jobrecomm_dict = final_jobrecomm.to_dict(orient='records')
        return final_jobrecomm_dict
