# import libraties
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import *
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import nltk


class ProductRecommendation:
    
    def __init__(self):
        nltk.data.path.append('./nltk_data/')
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        self.data = pickle.load(open('data.pkl','rb'))
        self.user_final_rating = pickle.load(open('user_final_rating.pkl','rb'))
        self.model = pickle.load(open('logistic_regression.pkl','rb'))   ### logistic regression was the best performed hence using it
        self.raw_data = pd.read_csv("sample30.csv",usecols=['id','name','brand','categories','manufacturer'])  ## only reading  what is required
        self.data = pd.concat([self.raw_data[['id','name','brand','categories','manufacturer']],self.data], axis=1)


    def getTop5Products(self, user):
        try:
            items = self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index
        except KeyError:
            # If user doesn't exist praise an error and catch it. 
            errorMessage = "ERROR: Unable to recommend products to username '{}', as it doesn't not exist in the system!\n\
            Please try again with a user from 'Available Users' list provided above. ".format(user)
            return errorMessage       
        tfs=pd.read_pickle('tfidf.pkl')
        ##print(items)
        df=self.data[self.data.id.isin(items)]
        ##print(temp)
        X = tfs.transform(df['Review'].values.astype(str))
        df=df[['id']]
        df['prediction'] = self.model.predict(X)
        df['prediction'] = df['prediction'].map({'Postive':1,'Negative':0})
        df=df.groupby('id').sum()
        df['positive_percent']=df.apply(lambda x: x['prediction']/sum(x), axis=1)
        final_list=df.sort_values('positive_percent', ascending=False).iloc[:5,:].index  ### only taking 5 of the top predictions as required
        return self.data[self.data.id.isin(final_list)][['id', 'brand', 'categories', 'manufacturer', 'name']].drop_duplicates().to_json(orient="table")