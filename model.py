# import libraties
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
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
        self.model = pickle.load(open('logistic_regression.pkl','rb'))
        self.raw_data = pd.read_csv("sample30.csv")
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
        temp=self.data[self.data.id.isin(items)]
        X = tfs.transform(temp['Review'].values.astype(str))
        temp=temp[['id']]
        temp['prediction'] = self.model.predict(X)
        temp['prediction'] = temp['prediction'].map({'Postive':1,'Negative':0})
        temp=temp.groupby('id').sum()
        temp['positive_percent']=temp.apply(lambda x: x['prediction']/sum(x), axis=1)
        final_list=temp.sort_values('positive_percent', ascending=False).iloc[:5,:].index
        print(len(final_list))
        return self.data[self.data.id.isin(final_list)][['id', 'brand', 'categories', 'manufacturer', 'name']].drop_duplicates().to_json(orient="table")