#Importing necessary libraries
import os
import numpy as np
import pandas as pd
import re
from numpy import float32 as REAL
import pickle
from gensim import utils
import numpy as np
from collections import defaultdict
import ast
from scipy.spatial.distance import cdist as scipy_cdist
import matplotlib.pyplot as plt
import faiss
import tensorflow_hub as hub
import time
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, classification_report
from UTILS.Similarity_search_GUSE_FAISS import *
from UTILS.custom_preprocessing_function import *

###############################Main Function For Checking Intent #########################################
def check_intent(df, kpi_df, kpi_type):
    #Provide threshold for different baselines
    faiss_similarity_threshold_buyer_seller = 0.448
    faiss_similarity_threshold_buyer= 0.447
    faiss_similarity_threshold_seller = 0.448
    #Loading the baselines for checking buyer, seller, buyer_seller
    d=defaultdict()
    kpi_df=kpi_df[kpi_df.sub_type==kpi_type]
    d['kpi_id']=kpi_df.id.iloc[0]
    d['kpi_name']=kpi_df.name.iloc[0]
    actor_type=kpi_df.actor_type.iloc[0]
    all_baselines=ast.literal_eval(kpi_df.processed_baseline.iloc[0])
    buyer_seller_baselines=all_baselines['buyer_seller']
    buyer_baselines=all_baselines['buyer_baselines']
    seller_baselines=all_baselines['seller_baselines']
    
    res=[]
    id = df['Id'].unique()[0]
    query=df.FeedText.tolist()
    if any(df.ProfileName.isna()):
        user_name="Name-Unavailable"
    else:
        try:
            user_name =  df['ProfileName'].unique()[0]
        except IndexError:
            user_name='None'
    #Creating indexes for similarity search
    index_buyer_seller=faiss_index_creation(buyer_seller_baselines)
    index_buyer=faiss_index_creation(buyer_baselines)
    index_seller=faiss_index_creation(seller_baselines)
    
    #Similarity Search for three baselines
    similarity_results_buyer_seller = search_faiss(df,buyer_seller_baselines,index_buyer_seller,faiss_similarity_threshold_buyer_seller)
    check_met_similarity_results_buyer_seller=[sm[2] for sm in similarity_results_buyer_seller if  sm[2]>0]
    
    similarity_results_buyer = search_faiss(df,buyer_baselines,index_buyer,faiss_similarity_threshold_buyer)
    check_met_similarity_results_buyer=[sm[2] for sm in similarity_results_buyer if  sm[2]>0]

    similarity_results_seller = search_faiss(df,seller_baselines,index_seller,faiss_similarity_threshold = faiss_similarity_threshold_seller)
    check_met_similarity_results_seller=[sm[2] for sm in similarity_results_seller if  sm[2]>0]
    
    #Main conditions check for intent classifications
    if len(check_met_similarity_results_buyer_seller)> 0:
        res.append((id, user_name, 'buyer_seller'))
    elif len(check_met_similarity_results_buyer) > 0:
        res.append((id, user_name, 'buyer'))
    elif len(check_met_similarity_results_seller) > 0:
        res.append((id, user_name, 'seller'))
    else:
        res.append((id, user_name, 'neutral'))
    #Storing the results
    d['score'] =  res
    return d