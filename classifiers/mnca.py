import requests
import os
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import schedule
import datetime
import newspaper
import time
import mnca_train
from sklearn.externals import joblib

def extract_article(url):
    """Function that takes the url string of a news article and returns the
    title and text of the article as a Python dictionary. Built on top of
    Newspaper's article scraping & curation library."""
    link = newspaper.Article(url)
    link.download()
    link.parse()

    article = {}
    article["title"] = link.title
    article["text"] = link.text

    return(article)

def create_article(title, text):
    """Function that produces a Python dictionary containing manually-inputted
    title and text."""
    article = {}
    article["title"] = title
    article["text"] = text
    
    return(article)

def most_predictive_feats(filtered_text, clf, label, n=10):
    """Function returns most predictive words (by differential log_prob) in 
    filtered_text in support of a "Credible" or "Non-Credible" labeling. n 
    can be set to adjust the number of words returned."""
    features = pd.DataFrame()
    features["word"] = clf.named_steps['tfidf'].get_feature_names()
    features["tfidf"] = list(clf.named_steps['tfidf'].transform([filtered_text]).toarray()[0])
    features["log_prob_0"] = clf.named_steps['clf'].feature_log_prob_[0]
    features["log_prob_1"] = clf.named_steps['clf'].feature_log_prob_[1]
    labels = []
    for lp0, lp1 in zip(features["log_prob_0"], features["log_prob_1"]):
        if lp1 > lp0:
            labels.append(1)
        else:
            labels.append(0)
    features["label"] = labels
    features["log_prob_diff"] = abs(features["log_prob_1"] - features["log_prob_0"])
    features["weighted_log_prob_diff"] = features["tfidf"] * features["log_prob_diff"]
    features_c_sort = features[features["label"]==0].sort_values(by=["weighted_log_prob_diff"], ascending=False)
    features_nc_sort = features[features["label"]==1].sort_values(by=["weighted_log_prob_diff"], ascending=False)
    if label == "Non-Credible":
        word_features = list(features_nc_sort["word"].head(n))
    elif label == "Credible":
        word_features = list(features_c_sort["word"].head(n))
        
    return(word_features)

def classify_article(article):
    """Function accepts articles in the form of Python dictionaries
    containing the raw text (article['text']) and title (article['title']).
    It returns a Python dictionary containing the classification label
    (label), probability (label_prob), and a dictionary containing the
    word and tonal features that had the greatest impact on classification
    (interpretation)."""
    article['filtered_text'] = mnca_train.remove_shortwords(article['text'])
    article['sentences'] = mnca_train.split_into_sentences(article['text'])
    article['text_sentiment'] = mnca_train.sent_analysis(article['sentences'])
    article['title_sentiment'] = mnca_train.sent_analysis(article['title'], uoa="string")
    article['pct_char_quesexcl_title'] = mnca_train.pct_char_quesexcl(article['title'])
    article['pct_punc_quesexcl_text'] = mnca_train.pct_punct_quesexcl(article['text'])
    article['pct_allcaps_title'] = mnca_train.pct_allcaps(article['title'])
    
    with open('DONOTDELETE.json') as json_data:
        perform_statistics = json.load(json_data)

    ada_feats = [article['pct_allcaps_title'],
                 article['pct_punc_quesexcl_text'],
                 article['pct_char_quesexcl_title'],
                 article['text_sentiment'],
                 article['title_sentiment']]
    
    mnb_clf = joblib.load('mnb_clf.pkl')
    ada_clf = joblib.load('ada_clf.pkl')
    
    mnb_prob = mnb_clf.predict_proba([article['filtered_text']])
    ada_prob = ada_clf.predict_proba([np.asarray(ada_feats)])
    
    classification = {}
    classification["label_prob"] = [(((perform_statistics['mnb_accuracy']*mnb_prob[0][0])+(perform_statistics['ada_accuracy']*ada_prob[0][0]))/(perform_statistics['mnb_accuracy']+perform_statistics['ada_accuracy'])), (((perform_statistics['mnb_accuracy']*mnb_prob[0][1])+(perform_statistics['ada_accuracy']*ada_prob[0][1]))/(perform_statistics['mnb_accuracy']+perform_statistics['ada_accuracy']))]
    if classification["label_prob"][1] > classification["label_prob"][0]:
        classification["label"] = "Non-Credible"
    else:
        classification["label"] = "Credible"
            
    interpretation = {}
    interpretation["word_feats"] = most_predictive_feats(article["filtered_text"], mnb_clf, classification["label"], n=10)
    tone_feats = {}
    
    tone_feat_classifications = []
    if abs(article['pct_allcaps_title'] - perform_statistics['mean_pct_allcaps_title_noncred']) < abs(article['pct_allcaps_title'] - perform_statistics['mean_pct_allcaps_title_cred']):
        tone_feat_classifications.append(1)
    else:
        tone_feat_classifications.append(0)
    
    if abs(article['pct_punc_quesexcl_text'] - perform_statistics['mean_pct_punc_text_noncred']) < abs(article['pct_punc_quesexcl_text'] - perform_statistics['mean_pct_punc_text_cred']):
        tone_feat_classifications.append(1)
    else:
        tone_feat_classifications.append(0)
    
    if abs(article['pct_char_quesexcl_title'] - perform_statistics['mean_pct_punc_title_noncred']) < abs(article['pct_char_quesexcl_title'] - perform_statistics['mean_pct_punc_title_cred']):
        tone_feat_classifications.append(1)
    else:
        tone_feat_classifications.append(0)
        
    if abs(article['text_sentiment'] - perform_statistics['mean_sent_score_text_noncred']) < abs(article['text_sentiment'] - perform_statistics['mean_sent_score_text_cred']):
        tone_feat_classifications.append(1)
    else:
        tone_feat_classifications.append(0)
        
    if abs(article['title_sentiment'] - perform_statistics['mean_sent_score_title_noncred']) < abs(article['title_sentiment'] - perform_statistics['mean_sent_score_title_cred']):
        tone_feat_classifications.append(1)
    else:
        tone_feat_classifications.append(0)
        
    if (tone_feat_classifications[0]*ada_clf.feature_importances_[0] + tone_feat_classifications[1]*ada_clf.feature_importances_[1] + tone_feat_classifications[2]*ada_clf.feature_importances_[2])/( ada_clf.feature_importances_[0] + ada_clf.feature_importances_[1] + ada_clf.feature_importances_[2]) > 0.5:
        convention_pred = "Non-Credible"
    else:
        convention_pred = "Credible"
        
    if (tone_feat_classifications[3]*ada_clf.feature_importances_[3] + tone_feat_classifications[4]*ada_clf.feature_importances_[4])/( ada_clf.feature_importances_[3] + ada_clf.feature_importances_[4]) > 0.5:
        sentiment_pred = "Non-Credible"
    else:
        sentiment_pred = "Credible"
    
    if convention_pred == classification["label"]:
        tone_feats["convention"] = convention_pred
        
    if sentiment_pred == classification["label"]:
        tone_feats["sentiment"] = sentiment_pred
    
    interpretation["tone_feats"] = tone_feats
   
    classification["interpretation"] = interpretation
        
    return(classification)

    
    
    