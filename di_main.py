#!/usr/bin/python
"""Domain Identification: Modeling the Message"""

# import the necessary packages
import os
import sys
import argparse
import pandas as pd
import re
import string
import cPickle
import numpy as np
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import WordNetLemmatizer
from nltk import wordpunct_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# doc string
__author__ = "Sanchit Aggarwal"
__email__ = "sanchitagarwal108@gmail.com"


stopwords = set(sw.words('english'))
punctuation = set(string.punctuation)
lemmatizer = WordNetLemmatizer()
minlength = 3  # remove words smaller than 3
maxlength = 25 # remove words greater than 25


"""
# Function to construct argument parser
"""
def parseArguments():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training_datafile",required=False, help="path to training data file.")
    ap.add_argument("-m", "--modelpath",required=False, help="path to learned model file for testing.")
    ap.add_argument("-i", "--test_datafile",required=False, help="path to test data file.")
    ap.add_argument("-r", "--resultfile",required=False, help="result file name.")

    # parsing the arguments
    args = vars(ap.parse_args())
    # printing arguments
    for keys,values in args.items():
        print("Parsing Arguments: {} : {}".format(keys,values))

    if args["training_datafile"] is None and args["modelpath"] is None:
        ap.error('-- either of one ( training_datafile/modelpath) is required')
    if args["modelpath"] is not None and args["test_datafile"] is None:
        ap.error('-- both modelpath and test_datafile are required for testing')
    return args

"""
# Function to map tree bank corpus tags with wordnet tags
"""
def getWornetPOS(tag):
    tagMap = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }
    if tag[0] in tagMap.keys():
        return tagMap[tag[0]]
    else:
        return ''

"""
# Function to convert a raw message to a string of words
# The input is a single string (a raw message), and
# the output is a single string (a preprocessed message)
"""
def messageToWords( message ):
    message_text = BeautifulSoup(message,"html.parser").get_text() # remove html
    clean_message = re.sub("[^a-zA-Z]", " ", message_text) # remove non-letters
    words = []
    for word, tag in pos_tag(wordpunct_tokenize(clean_message)):
        word = word.lower() # convert to lower case
        word = word.strip()
        word = word.strip('_')
        word = word.strip('*')
        # remove stopword
        if word in stopwords:
            continue
        # ignore punctuation
        if all(char in punctuation for char in word):
            continue
        tag = getWornetPOS(tag)
        if tag == '':
            continue
        else:
            word = lemmatizer.lemmatize(word,tag)
        words.append(word)

    words = [w for w in words if minlength < len(w) < maxlength]
    return( " ".join( words ))

"""
# Function to get dataframe from data file
"""
def getDataFrame(datafile):
    # read the data file (training or testing)
    print "reading data file..."
    data_df = pd.read_csv(datafile, header=0, delimiter="\t", quoting=3)
    print "data shape:", data_df.shape
    print "data columns:", data_df.columns.values
    return data_df

"""
# Function to clean dataframe
"""
def cleanDataFrame(dataframe):
    # preprocessing of data into tokens/words
    print "pre-processing data..."
    for index, row in dataframe.iterrows():
        row['message'] =  messageToWords( row['message'] )
    return dataframe

"""
# Function to train the model for given pipeline
"""
def trainModel(training_set, pipeline):
    # Learning Model
    print "learning model"
    model = pipeline.fit(training_set["message"],  training_set["domain"])
    return model

"""
# Function to predict from domain of messages from learned model
"""
def predictModel(test_set,model):
    predicted = model.predict(test_set["message"])
    return predicted

"""
# Function to save the model
"""
def saveModel(model, filename="model.pkl"):
    # save the classifier
    with open(filename, 'wb') as fid:
        cPickle.dump(model, fid)

"""
# Function to read the model
"""
def readModel(filename="model.pkl"):
    model = None
    # load model
    if os.path.isfile(filename):
        with open(filename, 'rb') as fid:
            model = cPickle.load(fid)
    return model


if __name__ == '__main__':
    # parsing arguments
    args = parseArguments()

    # checking for training data or model
    if args["training_datafile"] is not None:
        print "----Training----"
        # get the data frame for training data
        original_training_df = getDataFrame(args["training_datafile"])
        clean_training_df = cleanDataFrame(original_training_df.copy())

        # split into training and validation set
        print "splitting data into training and validation set"
        training_set, validation_set = train_test_split(clean_training_df, test_size = 0.3)
        print training_set.shape
        print validation_set.shape

        # define the pipeline
        pipeline = Pipeline([
        # ('vect', CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None)),
        ('vect', CountVectorizer(ngram_range=(1, 4),token_pattern=r'\b\w+\b', analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None,  min_df=1)),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])
        # ('clf', MultinomialNB())])
        # ('classifier',         BernoulliNB(binarize=0.0)) ])


        # get training model
        model = trainModel(training_set, pipeline)

        # predict for validation set
        print "----Validation and Classification Report----"
        predicted = predictModel(validation_set,model)
        target_domains =  list(set(validation_set["domain"]))
        print(classification_report(validation_set["domain"], predicted, target_names=target_domains))

        # save model
        print "----Saving Model----"
        if args["modelpath"] is not None:
            saveModel(model, args["modelpath"])
        else:
            saveModel(model)

    if args["modelpath"] is not None and args["test_datafile"] is not None:
        print "----Testing----"
        # read model
        model =  readModel(args["modelpath"])
        # get data frame for test data
        original_test_df = getDataFrame(args["test_datafile"])
        clean_test_df = cleanDataFrame(original_test_df.copy())

        # predict for test set
        predicted = predictModel(clean_test_df,model)

        print original_test_df["message"][0]
        print clean_test_df["message"][0]
        # Save results to output object
        print "----Saving Results----"
        output = pd.DataFrame( data={"message":original_test_df['message'], "predicted":predicted} )
        if args["resultfile"] is not None:
            resultfile = args["resultfile"]
        else:
            resultfile = "result.tsv"
        output.to_csv(resultfile, index=False, sep='\t', quoting=3 )

# pipeline = Pipeline([('vect', CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None)),
#                            ('tfidf', TfidfTransformer()),
#                            ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])
#
#
# pipeline = Pipeline([
# ('count_vectorizer',   CountVectorizer(ngram_range=(1, 2))),
# ('tfidf', TfidfTransformer()),
# ('classifier',         BernoulliNB(binarize=0.0)) ])
#
#
# pipeline = Pipeline([
# ('count_vectorizer',   CountVectorizer(ngram_range=(1, 2))),
# ('tfidf', TfidfTransformer()),
# ('classifier', RandomForestClassifier(n_estimators = 100))])
