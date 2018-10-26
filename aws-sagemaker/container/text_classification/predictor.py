# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import StringIO
import sys
import signal
import traceback

import flask

import numpy as np
import pandas as pd
import boto3
import botocore

from keras.preprocessing import sequence
from keras.models import load_model
from keras import backend as K

from keras.models import Model
from keras.layers import Dropout, Dense, Input, Embedding, Concatenate, Reshape, Conv2D, MaxPool2D, Flatten

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
dict_path = os.path.join(prefix, 'output')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            cls.model = load_model(os.path.join(model_path, 'text_clas.hdf5'))
            # cls.model._make_prediction_function()
            # cls.model = build_model()
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        sess = K.get_session()
        with sess.graph.as_default():
            clf = cls.get_model()
            return clf.predict(input)
        '''
        clf = build_model()
        prediction = model.predict(input)
        return prediction
        '''
    
    
num_classes = 14
maxLen = 350
sequence_length = 350
embedding_dim = 100
num_filters = 256
filter_sizes = [3,4,5]
drop = 0.5
learning_rate = 0.0001
vocab_size = 1037934


def transform_data(dataset):
    # download dictionary from s3
    Bucket = 'sagemaker-1017'
    Key = 'dictionary.json'
    outPutname = 'dict.json'
    s3 = boto3.resource('s3')
    try:
        s3.Bucket(Bucket).download_file(Key, outPutname)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print('File does not exist.')
        else:
            raise
    with open('dict.json') as json_file:
        dictionary = json.load(json_file)
    '''
    # get dictionary file 
    infiles = [ os.path.join(dict_path, file) for file in os.listdir(dict_path) ]
    for file in infiles:
        names = file.split('.')
        if names[-1] == 'json':
            dic = file
    # load dictionary
    with open(dic) as f:
        dictionary = json.load(dic)
    '''    
    # transform raw data to one-hot encoding
    text_data = []
    text = []
    for word in dataset:
        index = 0
        if word in dictionary:
            index = dictionary[word]
        text.append(index)
    # text_data.append(text)
        
    # text_data = sequence.pad_sequences(text, 350)
    # text = text[:350]
    text_data.append(text)
    text_data = sequence.pad_sequences(text_data, 350)
    print(text_data)
    # text_data = np.array(text_data)
    return text_data
    

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as string, convert
    it to a ndarray for internal use and then convert the predictions back to csv
    """
    data = None

    # Convert from string to ndarray
    if flask.request.content_type == 'text/csv':
        data = flask.request.data#.decode('utf-8')
        # print('Type of decoded data: ', type(data))
        data = data.split()#list(data)
        # print('Length of data: ', len(data))
        data = transform_data(data)
        # print('Type of transformed data: ', type(data))
        # print('Shape of transformed data: ', data.shape)
        # # original method to convert data
        # data = flask.request.data.decode('utf-8')
        # X = transform_data(data)
        # s = StringIO.StringIO(X)
        # data = pd.read_csv(s, header=None)
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.shape[0]))

    # Do the prediction
    predictions = ScoringService.predict(data)
    # predictions = ScoringService.predict(transformed_data)

    # Convert from numpy back to csv
    '''
    out = StringIO.StringIO()
    pd.DataFrame(predictions).to_csv(out, header=False, index=False)
    # pd.DataFrame({'results':predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()
    '''
    # each position is scores of corresponding class
    labels = ['APPLICATION', 'BILL', 'BILL BINDER', 'BINDER', 'CANCELLATION NOTICE', 'CHANGE ENDORSEMENT',\
             'DECLARATION', 'DELETION OF INTEREST', 'EXPIRATION NOTICE', 'INTENT TO CANCEL NOTICE',\
             'NON-RENEWAL NOTICE', 'POLICY CHANGE', 'REOMSTATEMENT NOTICE', 'RETURNED CHECK']
    result = []
    for l in predictions:
        l = list(l)
        print('Raw prediction: ', l)
        maxIndex = l.index(max(l))
        print('String prediction: ', labels[maxIndex])
        result.append(labels[maxIndex])
    
    print('Result: ', result)
    out = StringIO.StringIO()
    out.write(result)
    pd.DataFrame(result).to_csv(out, header=False, index=False)
    res = out.getvalue()
    
    return flask.Response(response=res, status=200, mimetype='text/csv')
