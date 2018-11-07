from chalice import Chalice
from chalice import BadRequestError
import os, boto3, ast, csv
import numpy as np
import flask

try:
    from urlparse import urlparse, parse_qs
except ImportError:
    from urllib.parse import urlparse, parse_qs

app = Chalice(app_name='predictor')
app.debug=True

@app.route('/', methods=['POST'], content_types=['application/x-www-form-urlencoded'])
def index():
    # body = app.current_request.json_body
    d = parse_qs(app.current_request.raw_body)

    # to csv
    try:
        for k, v in d.iteritems():
            print(k)
            my_dict = {k:str(v[0])}
        # my_dict = {k:str(v[0]) for k, v in d.iteritems()}
    except AttributeError:
        for k, v in d.items():
            print(k)
            my_dict = {k:str(v[0])}
        # my_dict = {k:str(v[0]) for k, v in d.items()}
    
    document = my_dict['content']
    endpoint = os.environ['ENDPOINT_NAME']

    topK = 3    # get the top 3 possible predictions

    # print('%s %d' % (endpoint, topK))
    
    runtime = boto3.Session().client(service_name='sagemaker-runtime', region_name='us-east-1')
    response = runtime.invoke_endpoint(EndpointName=endpoint, ContentType='text/csv', Body=document)
    probs = response['Body'].read().decode()    # not sure if we need to decode

    #return {'response': str(probs)}
    return str(probs)
