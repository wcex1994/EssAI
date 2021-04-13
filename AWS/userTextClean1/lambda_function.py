import json
import re
import numpy
import pandas as pd
import nltk
import requests
import time

from boto3 import client as boto3_client
from datetime import datetime
import json

lambda_client = boto3_client('lambda')

from TextPrep import *

# # nltk.download('punkt')
# # tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# from nltk.tokenize import word_tokenize
# nltk.data.path.append("/tmp")
# nltk.download("punkt", download_dir = "/tmp")
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



def lambda_handler(event, context):
    print('UserTextClean invoked. This is step1')
    print(context)
    print(event)
   

    test_run = 0

    # assign user_text and chunk_size from the parameters
    if test_run == 0:
        in_txt = event['body']['in_txt']
        chunk = event['body']['chunk']
    else: 
        in_txt = """ It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way--in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only. There were a king with a large jaw and a queen with a plain face, on the throne of England; there were a king with a large jaw and a queen with a fair face, on the throne of France."""
        chunk = 'text'
        
    if chunk == 'sentence':
    # split by sentence 
        print('I AM IN IF')
        dict_chunk = Text2Dict_sentence(in_txt)['Text']
    else : 
        print('I AM IN ELSE')
        dict_chunk = Text2Dict(in_txt)['Text']


    print(' I AM BACK FROM TEXTDICT')
    # print(type(dict_chunk))
    # print(dict_chunk)
    r_text = [] 
    o_text = []
    
    # invoke UserTextClean - step2. this step applies spacy for NER replace
    for d in dict_chunk:
        # print(d) 
        msg = {"data" : d, 'invocation_type' : 'Sync'}
        invoke_response = lambda_client.invoke(FunctionName="userTextClean2",
                                              InvocationType='RequestResponse',
                                              Payload=json.dumps(msg))
 
        lambda_response = json.load(invoke_response['Payload'])['body']
        r_text.append(lambda_response)
        o_text.append(d)

    return_json = {
        "Text": r_text,
        "Text_Original" : o_text
    }
        
    print("****** Just before return******")
    print(r_text)
    print(return_json)
    
    return {
        'statusCode': 200,
        'body': return_json 
    }







# def lambda_handler(event, context):
#     # TODO implement
#     msg = {"key":"new_invocation", 'invocation_type' : 'Sync'}
#     invoke_response = lambda_client.invoke(FunctionName="First_Lambda",
#                                           InvocationType='RequestResponse',
#                                           Payload=json.dumps(msg))
#     print(invoke_response)
    
#     lambda_response = json.load(invoke_response['Payload'])
    
#     print(lambda_response)


# 'http://52.138.105.139:80/api/v1/service/jane256/score'    
    
    
    # return {
    #     'statusCode': 200,
    #     'body': 'working'
    # }
