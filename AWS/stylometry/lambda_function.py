import json
from boto3 import client as boto3_client
import pandas as pd

lambda_client = boto3_client('lambda')

# import spacy
# nlp = spacy.load('/opt/en_core_web_sm-2.1.0')

# from metrics import *

def lambda_handler(event, context):
    # test_run = 1
    print(event)
    
    if 'body' in event:
        in_txt = json.loads(event['body'])['in_txt']   
        author = json.loads(event['body'])['author_name'] 
        stage  = json.loads(event['body'])['stage'] 
    else:
        in_txt = event['in_txt']   
        author = event['author_name']
        stage  = event['stage'] 

    
    if stage == 'test':
        f_ex_out = open ('example_output_new.json', "r") 
        example_output = json.loads(f_ex_out.read()) 

        return {
            'statusCode': 200,
            'body': json.dumps(example_output),
            'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                    'Access-Control-Allow-Methods': 'OPTIONS,GET, POST,  PUT, DELETE',
                    'Access-Control-Allow-Credentials': 'true',
                    'Content-Type': 'application/json'
                }
            }

    print(in_txt)
    print(author)
    
    if author.upper() == 'DICKENS':
        f_author = open ('top5_Dickens.json', "r") 
        author_metrics = ["avgWordFrequencyClass","countFunctionalWords","daleChallReadability","simpsonsIndex","specicalCharacterCount"]
    elif author.upper() == 'FITZGERALD':
        f_author = open ('top5_Fitzgerald.json', "r") 
        author_metrics = ["avgWordFrequencyClass","countFunctionalWords","shannonEntropy","specicalCharacterCount","typeTokenRatio"]
    elif author.upper() == 'TWAIN':
        f_author = open ('top5_Twain.json', "r") 
        author_metrics = ["avgSyllablesPerWord","avgWordFrequencyClass","avgWordLength","countFunctionalWords","specicalCharacterCount"]

    else:
        f_author = open ('top5_Austen.json', "r") 
        author_metrics = ["avgWordFrequencyClass","brunetsMeasureW","hapaxLegemena","typeTokenRatio","yulesK"]

        
    top5_author = json.loads(f_author.read())  

    body = {"in_txt" : in_txt ,
            "chunk" : 'text'
    }
    clean1_msg = {"body" : body , 'invocation_type' : 'Sync'}
    clean1_res = lambda_client.invoke(FunctionName="userTextClean1",
                                              InvocationType='RequestResponse',
                                              Payload=json.dumps(clean1_msg))
    print(clean1_res)
    clean1_res_bdy = json.load(clean1_res['Payload'])['body']['Text']

    print(clean1_res_bdy)
    
    stdPL_msg = {"data" : clean1_res_bdy, 'invocation_type' : 'Sync'}
    print(stdPL_msg)
    stdPL_res = lambda_client.invoke(FunctionName="stylometry_standardPL",
                                              InvocationType='RequestResponse',
                                              Payload=json.dumps(stdPL_msg))
    print('standard PL')
    stdPL_res_bdy = json.load(stdPL_res['Payload'])['body']
    print(stdPL_res_bdy)
    stdPL_res_json = json.loads(stdPL_res_bdy)
    print(stdPL_res_json)
    stdPL_res_DF = pd.DataFrame(stdPL_res_json)
    

    
    tknPL_msg = {"data" : clean1_res_bdy, 'invocation_type' : 'Sync'}
    tknPL_res = lambda_client.invoke(FunctionName="stylometry_tokenPL",
                                              InvocationType='RequestResponse',
                                              Payload=json.dumps(tknPL_msg))
    
    print('Token PL')
    tklPL_res_bdy = json.load(tknPL_res['Payload'])['body']
    tklPL_res_json = json.loads(tklPL_res_bdy)
    tklPL_res_DF = pd.DataFrame(tklPL_res_json)


    stylometry_DF = pd.merge(stdPL_res_DF, tklPL_res_DF, on='Text', how='inner')
    stylometry_DF.drop(columns=['index_y', 'Author_y'],inplace = True)
    stylometry_DF.rename(columns={'index_x': 'index', 'Author_x': 'Author'},inplace = True)

    print(stdPL_res_DF.columns)
    print(tklPL_res_DF.columns)   
    print(stylometry_DF.columns)


    stylometry_res = pd.melt(stylometry_DF, 
                  id_vars=['index','Author', "Text"],
                  value_vars = author_metrics,
                  var_name = "Metric",
                  value_name = "Value", ignore_index=False)
    print(stylometry_res)
    
    # # This step is not required. 
    # # stylometry_stats = input_to_stats(stylometry_DF, austen_metrics)
    # # result_DF = stylometry_stats.append(top5_Austen_DF, ignore_index=True, sort=False)
    
    result_json = {
        # 'user_metrics' : stylometry_res.to_json(orient='records') ,
        'user_metrics' : stylometry_res.to_dict('records'), 
        'author_top5' : top5_author
    }

    print(result_json)
   
    return {
        'statusCode': 200,
        # 'body': json.dumps('Hello from Lambda!- Stylometry'),
        'body': json.dumps(result_json),
        'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                'Access-Control-Allow-Methods': 'OPTIONS,GET, POST,  PUT, DELETE',
                'Access-Control-Allow-Credentials': 'true',
                'Content-Type': 'application/json'
            }
    }









# import boto3

# import pickle
# import math
# import string
# import pandas as pd

# import nltk
# import spacy 
# import collections as coll

# from TextPrep import *

# # nltk.download('cmudict')
# # nltk.download('stopwords')


# from nltk.corpus import cmudict
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.tokenize import word_tokenize
# nltk.data.path.append("/tmp")
# nltk.download("cmudict", download_dir = "/tmp")

# s3 = boto3.resource('s3')

# def lambda_handler(event, context):
    

#     familiarWords = pickle.loads(s3.Bucket("w210essai").Object("dale-chall.pkl").get()['Body'].read())
#     print(type(familiarWords))
    
#     # text = """In the midst of them, the hangman, ever busy and ever worse than useless, was in constant requisition; now, stringing up long rows of miscellaneous criminals; now, hanging a housebreaker on Saturday who had been taken on Tuesday; now, burning people in the hand at Newgate by the dozen, and now burning pamphlets at the door of Westminster Hall; to-day, taking the life of an atrocious murderer, and to-morrow of a wretched pilferer who had robbed a farmer’s boy of sixpence."""
    
#     # t = NER_replace(text)
#     # print(t)
    
#     # TODO implement
#     return {
#         'statusCode': 200,
#         'body': json.dumps('Hello from Lambda!')
#     }

