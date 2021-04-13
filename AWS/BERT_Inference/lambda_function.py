import json
from boto3 import client as boto3_client
import pandas as pd
import time 
import requests 
lambda_client = boto3_client('lambda')

import requests
import json
import time


def lambda_handler(event, context):
    # 1 to just return test response ; 2 to process text and then return return test response  ; 0 actual run
    print(event)
    
    if 'body' in event:
        in_txt = json.loads(event['body'])['in_txt']   
        author = json.loads(event['body'])['author_name'] 
        stage  = json.loads(event['body'])['stage']
        test_run = 0
         
    else:
        in_txt = event['in_txt']   
        author = event['author_name']
        stage  = event['stage'] 
        test_run = 0
        
    

    
    if stage == 'test':
        f_ex_out = open ('example_output.json', "r") 
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

    body = {"in_txt" : in_txt ,
            "chunk" : 'sentence'
    }
    
    clean1_msg = {"body" : body , 'invocation_type' : 'Sync'}
    clean1_res = lambda_client.invoke(FunctionName="userTextClean1",
                                              InvocationType='RequestResponse',
                                              Payload=json.dumps(clean1_msg))
 
    print(clean1_res)
    
    clean1_res_body_pl = json.load(clean1_res['Payload'])['body']
    clean1_res_bdy = clean1_res_body_pl['Text']    
    original_Text = clean1_res_body_pl['Text_Original']  


    
    print(clean1_res_bdy)

    model_json = {
          "Text": clean1_res_bdy,
          "MAX_LEN" : 0,
          "TRAIN_SIZE": 0.8
        }
    
    # if test_run == 2:
    #     resp = [
    #         {"Text": "this is a test.", "probpercentage": "0.00128"}, 
    #         {"Text": "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters. My dear Mr. PERSON, said his lady to him one day, have you heard that FAC is let at last? Mr. PERSON replied that he had not. But it is, returned she; for Mrs. PERSON has just been here, and she told me all about it. Mr. PERSON made no answer. Do you not want to know who has taken it? cried his wife impatiently. _You_ want to tell me, and I have no objection to hearing it. This was invitation enough.", "probpercentage": "99.98531"}
    #         ]
        
    if test_run == 0:    
        # Model Inference - BERT
        print('model inference')
        
        if author.upper() == 'TWAIN':
            scoring_uri = 'http://52.252.79.133:80/api/v1/service/mark128/score'
        # elif author.upper() == 'DICKENS':
        #     scoring_uri = 'http://52.152.186.230:80/api/v1/service/charles256n/score'
        # elif author.upper() == 'FITZGERALD':
        #     scoring_uri = 'http://52.152.186.230:80/api/v1/service/fitzgerald128/score'
       
        else:
            scoring_uri = 'http://52.152.186.230:80/api/v1/service/jane256/score'
            # # Interpretability - most important sentence
            # impSent_uri = 'http://52.138.105.139:80/api/v1/service/jane256sent/score'
        
    # scoring_uri_jane = ‘http://52.138.105.139:80/api/v1/service/jane256/score’
    # scoring_uri_mark = ‘http://52.138.105.139:80/api/v1/service/mark128/score’
    
    # scoring_uri_charles = ‘http://52.152.186.230:80/api/v1/service/charles256n/score’
    # scoring_uri_fitzgerald = ‘http://52.152.186.230:80/api/v1/service/fitzgerald128/score’

    # New links:
    # scoring_uri_jane = ‘http://52.152.186.230:80/api/v1/service/jane256/score’
    # scoring_uri_mark = ‘http://52.152.186.230:80/api/v1/service/mark128/score’

        # Convert to JSON string
        input_data = json.dumps(model_json)
        
        print('*****INPUT JSON*****')
        print(input_data)
    
        # Set the content type
        headers = {'Content-Type': 'application/json'}
    
        # If authentication is enabled, set the authorization header
        #headers['Authorization'] = f'Bearer {key}'
        
        since = time.time()
        # resp = requests.post(scoring_uri, input_data, headers=headers)
    
        try:
            # Make the request and display the response
            resp = requests.post(scoring_uri, input_data, headers=headers)
            print(json.loads(resp.text)[0]) # for scoring_uri
            time_elapsed = time.time() - since
            print('Predict complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
            print(type(resp))
            resp_json = json.loads(resp.text)
            resp_DF = pd.DataFrame(resp_json)
            resp_DF.rename(columns={'Text': 'Text_modified'},inplace = True)
            resp_DF.rename(columns={'probpercentage': 'Probability'},inplace = True)
            resp_DF['Author'] = 'User'
            resp_DF['Text'] = original_Text
            print(resp_DF)
            # resp_DF["ProbabilityRank"] = resp_DF["Probability"].rank()
            resp_DF = resp_DF.reset_index()
            resp_DF["temp"] = resp_DF["Probability"].astype(float) - resp_DF["index"].astype(float)*0.001
            resp_DF["ProbabilityRank"] = resp_DF["temp"].rank() - 1
            print(resp_DF)
        except ConnectionError:
            sc = 400
            raise Exception('BERT is down. ConnectionError')
        except: 
            sc = 400
            raise Exception('Some other error')

        sc = 200
        # resp_json = resp_DF.reset_index().to_json(orient='records')
        resp_json = resp_DF.to_json(orient='records')
                    
    return {
            'statusCode': sc,
            'body': resp_json,
            'headers': {
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                        'Access-Control-Allow-Methods': 'OPTIONS,GET, POST,  PUT, DELETE',
                        'Access-Control-Allow-Credentials': 'true',
                        'Content-Type': 'application/json'
                        }
                }
    
    # return {
    #     'statusCode': 200,
    #     'body': json.dumps('Hello from Lambda!- BERT_Inference'),
    #     'headers': {
    #             'Access-Control-Allow-Origin': '*',
    #             'Access-Control-Allow-Headers': 'Content-Type,Authorization',
    #             'Access-Control-Allow-Methods': 'OPTIONS,GET, POST,  PUT, DELETE',
    #             'Access-Control-Allow-Credentials': 'true',
    #             'Content-Type': 'application/json'
    #         }
    # }
