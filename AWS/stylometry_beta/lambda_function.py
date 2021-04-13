import json
import pandas as pd 
from boto3 import client as boto3_client

lambda_client = boto3_client('lambda')

def lambda_handler(event, context):
    # TODO implement
    status_code = 0
    msg = {}
    print(event)
    print(context)
   
    if 'requestContext' in event.keys():
        print('in if')
        in_txt_j = json.loads(event['body'])['text']
        print(type(in_txt_j))
        in_txt = pd.json_normalize(in_txt_j)
        print(type(in_txt))
    else: 
        print('i am in else')
         # in_txt = [{"index":0,"original-text":"She laughed again, as if she said something very witty, and held my hand for a moment, looking up into my face, promising that there was no one in the world she so much wanted to see","recreation":"toasters and waffles"},{"index":1,"original-text":" That was a way she had","recreation":"toasters and waffles"},{"index":2,"original-text":" She hinted in a murmur that the surname of the balancing girl was Baker","recreation":"toasters and waffles"},{"index":3,"original-text":" (I'\''ve heard it said that Daisy'\''s murmur was only to make people lean toward her; an irrelevant criticism that made it no less charming","recreation":"toasters and waffles"},{"index":4,"original-text":")\n\nAt any rate Miss Baker'\''s lips fluttered, she nodded at me almost imperceptibly and then quickly tipped her head back again—the object she was balancing had obviously tottered a little and given her something of a fright","recreation":"toasters and waffles"},{"index":5,"original-text":" Again a sort of apology arose to my lips","recreation":"toasters and waffles"},{"index":6,"original-text":" Almost any exhibition of complete self sufficiency draws a stunned tribute from me","recreation":"toasters and waffles"}]
        # in_txt = open ('input.json', "r") 
        in_txt = pd.read_json("input1.json")

    print(in_txt)
    
    in_txt_DF = pd.melt(in_txt, 
                    id_vars= ["index"],
                    value_vars = ["original_text", "recreation"], 
                    var_name= "author",
                    value_name = "Text")
    
    print('in text df')
    print(in_txt_DF)
                    
    in_txt_DF_Text = in_txt_DF['Text']
    
    print('in txt df text')
    print(in_txt_DF_Text)
    print(type(in_txt_DF_Text))
  

    txt_list = []
    
    for t in in_txt_DF_Text:
        txt_list.append(t)
    
    print('txt list')
    print(txt_list)
     
    clean1_res_bdy = txt_list
    # clean1_res_bdy = ["toasters and waffles","She laughed again, as if she said something very witty, and held my hand for a moment, looking up into my face, promising that there was no one in the world she so much wanted to see"]
    
    print('standard PL')
    stdPL_msg = {"data" : clean1_res_bdy, 'invocation_type' : 'Sync'}
    stdPL_res = lambda_client.invoke(FunctionName="stylometry_standardPL",
                                              InvocationType='RequestResponse',
                                              Payload=json.dumps(stdPL_msg))
    stdPL_res_bdy = json.load(stdPL_res['Payload'])['body']
    stdPL_res_json = json.loads(stdPL_res_bdy)
    stdPL_res_DF = pd.DataFrame(stdPL_res_json)
        
    print('Token PL')
    tknPL_msg = {"data" : clean1_res_bdy, 'invocation_type' : 'Sync'}
    tknPL_res = lambda_client.invoke(FunctionName="stylometry_tokenPL",
                                              InvocationType='RequestResponse',
                                              Payload=json.dumps(tknPL_msg))
    tklPL_res_bdy = json.load(tknPL_res['Payload'])['body']
    tklPL_res_json = json.loads(tklPL_res_bdy)
    tklPL_res_DF = pd.DataFrame(tklPL_res_json)
    
    stylometry_DF = pd.merge(stdPL_res_DF, tklPL_res_DF, on='Text', how='inner')
    stylometry_DF.drop(columns=['index_y', 'Author_y','Author_x'],inplace = True)
    stylometry_DF.rename(columns={'index_x': 'index'},inplace = True)
    
    print(stylometry_DF.columns)
    print(in_txt_DF.columns)
    
    return_DF = pd.merge(stylometry_DF, in_txt_DF, on='Text', how='inner')
    print(return_DF.columns)
    return_DF.drop(columns=['index_x'],inplace = True)
    return_DF.rename(columns={'index_y': 'index'},inplace = True)
    print(return_DF.columns)

    print(return_DF['author'])

    return_data = return_DF.to_json(orient= "records")

    
    # print(stdPL_res_DF)
    status_code = 200 
    msg = 'Working'

    return {'statusCode': status_code,
            'body': return_data,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                'Access-Control-Allow-Methods': 'OPTIONS,GET, POST,  PUT, DELETE',
                'Access-Control-Allow-Credentials': 'true',
                'Content-Type': 'application/json'
            }
            }

