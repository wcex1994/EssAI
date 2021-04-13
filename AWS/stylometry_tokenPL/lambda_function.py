import json
from tokens import *
from metrics import * 

import sys
sys.setrecursionlimit(1500)


def lambda_handler(event, context):
    test_run = 0
    if test_run == 1:
        data = ['I went to the beach and stretched myself out.A ferryboat trudged across the river, devoid of any other lights.']
                # 'The moon rose and I become aware of the ', 'I like to eat eggs and toast.', 
                # 'And fish - fish is very tasty.', 
                # '''Do I like eggs or toast or fish more? It's difficult to say. But I think I like fish more.''']
    else : 
        data = event['data'] 
    
    user_data = pd.DataFrame(data, columns=['Text'])
    user_data["Author"] = "User"
 
    user_data = tokenPipeline(user_data, "Text")
    print('user_data')
    print(user_data.columns)
    user_data.drop(columns=['token'],inplace = True)
    print('user_data-col drp')
    print(user_data.columns)
    
    cols = user_data.columns
    # user_data_sml = user_data[['Text', 'Author', 'partOfSpeech', 'posEntropy', 'ADJ', 'ADP', 'ADV', 'CCONJ', 'DET', 'NOUN', 'PART', 'PRON', 'PUNCT', 'VERB']]
    user_data_sml = user_data[cols]

    result = user_data_sml.reset_index().to_json(orient='records')
    # result = input_to_stats(user_data, METRICS)

    return {
        'statusCode': 200,
        # 'body': json.dumps('Hello from Lambda!- TokenPL')
        'body': result
    }
