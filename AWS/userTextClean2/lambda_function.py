# arn:aws:lambda:us-east-2:113088814899:layer:Klayers-python37-spacy:1
# arn:aws:lambda:us-east-2:113088814899:layer:Klayers-python37-spacy:28

import json

from TextPrep import *

def lambda_handler(event, context):
    # TODO implement
    # print(event)
    if 'data' in event.keys():
        text = NER_replace(event['data'])
    else: 
        text = """In the midst of them, the hangman, ever busy and ever worse than useless, was in constant requisition; now, stringing up long rows of miscellaneous criminals; now, hanging a housebreaker on Saturday who had been taken on Tuesday; now, burning people in the hand at Newgate by the dozen, and now burning pamphlets at the door of Westminster Hall; to-day, taking the life of an atrocious murderer, and to-morrow of a wretched pilferer who had robbed a farmer’s boy of sixpence."""

    
    t = NER_replace(text)
    print(t)
    print(json.dumps(t))
    
    return {
        'statusCode': 200,
        # 'body': json.dumps(t)
        'body': t
    }
