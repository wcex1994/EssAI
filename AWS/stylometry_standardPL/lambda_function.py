import json
import pandas as pd
from pipeline import *

def lambda_handler(event, context):
    test_run = 0
    if test_run == 1:
        data = [" It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of ORG, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way--in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only. There were a king with a large jaw and a queen with a plain face, on the throne of ORG; there were a king with a large jaw and a queen with a fair face, on the throne of ORG."]
    else : 
        data = event['data'] 
        
    print('before any trnsf')    
    user_data = pd.DataFrame(data, columns=['Text'])
    user_data["Author"] = "User"
    print(user_data)
    
    print('calling standard pipeline now')    
    user_data = standardPipeline(user_data, "Text")
    print(user_data.columns)
    user_data_sml = user_data[['Text', 'Author','avgWordLength', 'sentenceLengthByChar', 'specicalCharacterCount',
       'avgSyllablesPerWord', 'countFunctionalWords', 'hapaxLegemena',
       'honoreMeasureR', 'hapaxDisLegemena', 'sichelesMeasureS',
       'avgWordFrequencyClass', 'typeTokenRatio', 'brunetsMeasureW', 'yulesK',
       'simpsonsIndex', 'wordCount', 'uniqueWordCount', 'shannonEntropy' ]]

  # ['fleschCincadeGradeLevel', 'sentenceCount', 'fleschReadingEase', 'gunningFoxIndex', 'daleChallReadability']  
    result = user_data.reset_index().to_json(orient='records')
    print(result)
    
    return {
        'statusCode': 200,
        # 'body': json.dumps('Hello from Lambda!- StandarPL')
        'body': result
    }
