import json 
import spacy
 
# nlp = spacy.load("en_core_web_sm")

nlp = spacy.load('/opt/en_core_web_sm-2.1.0')


def lambda_handler(event, context):
    # TODO implement
    
    print(event)
    
    if 'body' in event:
        print('in event body')
        in_txt = json.loads(event['body'])['in_txt'] 
        keep_pos = json.loads(event['body'])['keep_pos'] 
        # author = json.loads(event['body'])['author_name'] 
        stage  = json.loads(event['body'])['stage']
        
        print(in_txt)
        print(keep_pos)
        print(stage)
        
        

    else:
        # in_txt = event['in_txt']   
        # author = event['author_name']
        # stage  = event['stage'] 
        in_txt = '''
            Most of the big shore places were closed now and there were hardly any lights except the shadowy, moving glow of a ferryboat across the Sound. And as the moon rose higher the inessential houses began to melt away until gradually I became aware of the old island here that flowered once for Dutch sailors' eyes—a fresh, green breast of the new world. Its vanished trees, the trees that had made way for Gatsby's house, had once pandered in whispers to the last and greatest of all human dreams; for a transitory enchanted moment man must have held his breath in the presence of this continent, compelled into an aesthetic contemplation he neither understood nor desired, face to face for the last time in history with something commensurate to his capacity for wonder.
            
            And as I sat there brooding on the old, unknown world, I thought of Gatsby's wonder when he first picked out the green light at the end of Daisy's dock. He had come a long way to this blue lawn and his dream must have seemed so close that he could hardly fail to grasp it. He did not know that it was already behind him, somewhere back in that vast obscurity beyond the city, where the dark fields of the republic rolled on under the night.
            
            Gatsby believed in the green light, the orgastic future that year by year recedes before us. It eluded us then, but that's no matter—tomorrow we will run faster, stretch out our arms farther. . . . And one fine morning—
        '''
        keep_pos = ["PUNCT", "CCONJ"]
        stage = 'test'
    

        
    resp = generate_hint(in_txt, keep_pos) 
    resp_json = json.loads(resp)
    print(resp)
    print(resp_json)
    
    # return {
    #     'statusCode': 200,
    #     'body': json.dumps(resp_json)
    #     }
    
    return {
        'statusCode': 200,
        'body': json.dumps(resp_json),
        'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                'Access-Control-Allow-Methods': 'OPTIONS,GET, POST,  PUT, DELETE',
                'Access-Control-Allow-Credentials': 'true',
                'Content-Type': 'application/json'
            }
        }

def blankout_sentence(sentences, keep_pos):
    words = []
    for token in sentences:
        if (token.pos_ in keep_pos):
            words.append(token.text)
        else:
            words.append("-" * (len (token.text)))
        words.append(token.whitespace_)
    return "".join(words)
    
                
def blankout_text(doc, keep_pos):
    return [{"index" : i,
             "sentence" : sent.text,
             "blanked" : blankout_sentence(sent, keep_pos)
            } 
            for i, sent in enumerate(doc.sents)]

def generate_hint(text, keep_pos):
    parsed = nlp(text)
    blanked = blankout_text(parsed, keep_pos)
    return json.dumps(blanked, indent=4, sort_keys=True)


# test_str = '''
# Most of the big shore places were closed now and there were hardly any lights except the shadowy, moving glow of a ferryboat across the Sound. And as the moon rose higher the inessential houses began to melt away until gradually I became aware of the old island here that flowered once for Dutch sailors' eyes—a fresh, green breast of the new world. Its vanished trees, the trees that had made way for Gatsby's house, had once pandered in whispers to the last and greatest of all human dreams; for a transitory enchanted moment man must have held his breath in the presence of this continent, compelled into an aesthetic contemplation he neither understood nor desired, face to face for the last time in history with something commensurate to his capacity for wonder.

# And as I sat there brooding on the old, unknown world, I thought of Gatsby's wonder when he first picked out the green light at the end of Daisy's dock. He had come a long way to this blue lawn and his dream must have seemed so close that he could hardly fail to grasp it. He did not know that it was already behind him, somewhere back in that vast obscurity beyond the city, where the dark fields of the republic rolled on under the night.

# Gatsby believed in the green light, the orgastic future that year by year recedes before us. It eluded us then, but that's no matter—tomorrow we will run faster, stretch out our arms farther. . . . And one fine morning—

# '''
# # =======================================================
# # Use Case
# # =======================================================
# # You will get a JSON that has keys called "text" and "keep_pos" (part of speech)
# # If you use them in the below, it will return a JSON of the proper form.
# # generate_hint(test_str, ["PUNCT", "CCONJ"])




# import json


