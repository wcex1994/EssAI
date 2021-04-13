"""
This modeule is for cleaning user text and chunking it as specified. Preparing it for the BERT module. 
"""

import re
import json 
import spacy

# nlp = spacy.load('/opt/en_core_web_sm-2.1.0')
# nlp = spacy.load("en_core_web_sm")
# nlp = spacy.load('/tmp/en_core_web_sm/en_core_web_sm-2.0.0')
# nlp = spacy.load("/opt/en_core_web_sm-2.0.0")



nlp = spacy.load('/opt/en_core_web_sm-2.1.0')

def NER_replace(data):
    
    spc_words = list(nlp(data).ents)
    thistext = data
    print(spc_words)
    
    for ent in spc_words:
        print(ent)
        print(ent.label_)
        if ent.label_ in ['PERSON','FAC','GPE','LOC','ORG']:
            thistext = thistext.replace(ent.text,ent.label_)
    return thistext