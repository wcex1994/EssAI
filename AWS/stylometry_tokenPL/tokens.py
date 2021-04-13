#=================================================================
# Libraries
#=================================================================   

from utils import * 
# from scipy import stats as s
import pandas as pd

import numpy as np
from math import log, e

# This language model will be used for tokenizing and pos tagging
# import spacy
# nlp = spacy.load("en_core_web_sm")
import spacy
nlp = spacy.load('/opt/en_core_web_sm-2.1.0')
#=================================================================
# Tokens
#=================================================================   
               
def textToTokens(text):
    "Generates a column of Spacy tokens."
    return nlp(text)
            
def partOfSpeech(row):
    '''
    For each row, assigns a POS for
    each word in the text.
    '''
    return [token.pos_ for token in row]

def posEntropy(pos):
    '''Returns the POS for each unit.'''
    # counts = pd.Series(pos).value_counts()
    # return s.entropy(counts)
    labels = pos
    vc = pd.Series(labels).value_counts(normalize=True, sort=False)
    base = e 
    return -(vc * np.log(vc)/np.log(base)).sum()

def posCounts(pos):
    "Returns the part of speech counts for each unit."
    return pd.Series(pos).value_counts()


#=================================================================
# Run Transformation Pipeline
#================================================================= 

def tokenPipeline(df, textCol = "Text"):

    transforms = [
        
        (textToTokens,"token", textCol),
        (partOfSpeech, "partOfSpeech", "token"),
        (posEntropy, "posEntropy","partOfSpeech"),
        (posCounts, "AppendCounts", "partOfSpeech")
        
   
    ]
    
    return applyTransforms(df, transforms) 
      