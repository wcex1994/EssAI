#=================================================================
# Libraries
#=================================================================   

from pipeline.utils import * 
from scipy import stats as s
import pandas as pd

# This language model will be used for tokenizing and pos tagging
import spacy
nlp = spacy.load("en_core_web_sm")
       
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
    counts = pd.Series(pos).value_counts()
    return s.entropy(counts)

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
      