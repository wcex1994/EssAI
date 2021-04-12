from __future__ import print_function, division
import torch
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#from sklearn.externals import joblib
import torch.nn as nn
from interpret_text.experimental.common.utils_bert import Language, Tokenizer, BERTSequenceClassifier
from interpret_text.experimental.common.timer import Timer
from interpret_text.experimental.unified_information import UnifiedInformationExplainer
import argparse
import time
import os 



def init():
    global interpreter_unified
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'interpreter_jane256_ep2.pt')
    interpreter_unified = torch.load(model_path, map_location=lambda storage, loc: storage)
    interpreter_unified.device = device

def explainmodel(text):
    TEXT_COL = "Text"
    LABEL_COL = "Label"
    BERT_CACHE_DIR = "./temp"
    LANGUAGE = Language.ENGLISH
    TO_LOWER = True
    
    #just get one datapoint to run
    tokenizer = Tokenizer(LANGUAGE, to_lower=TO_LOWER, cache_dir=BERT_CACHE_DIR)
    thistext = text[TEXT_COL]
    true_label = text[LABEL_COL]
    tokens_test_explain = tokenizer.tokenize(list(text[TEXT_COL]))
    stopwords = ['.','!','?']
    salute = ['mr','ms','miss','mrs']
    if text.shape[0] > 1:
        sent = []
        importance = []
        i=0
        for t in thistext:
            explanation_unified = interpreter_unified.explain_local(t, true_label[i])

            sorted_local_importance_names = explanation_unified.get_ranked_local_names()
            sorted_local_importance_values = explanation_unified.get_ranked_local_values()

            word_dict = dict()
            for name, value in zip(sorted_local_importance_names,sorted_local_importance_values):
                if name in word_dict and word_dict.get(name) < value:
                    word_dict[name] = value
                elif name not in word_dict:
                    word_dict[name] = value

            thisvalue = 0
            tokens = []
            prevword = ''
            
            for word in tokens_test_explain[0]:
                if word in stopwords and prevword not in salute:
                    if word in word_dict:
                        thisvalue += word_dict.get(word)
                    tokens.append(word)
                    thissent = ' '.join([x for x in tokens])
                    thissent = thissent.replace(' ##', '')
                    sent.append(thissent)
                    importance.append(thisvalue)

                    tokens = []
                    thisvalue = 0
                else:   
                    tokens.append(word)
                    if word in word_dict:
                        thisvalue += word_dict.get(word)
                prevword = word
            i+=1
        mostimportantsent = sent[importance.index(max(importance))]
        print(sent)
        

    else:
        explanation_unified = interpreter_unified.explain_local(thistext[0], true_label[0])

        sorted_local_importance_names = explanation_unified.get_ranked_local_names()
        sorted_local_importance_values = explanation_unified.get_ranked_local_values()
        print("finished paragraph interpretation")
    
        word_dict = dict()
        for name, value in zip(sorted_local_importance_names,sorted_local_importance_values):
            if name in word_dict and word_dict.get(name) < value:
                word_dict[name] = value
            elif name not in word_dict:
                word_dict[name] = value

        sent = []
        importance = []
        thisvalue = 0
        tokens = []
        prevword = ''
        for word in tokens_test_explain[0]:
            if word in stopwords and prevword not in salute:
                if word in word_dict:
                    thisvalue += word_dict.get(word)
                tokens.append(word)
                thissent = ' '.join([x for x in tokens])
                thissent = thissent.replace(' ##', '')
                sent.append(thissent)
                importance.append(thisvalue)

                tokens = []
                thisvalue = 0
            else:   
                tokens.append(word)
                if word in word_dict:
                    thisvalue += word_dict.get(word)
            prevword = word
        
        mostimportantsent = sent[importance.index(max(importance))]
        print(sent)
        
    print("finished testing one example")
    return mostimportantsent


def run(raw_data):
    Text = np.array(json.loads(raw_data)['Text'])
    Label = np.full((1, len(Text)), 1)[0]

    d = {'Text': Text, 'Label': Label}
    input_data = pd.DataFrame(data=d)

    since = time.time()

    mostimportantsent = explainmodel(input_data)
    print(mostimportantsent)
    time_elapsed = time.time() - since
    print('Predict complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("finished calculating most important sentence")

    result = json.dumps({"most_imp_sent": mostimportantsent})
    return result
