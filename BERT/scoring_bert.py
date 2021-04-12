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
    global classifier
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    classifier = BERTSequenceClassifier(language=Language.ENGLISH, num_labels=2, cache_dir="./temp")
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'Mark_128_ep2.pt')
    # jane_256_ep2  450
    # Charles_256_ep4 468
    # Fitzgerald_128_ep2 250
    # Mark_128_ep2 430
    #classifier.model = joblib.load(model_path)
    classifier.model = torch.load(model_path, map_location=lambda storage, loc: storage)
    ##, map_location=lambda storage, loc: storage
    #model.eval()

def transformdata(df, MAX_LEN):
    """Load the train/val data."""
    TEXT_COL = "Text"
    BERT_CACHE_DIR = "./temp"
    LANGUAGE = Language.ENGLISH
    TO_LOWER = True

    tokenizer = Tokenizer(LANGUAGE, to_lower=TO_LOWER, cache_dir=BERT_CACHE_DIR)
    tokens_text = tokenizer.tokenize(list(df[TEXT_COL]))
    
    max_token_length = max([len(sen) for sen in tokens_text])
    print("max token length: {}".format(max_token_length))
    tokens_text, mask_text,_ = tokenizer.preprocess_classification_tokens(tokens_text, MAX_LEN)
    return tokens_text, mask_text



def run(raw_data):
    Text = np.array(json.loads(raw_data)['Text'])
    Label = np.full((1, len(Text)), 1)[0]

    #MAX_LEN = json.loads(raw_data)['MAX_LEN']
    MAX_LEN = 430
    d = {'Text': Text, 'Label': Label}
    input_data = pd.DataFrame(data=d)

    tokens_text, mask_text = transformdata(input_data, MAX_LEN)
    print("finished transforming user input")

    since = time.time()

    preds = classifier.predict(token_ids=tokens_text, 
                               input_mask=mask_text, 
                               batch_size=1,
                               probabilities = True)

    time_elapsed = time.time() - since
    print('Predict complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print("finished prediction on user input")

    rounded = [np.round(x,5) for x in preds.probabilities[:,1]*100]

    lineitems=[]
    i = 0
    for q in rounded:
        myjson3 = {
                    'Text': input_data.Text[i],
                    'probpercentage': str(q)
                }
        lineitems.append(myjson3)
        i+=1
    #result = json.dumps(lineitems)

    return lineitems
