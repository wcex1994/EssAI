# -*- coding: utf-8 -*-

"""Ref: https://github.com/interpretml/interpret-text/blob/master/notebooks/text_classification/text_classification_unified_information_explainer.ipynb"""



# Copyright (c) 2017, PyTorch contributors
# Modifications copyright (C) Microsoft Corporation
# Licensed under the BSD license
# Adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
from __future__ import print_function, division
import torch
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn as nn
from interpret_text.experimental.common.utils_bert import Language, Tokenizer, BERTSequenceClassifier
from interpret_text.experimental.common.timer import Timer
from interpret_text.experimental.unified_information import UnifiedInformationExplainer
from azureml.core.run import Run
import argparse
import time

from azureml.core.workspace import Workspace
from azureml.core import Experiment, Dataset
from azureml.core import Environment

# get the Azure ML run object
run = Run.get_context()


def load_data(df,TRAIN_SIZE, MAX_LEN):
    """Load the train/val data."""
    
    LABEL_COL = "Label"
    TEXT_COL = "Text"
    BERT_CACHE_DIR = "./temp"
    LANGUAGE = Language.ENGLISH
    TO_LOWER = True
    
    df_train, df_test = train_test_split(df, train_size = TRAIN_SIZE, random_state=2018)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    
    # encode labels
    label_encoder = LabelEncoder()
    labels_train = label_encoder.fit_transform(df_train[LABEL_COL])
    labels_test = label_encoder.transform(df_test[LABEL_COL])
    num_labels = len(np.unique(labels_train))
    
    run.log('num_labels', np.int(num_labels))
    run.log('training_size', np.int(df_train.shape[0]))
    run.log('test_size', np.int(df_test.shape[0]))
    
    
    tokenizer = Tokenizer(LANGUAGE, to_lower=TO_LOWER, cache_dir=BERT_CACHE_DIR)
    tokens_train = tokenizer.tokenize(list(df_train[TEXT_COL]))
    tokens_test = tokenizer.tokenize(list(df_test[TEXT_COL]))
    
    max_token_length = max([len(sen) for sen in tokens_train])
    
    run.log('max_token_length', np.int(max_token_length))
    
    if MAX_LEN < max_token_length and max_token_length<=512:
        MAX_LEN = max_token_length
        print("MAX_LEN is too small for the largest token size , replace with: {}".format(max_token_length))
    elif MAX_LEN < max_token_length and max_token_length>512:
        MAX_LEN = 512
        print("MAX_LEN is too small for the largest token size , replace with: 512")
        
        
    tokens_train, mask_train, _ = tokenizer.preprocess_classification_tokens(tokens_train, MAX_LEN)
    tokens_test, mask_test, _ = tokenizer.preprocess_classification_tokens(tokens_test, MAX_LEN)
    
    return tokens_train, mask_train, tokens_test, mask_test, num_labels, labels_train,labels_test


def fine_tune_model(OUTPUTDIR, NUM_EPOCHS, BATCH_SIZE, BATCH_SIZE_PRED, MODELNAME, MODELLOC, pass_training, tokens_train, mask_train, tokens_test, mask_test,num_labels, labels_train,labels_test):
    """Load a pretrained model and reset the final fully connected layer."""
    
    BERT_CACHE_DIR = "./temp"
    LANGUAGE = Language.ENGLISH
    
    ## Pre-trained 
    classifier = BERTSequenceClassifier(language=LANGUAGE, num_labels=num_labels, cache_dir=BERT_CACHE_DIR)
    if MODELLOC != '':
        classifier.model = torch.load(MODELLOC)
        print("loaded previous checkpoint!")
    
    if pass_training == False:
        ## Training
        since = time.time()

        classifier.fit(token_ids=tokens_train,
                        input_mask=mask_train,
                        labels=labels_train,    
                        num_epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE,    
                        verbose=True)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        ## Save checkpoint 
        torch.save(classifier.model, os.path.join(OUTPUTDIR, MODELNAME))
    
    ## Scoring 
    since = time.time()

    preds = classifier.predict(token_ids=tokens_test, 
                               input_mask=mask_test, 
                               batch_size=BATCH_SIZE_PRED,
                               probabilities = True)
    
    time_elapsed = time.time() - since
    print('Predict complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    rounded = [np.round(x,5) for x in preds.probabilities[:,1]*100]
    
    thisclass = np.array([0,1])

    report = classification_report(labels_test, preds.classes, target_names=thisclass, output_dict=True) 
    accuracy = accuracy_score(labels_test, preds.classes)
    
    run.log('accuracy', np.float(accuracy))
    run.log('report', report)
    run.log('percentage probability of 1', rounded)
    print(accuracy)
    
    return classifier, preds, rounded


def main():
    print("Torch version:", torch.__version__)
    
    # get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='size of batch')
    parser.add_argument('--batch_size_pred', type=int, default=12,
                        help='size of batch for prediction')
    parser.add_argument('--max_len', type=int, default=450,
                        help='max length of the tokens')    
    parser.add_argument('--output_dir', type=str, help='output directory e.g. ./checkpoints', default = './outputs')
    parser.add_argument('--model_name', type=str, help='e.g. model.pt, will overwritten')
    parser.add_argument('--data_dir', type=str, help='e.g. ./Jane_256_di.csv', default = '')
    parser.add_argument('--train_size', type=float, default=0.8,
                        help='size of training split (e.g. 0.8)')
    parser.add_argument('--interpret_index', type=int, default=1,
                        help='one index of dftest to run the interpretability')
    parser.add_argument('--model_loc', type=str, help='previous checkpoint directory e.g. ./checkpoints/jane_256_ep1.pt', default = '')
    parser.add_argument('--training_skip', type=bool, help='skip training only for testing', default = False)
    args = parser.parse_args()
    
    # make output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # import data from registered dataset 
    df = pd.read_csv(args.data_dir, header=0)
    print("df imported")
    # tokenize data 
    tokens_train, mask_train, tokens_test, mask_test,num_labels,labels_train,labels_test = load_data(df,args.train_size,args.max_len)
    
    print("tokenized input successfully")
    
    classifier, preds, rounded  = fine_tune_model(args.output_dir, args.num_epochs, args.batch_size, args.batch_size_pred, args.model_name, args.model_loc, args.training_skip, tokens_train, mask_train, tokens_test, mask_test,num_labels, labels_train,labels_test)
    print(rounded)
    
    torch.save(classifier.model, os.path.join(args.output_dir, args.model_name))
    print("model training and scoring successfully")

if __name__ == "__main__":
    main()

