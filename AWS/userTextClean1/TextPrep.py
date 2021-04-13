"""
This modeule is for cleaning user text and chunking it as specified. Preparing it for the BERT module. 
"""
import nltk
import pandas as pd
import re
import json 
import numpy


# nltk.download('punkt')
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


from nltk.tokenize import word_tokenize
nltk.data.path.append("/tmp")
nltk.download("punkt", download_dir = "/tmp")
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# # Clean user text and break into chunks as required by the model. 
# def convert_text_to_dict(in_txt, debug=0):
#     out_df = Text2Dict(in_txt, debug)
#     return(out_df)


# Function to convert all txt files in given directory.
def Text2Dict_sentence(input_txt):
    print('i am in textdict')
    print(type(input_txt))
    # data = re.sub('.','. ',input_txt)
    data = input_txt
    lines_0 = data.split("\n")
    lines_0 = [data]
    double_quote_pattern = '“|”|"'
 
    line_idx = 1
    paragraph_idx = 1
    
    sent_list = []
    double_list = []
    wc_list = []
    data_df = pd.DataFrame() 
    li_list = []
    pi_list = []
    
    # for line in lines_0:
    double_quote_status = 0
    for i in range(1):
        for sent in tokenizer.tokenize(data):
            print(sent)
            d = {}
            double_quoted = 0
            double_quote_count = len(re.findall(double_quote_pattern, sent))
            if double_quote_count % 2:
                # Toggle double_quote_status
                double_quote_status = 1 - double_quote_status
            if double_quote_count > 0 or double_quote_status:
                double_quoted = 1
                # Remove quotes
                sent = re.sub(double_quote_pattern, '', sent)
            wc = len(sent.split())
   
            sent_list.append(sent)
            double_list.append(double_quoted)
            wc_list.append(wc)
            li_list.append(line_idx)
            pi_list.append(paragraph_idx)
            line_idx += 1
        paragraph_idx += 1
    
    data_df['LineNum'] = li_list
    data_df['ParagraphNum'] = pi_list
    data_df['Text'] = sent_list
    data_df['DoubleQuoted'] = double_list
    data_df['WordCount'] = wc_list
    return(data_df)

    
def Text2Dict(input_txt):
    # data = re.sub('.','. ',input_txt)
    data = input_txt
    lines_0 = data.split("\n")
    double_quote_pattern = '“|”|"'
 
    line_idx = 1
    paragraph_idx = 1
    
    sent_tot = 0
    double_tot = 0
    wc_tot = 0
    data_df = pd.DataFrame() 
    li_tot = 0
    pi_tot = 0
    
    for line in lines_0:
        double_quote_status = 0
        for sent in tokenizer.tokenize(line):
            print(sent)
            d = {}
            double_quoted = 0
            double_quote_count = len(re.findall(double_quote_pattern, sent))
            if double_quote_count % 2:
                # Toggle double_quote_status
                double_quote_status = 1 - double_quote_status
            if double_quote_count > 0 or double_quote_status:
                double_quoted = 1
                # Remove quotes
                sent = re.sub(double_quote_pattern, '', sent)
            wc = len(sent.split())
   
            # sent_tot = sent_tot + sent
            double_tot = double_tot + double_quoted
            wc_tot = wc_tot + wc
            li_tot = li_tot + line_idx
            pi_tot = pi_tot + paragraph_idx
            line_idx += 1
        paragraph_idx += 1
    
    data_df['LineNum'] = [li_tot]
    data_df['ParagraphNum'] = [pi_tot]
    data_df['Text'] = [input_txt]
    data_df['DoubleQuoted'] = [double_tot]
    data_df['WordCount'] = [wc_tot]
    return(data_df)
    
    

# def chunk_by_wordLength(input_txt, target_word_len):
    
#     data_df = Text2Dict(input_txt)
#     # Input
    
#     li_list_in = data_df['LineNum']
#     pi_list_in = data_df['ParagraphNum']
#     sent_list_in = data_df['Text'] 
#     double_list_in = data_df['DoubleQuoted']
#     wc_list_in = data_df['WordCount']
    
#     df_len = len(li_list_in)
 
#     cID_list = [] 
#     lc_list = []
#     ps_list = []
#     pe_list = [] 
#     pcnt_list = [] 
#     dq_list = [] 
#     wc_list = [] 
# #     wc_list2 = []
#     text_list = [] 
    
#     data_df_out = pd.DataFrame()  


#     LineCount = 0
#     pa_start = 0
#     pa_curr = 0

#     DoubleQuoted = 0
#     WordCount = 0
#     out_Text = ""
#     chunkID = 1
    
  
#     for i in range(0,df_len):

#         # Get line attributes.
#         pa = pi_list_in[i]
#         dq = double_list_in[i]
#         wc = wc_list_in[i]
#         text = sent_list_in[i]
        
#         # Ignore lines longer than target word length.
#         if wc > target_word_len: continue
        
#         # Check if line should be included.        
#         if WordCount + wc > target_word_len:
#             # Save previous line
#             if out_Text != "":
#                 cID_list.append(chunkID) 
#                 lc_list.append(LineCount)
#                 pa_start = pa_start if pa_start != 0 else 1
#                 ps_list.append(pa_start)
#                 pe_list.append(pa_curr)
#                 pcnt_list.append(pa_curr - pa_start + 1)
#                 dq_list.append(dq)
#                 wc_list.append(WordCount)
# #                 wc_list2.append(len(out_Text.split()))
#                 text_list.append(out_Text) 
#                 chunkID += 1
#                 i+=1
            
#             # Start new line
#             out_Text = text
#             LineCount = 1
#             WordCount = wc
#             pa_start = pa
#             pa_curr = pa
#             DoubleQuoted = dq
#         else:
#             # Add to current line
#             out_Text = "%s %s" %(out_Text, text)
#             LineCount += 1
#             WordCount += wc
#             pa_curr = pa

#             if dq: DoubleQuoted = 1
        
         
#     cID_list.append(chunkID) 
#     lc_list.append(LineCount)
#     ps_list.append(pa_start)
#     pe_list.append(pa_curr)
#     pcnt_list.append(pa_curr - pa_start + 1)
#     dq_list.append(dq)
#     wc_list.append(WordCount)
# #     wc_list2.append(len(out_Text.split()))
#     text_list.append(out_Text) 
    
#     # data_df_out['chunkID'] = cID_list
#     # data_df_out['LineCount'] = lc_list
#     # data_df_out['ParagraphStart'] = ps_list
#     # data_df_out['ParagraphEnd'] = pe_list
#     # data_df_out['ParagraphCount'] = pcnt_list
#     # data_df_out['DoubleQuoted'] = dq_list
#     # data_df_out['WordCount'] = wc_list
# #     data_df_out['WordCount2'] = wc_list2
#     data_df_out['Text'] = text_list
#     result = data_df_out.reset_index().to_json(orient='records')
    
# #     result = data_df_out.to_json(orient="split")
# #     parsed = json.loads(result)
# #     parsed1 = json.dumps(parsed, indent=4)  
# #     print(type(parsed1))
# #     print(parsed1)
#     return text_list
    
