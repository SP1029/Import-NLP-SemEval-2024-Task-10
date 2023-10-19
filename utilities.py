import os
import json
import pandas as pd

import string
import nltk
import re

import json
import pickle
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
# % matplotlib inline

# Setting the default torch device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)


def json_to_csv(filepath, is_erc = False):
    """Converts json data from competition to csv for usage in code repo

    Args:
        filepath: path to json file
        is_erc (bool, optional): set to `True` if data to be exported is for 
        ERC, else set to `False`. Defaults to `False`.
    Returns:
        csv_filepath: the path to csv file
    """
    path = os.path.split(filepath)
    filename = '.'.join(path[-1].split('.')[:-1])
    
    with open(filepath, 'r') as file:
        dataset = json.load(file)
    
    # Reading the data
    episodes_list = []
    speakers_list = []
    utterances_list = []
    triggers_list = []
    emotions_list = []

    for i in range(len(dataset)):
            episodes_list.append(dataset[i]['episode'])
            speakers_list.append(dataset[i]['speakers'])
            utterances_list.append(dataset[i]['utterances'])
            if is_erc==False:
                    triggers_list.append(dataset[i]['triggers'])
            emotions_list.append(dataset[i]['emotions'])  
        
    N = len(dataset)
    
    # Change the headings
    if is_erc==False:
        headings = ["Dialogue_Id", "Speaker", "Emotion_name", "Utterance", "Annotate(0/1)"]
    else:
        headings = ["Dialogue_Id", "Speaker", "Emotion_name", "Utterance"]
    
    data = [[],[],[],[],[]]
    
    if is_erc:
        data = [[],[],[],[]]
    
    prev_first_sen = None
    d_id = -1
    for i in range(N):
        if prev_first_sen!=utterances_list[i][0]:
            d_id+=1
            prev_first_sen = utterances_list[i][0]
            
        for j in range(len(utterances_list[i])):
            data[0].append(d_id)
            data[1].append(speakers_list[i][j])
            data[2].append(emotions_list[i][j])
            data[3].append(utterances_list[i][j])
            if is_erc==False:
                data[4].append(triggers_list[i][j])
            
        data[0].append("")
        data[1].append("")
        data[2].append("")
        data[3].append("")
        if is_erc==False:
            data[4].append("")
    
    df = pd.DataFrame(data)
    df = df.transpose()
    df.columns = headings
    csv_filepath = os.path.join(path[0], filename + ".csv")
    if (is_erc):
        filename = filename.replace('efr', 'erc')
        csv_filepath = os.path.join(path[0], filename + ".csv")
    df.to_csv(csv_filepath, index=False)
    return csv_filepath

def bert_embeddings(train_filepath, val_filepath = None):
    def remove_puntuations(txt):
        punct = set(string.punctuation)
        txt = " ".join(txt.split("."))
        txt = " ".join(txt.split("!"))
        txt = " ".join(txt.split("?"))
        txt = " ".join(txt.split(":"))
        txt = " ".join(txt.split(";"))
        
        txt = "".join(ch for ch in txt if ch not in punct)
        return txt

    def number_to_words(txt):
        for k in numbers.keys():
            txt = txt.replace(k,numbers[k]+" ")
        return txt

    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'_',' ',text)
        text = number_to_words(text)
        text = remove_puntuations(text)
        text = ''.join([i if ord(i) < 128 else '' for i in text])
        text = ' '.join(text.split())
        return text
    
    def get_sen_embed(utt):
        # Run the text through BERT, and collect all of the hidden states produced
        # from all 12 layers. 
        with torch.no_grad():

            # Define a new example sentence with multiple meanings of the word "bank"
            text = utt

            # Add the special tokens.
            marked_text = "[CLS] " + text + " [SEP]"

            # Split the sentence into tokens.
            tokenized_text = tokenizer.tokenize(marked_text)

            # Map the token strings to their vocabulary indeces.
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

            # Mark each of the 22 tokens as belonging to sentence "1".
            segments_ids = [1] * len(tokenized_text)

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            outputs = model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]

        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(hidden_states, dim=0)

        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1,0,2)

        # Stores the token vectors, with shape [22 x 3,072]
        token_vecs_cat = []

        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
        # Stores the token vectors, with shape [22 x 768]
        token_vecs_sum = []

        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
        for token in token_embeddings:

            # `token` is a [12 x 768] tensor

            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)
            
            # Use `sum_vec` to represent `token`.
            token_vecs_sum.append(sum_vec)

        # `hidden_states` has shape [13 x 1 x 22 x 768]

        # `token_vecs` is a tensor with shape [22 x 768]
        token_vecs = hidden_states[-2][0]

        # Calculate the average of all 22 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        
        return sentence_embedding
    
    # Add BERT Embeddings from a file to ans_dict
    def add_bert_embeddings(filepath, ans_dict = dict()):

        # Read File
        with open(filepath, 'r') as file:
                dataset = json.load(file)

        N = len(dataset)

        # Reading the data

        episodes_list = []
        speakers_list = []
        utterances_list = []
        triggers_list = []
        emotions_list = []

        for i in tqdm(range(len(dataset)), ncols=100, desc='Extracting data'):
                episodes_list.append(dataset[i]['episode'])
                speakers_list.append(dataset[i]['speakers'])
                utterances_list.append(dataset[i]['utterances'])
                triggers_list.append(dataset[i]['triggers'])
                emotions_list.append(dataset[i]['emotions'])

        for i in tqdm(range(N), ncols=100, desc='Generating embeddings'):
            for j in range(len(utterances_list[i])):
                pp_utt = preprocess_text(utterances_list[i][j])
                utt_emb = get_sen_embed(pp_utt)
                ans_dict[pp_utt] = utt_emb
            # if i%10==0:
            #   print(i)
    
    numbers = {
        "0":"zero",
        "1":"one",
        "2":"two",
        "3":"three",
        "4":"four",
        "5":"five",
        "6":"six",
        "7":"seven",
        "8":"eight",
        "9":"nine"
    }
    
    # # Load pre-trained model tokenizer (vocabulary)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # # Load pre-trained model (weights)
    # model = BertModel.from_pretrained(
    #     'bert-base-uncased',
    #     output_hidden_states = True, # Whether the model returns all hidden-states.
    #     )
    
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('l3cube-pune/hing-mbert')
    
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(
        'l3cube-pune/hing-mbert',
        output_hidden_states = True, # Whether the model returns all hidden-states.
        )
    
    ans_dict = {}
    add_bert_embeddings(train_filepath, ans_dict)
    if val_filepath:
        add_bert_embeddings(val_filepath, ans_dict)
    
    # Save the file
    file_dir = os.path.split(train_filepath)[0]
    with open(os.path.join(file_dir, 'sent2emb.pickle'), 'wb') as handle:
        pickle.dump(ans_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



numbers = {
    "0":"zero",
    "1":"one",
    "2":"two",
    "3":"three",
    "4":"four",
    "5":"five",
    "6":"six",
    "7":"seven",
    "8":"eight",
    "9":"nine"
}

def remove_puntuations(txt):
    punct = set(string.punctuation)
    txt = " ".join(txt.split("."))
    txt = " ".join(txt.split("!"))
    txt = " ".join(txt.split("?"))
    txt = " ".join(txt.split(":"))
    txt = " ".join(txt.split(";"))
    
    txt = "".join(ch for ch in txt if ch not in punct)
    return txt

def number_to_words(txt):
    for k in numbers.keys():
        txt = txt.replace(k,numbers[k]+" ")
    return txt

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'_',' ',text)
    text = number_to_words(text)
    text = remove_puntuations(text)
    text = ''.join([i if ord(i) < 128 else '' for i in text])
    text = ' '.join(text.split())
    return text
   

if __name__ == '__main__':
    # json_to_csv('./Data/MaSaC/MaSaC_train_efr.json')
    # json_to_csv('./Data/MaSaC/MaSaC_train_erc.json', is_erc=True)
    # json_to_csv('./Data/MaSaC/MaSaC_val_efr.json')
    # json_to_csv('./Data/MaSaC/MaSaC_val_erc.json', is_erc=True)
    bert_embeddings(
        train_filepath = './Data/MaSaC/MaSaC_train_efr.json',
        val_filepath = './Data/MaSaC/MaSaC_val_efr.json'
    )