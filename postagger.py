import nltk
from nltk.corpus import indian
from nltk.tag import tnt
import string
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import torch
from transformers import AutoTokenizer
import concurrent.futures

# model_name = "ai4bharat/IndicBART"
# tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False, keep_accents=True)

def train_pos():
    tagged_set = 'hindi.pos'
    word_set = indian.sents(tagged_set)
    count = 0
    for sen in word_set:
        count += 1
        sen = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in sen]).strip()

    train_perc = 0.9
    train_rows = int(train_perc*count)
    test_rows = train_rows+1

    data = indian.tagged_sents(tagged_set)
    train_data = data[:train_rows]
    test_data = data[test_rows:]

    pos_tagger = tnt.TnT()
    pos_tagger.train(train_data)
    pos_tagger.accuracy(test_data)

    return pos_tagger

def update_tag(dictionary_item):
    key, value = dictionary_item
    #key_g.append(key)
    #value_g.append(value)
    set_of_pos = {'RB','NN','PREP','VNN', 'VAUX'}
    if value not in set_of_pos:
        value = "UNK"     
    return key, value

def pos_tag_fn(sentence,pos_tagger,tokenizer):

    # s_lang = 'hi'
    # sentence += ' </s>' + f' <2{s_lang}>'
    words = sentence.split(" ")
    tagged_words = pos_tagger.tag(words)
    tags = []
    # dict_word_tag = dict(tagged_words)

    tokens = tokenizer(words)
    set_of_pos = {'RB','NN','PREP','VNN', 'VAUX'}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for key, new_value in executor.map(update_tag, tagged_words):
            tags.append(new_value)

    # tags = list(dict_word_tag.values())
    # print(words)
    # print(len(tags),tags)
    # print(len(tokens.input_ids),tokens.input_ids)

    pos_tag = []
    lnth_of_tokens = len(tokens.input_ids)
    for i in range(lnth_of_tokens):
        token_length = len(tokens.input_ids[i])
        if token_length > 3:
            k = 0
            num_divided_words = token_length - 2
            pos_tag += num_divided_words * [tags[i]]
        else:
            pos_tag.append(tags[i])

    pos_tag.insert(0,'<pos>')
    pos_tag.insert(lnth_of_tokens+2,'</pos>')

    return pos_tag
    
    
    
# sentence = "जब आप कोई भाषा को सीखते हैं"
# sentence = "इस ‘मन की बात’ के लिये आने वाले सुझाव, phone call की संख्या, सामान्य रूप से कई गुणा ज्यादा है |"
# temp = pos_tag_fn(sentence,tokenizer)
# print(temp)
    