import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
from transformers import  DistilBertTokenizer



def clean_and_lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    # Tokenize text
    words = nltk.word_tokenize(text)
    # Lemmatize words
    lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in words]  # Lemmatize verbs
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text

def find_matched_params(text, params_df):
    matched_params = []
    for param in params_df['Processed_Params']:
        if param in text:
            matched_params.append(param)
    return matched_params

def find_matched_params_inference(text, params_list):
    matched_params = []
    for param in params_list:
        if param in text:
            matched_params.append(param)
    return matched_params
    tokenizer = DistilBertTokenizer.from_pretrained('bert-base-uncased')

# [CLS] ... transcript tokens ... [SEP] ..actions..[SEP]
def prepare_bert_input(text, matched_params, max_seq_length=128):
    tokenizer = DistilBertTokenizer.from_pretrained('bert-base-uncased')
    text_tokens = tokenizer.tokenize(text)
    param_tokens = tokenizer.tokenize(' '.join(matched_params))

    max_text_length = (max_seq_length - 3) // 2  # 3 (2 for [CLS] and [SEP], 1 for [SEP] between text and params)
    text_tokens = text_tokens[:max_text_length]
    param_tokens = param_tokens[:max_seq_length - len(text_tokens) - 2]  # Account for [CLS], [SEP], and text tokens

    # [CLS] ... transcript tokens ... [SEP] ..actions..[SEP]
    input_tokens = ['[CLS]'] + text_tokens + ['[SEP]'] + param_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

    attention_mask = [1] * len(input_ids)  # (1 for real tokens, 0 for padding tokens)
    # Pad sequences to max_seq_length
    padding_length = max_seq_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    attention_mask += [0] * padding_length
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

params_list = actions_list = [
    "post up",
    "double team",
    "finger roll",
    "pump fake",
    "floater",
    "slam dunk",
    "pick and roll",
    "coast to coast",
    "outlet pass",
    "fadeaway",
    "tip in",
    "alley oop",
    "rainbow shot",
    "teardrop",
    "nothing but net",
    "splash",
    "between the legs",
    "tomahawk",
    "bank shot",
    "poster",
    "take it to the rack",
    "swish",
    "jab step",
    "give and go",
    "flop",
    "baseball pass",
    "reverse dunk",
    "step back",
    "fake",
    "backdoor",
    "lob",
    "jam",
    "behind the back",
    "dime",
    "side step",
    "shake and bake",
    "no look pass",
    "euro step"
]

