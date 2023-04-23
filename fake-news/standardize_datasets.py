# We will use different datatsets
# and we will standardize them depending on the model we will use

import pandas as pd
import re
from datasets import load_dataset

import spacy
from spacy.lang.en.stop_words import STOP_WORDS


def clean_unicode(text):
    # Using regex to clean the text for the following cases
    # \n : remove
    # &quot; : remove
    # &#+any number; : remove

    text = re.sub(r'\n', '', text)
    text = re.sub(r'&quot;', ' ', text)
    # text = re.sub(r'&#39;', "'", text)  # Replace &#39 with '
    text = re.sub(r'&#\w+;', ' ', text)
    return text

# Preprocess


nlp = None


def remove_stopwords_and_lemmatize_spacy_multiprocessed(batch, n_process=4):
    global nlp
    if nlp is None:
        nlp = spacy.load('en_core_web_sm')
    docs = nlp.pipe(batch, batch_size=100, n_process=n_process, disable=['parser', 'ner'])
    lemmatized_sentence = [[w.lemma_ for w in doc if w.text not in STOP_WORDS] for doc in docs]
    return lemmatized_sentence


def remove_blanck_space(list_of_sentences):
    return [[word for word in sentence if word.find(' ') == -1] for sentence in list_of_sentences]


def to_lower(list_of_words):
    return [word.lower() for word in list_of_words]


def preprocess(list_of_sentences, n_process=4):
    return [to_lower(sent) for sent in remove_blanck_space(remove_stopwords_and_lemmatize_spacy_multiprocessed(list_of_sentences, n_process=n_process))]