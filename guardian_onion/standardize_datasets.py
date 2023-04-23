# We will use different datatsets
# and we will standardize them depending on the model we will use

import pandas as pd
import re
from datasets import load_dataset

import spacy
from spacy.lang.en.stop_words import STOP_WORDS


def standardize_without_theme():
    # Open data files
    guardian = pd.read_csv('the_guardian/articles_processed.csv')
    onion = pd.read_csv('the_onion/articles_processed.csv')

    onion.columns = ['url', 'headline', 'date', 'theme', 'article', 'length']
    guardian.columns = [
        'index', 'apiurl', 'article', 'pillarName', 'theme', 'type',
        'date', 'headline', 'url', 'filtered_bodyText', 'length']

    # We will drop the columns that we don't need
    onion.drop(['url', 'date', 'theme', 'length'], axis=1, inplace=True)
    guardian.drop(['index', 'apiurl', 'pillarName', 'theme', 'type', 'date', 'url', 'filtered_bodyText', 'length'], axis=1, inplace=True)
    return guardian, onion


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


def remove_punctuation(text):
    # Using regex to remove punctuation
    text = re.sub(r"(?!['])\W", ' ', text)
    return text


def merge(df1, df2):
    df1['label'] = 1
    df2['label'] = 0
    df = pd.concat([df1, df2])
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def get_merge_dataset():
    df1, df2 = standardize_without_theme()
    df = merge(df1, df2)
    return df


def get_clean_dataset():
    df = pd.read_csv('data/satire_dataset.csv')
    return df


def remove_guillemets(text):
    text = re.sub("'", '', text)
    return text


def str_to_list(text):
    liste = text.split(', ')
    liste[0] = liste[0][1:]  # Remove the first "[
    liste[-1] = liste[-1][:-1]  # Remove the last "]"
    liste = [remove_guillemets(word) for word in liste]
    return liste


def get_lemmatized_dataset():
    df = pd.read_csv('data/preprocessed_satire_dataset.csv')
    df['article'] = df['article'].apply(str_to_list)
    return df


def get_standardized_liar_dataset():
    dataset = load_dataset('liar')
    dt_test = pd.DataFrame(dataset['validation'])
    dt_test["label"] = dt_test["label"].apply(lambda x: 1 if (x == 2 or x == 3 or x == 1) else 0)
    return dt_test

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