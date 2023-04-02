#We will use different datatsets
#and we will standardize them depending on the model we will use

import pandas as pd
import numpy as np

def standardize_without_theme():
    #Open data files
    guardian = pd.read_csv('the_guardian/articles_processed.csv')
    onion = pd.read_csv('the_onion/articles_processed.csv')

    onion.columns = ['url', 'headline', 'date', 'theme', 'article', 'length']
    guardian.columns = ['index', 'apiurl', 'article', 'pillarName', 'theme', 'type', 'date', 'headline','url', 'filtered_bodyText', 'length' ]

    #We will drop the columns that we don't need
    onion.drop(['url', 'date', 'theme', 'length'], axis=1, inplace=True)
    guardian.drop(['index', 'apiurl', 'pillarName', 'theme', 'type', 'date', 'url', 'filtered_bodyText', 'length'], axis=1, inplace=True)
    return guardian, onion

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
    
