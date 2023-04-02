#We will use different datatsets
#and we will standardize them depending on the model we will use

import pandas as pd
import numpy as np

def standardize_without_theme(guardian, onion):
    onion.columns = ['url', 'headline', 'date', 'theme', 'article', 'length']
    guardian.columns = ['index', 'apiurl', 'article', 'pillarName', 'theme', 'type', 'date', 'headline','url', 'filtered_bodyText', 'length' ]

    #We will drop the columns that we don't need
    onion.drop(['url', 'date', 'theme', 'length'], axis=1, inplace=True)
    guardian.drop(['index', 'apiurl', 'pillarName', 'theme', 'type', 'date', 'url', 'filtered_bodyText', 'length'], axis=1, inplace=True)
    return guardian, onion
