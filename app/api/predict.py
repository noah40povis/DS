import logging
import random
import sklearn
from joblib import load
from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator

log = logging.getLogger(__name__)
router = APIRouter()


class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    title: str = Field(..., example="my hangnail is getting so painful")
    selftext: str = Field(..., example=(f"Comedian Joe Rogan to collaborate with BBC"
                                        f"host Hersha Patel on cooking video after Asian"
                                        f"netizens in uproar over controversial egg fried"
                                        f"rice tutorial."))

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])



@ router.post('/test_predict')
async def dummy_predict(item: Item):
    """dummy model to return some data for testing the API

    Parameter input :  str title, str selftext, int n_results

    Returns dict recommendations :

    """

    predictions = ['HomeDepot', 'DunderMifflin', 'hometheater', 'EnterTheGungeon',
                   'cinematography', 'Tinder', 'LearnJapanese',
                   'futarp', 'OnePieceTC', 'Firefighting', 'fleshlight', 'lotr',
                   'knifeclub', 'sociopath', 'bleach', 'SCCM', 'GhostRecon',
                   'Ayahuasca', 'codes', 'preppers', 'grammar', 'NewSkaters',
                   'Truckers', 'southpark', 'Dreams', 'JUSTNOMIL', 'bigdickproblems',
                   'EternalCardGame', 'evangelion', 'mercedes_benz', 'Cuckold',
                   'writing', 'afinil', 'synology', 'thinkpad', 'MDMA', 'sailing',
                   'cfs', 'siacoin', 'ASUS', 'OccupationalTherapy', 'biology',
                   'thelastofus', 'lonely', 'swrpg', 'acting', 'transformers',
                   'vergecurrency', 'Beekeeping']

    recs = {}  # store in dict
    
    n_results = 5             # fix to 5 results 

    recommendations = random.sample(predictions, n_results)
    return {'subreddits': recommendations }

@ router.post('/nsfw_predict')
async def nsfw_predict(item: Item):
    """ WARNING: May return NSFW content! \n
    load, query the prediction model
    return : [5 best results]
    """

    model = load('subreddit_mvp.joblib')   
    tfidf = load('reddit_mvp_tfidf.joblib')
    
    df = pd.read_csv('25325_subreddits.csv')
    subreddits = df['subreddit']

    predictions = []
    query = tfidf.transform([item.title+item.selftext])
    pred = model.kneighbors(query.todense())
    for i in pred[1][0]:
        predictions.append(subreddits[i])
        output = list(predictions)
    return {'recommendations' : output }
                                        
@ router.post('/predict')
async def swf_predict(item: Item):
    """ ingest title, selftext content 
    spit out 5 best subreddits     uri = f"https://raw.githubusercontent.com/worldwidekatie/BW_4/master/cleaned_subs.csv"

    """
    df = pd.read_csv("https://raw.githubusercontent.com/worldwidekatie/BW_4/master/cleaned_subs.csv")
    subreddits1 = df['subreddit']
    swf_model = load('nn_cleaned.joblib')
    tfidf = load('tfidf_cleaned.joblib')

    predictions = []
    query = tfidf.transform([item.title+item.selftext])
    pred = swf_model.kneighbors(query.todense())
    print(len(subreddits1))
    
    for i in pred[1][0]:
        if subreddits1[i] not in predictions:
            predictions.append(subreddits1[i])
    
    return {'recommendations': predictions[:5] }
    