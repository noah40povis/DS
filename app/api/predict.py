import logging
import random
import sklearn
from joblib import load
from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator

log = logging.getLogger(__name__)
router = APIRouter()

# load optional nsfw models and labels

ns_model = load('subreddit_mvp.joblib')
ns_tfidf = load('reddit_mvp_tfidf.joblib')
ns_df = pd.read_csv('25325_subreddits.csv')
ns_labels = ns_df['subreddit']

# load default models and labels
df = pd.read_csv(
    "https://raw.githubusercontent.com/worldwidekatie/BW_4/master/cleaned_subs.csv")
sfw_labels = df['subreddit']
sfw_model = load('nn_cleaned.joblib')
sfw_tfidf = load('tfidf_cleaned.joblib')


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


def pred(postdataItem, model=sfw_model):
    """
    generic function to get the raw prediction list
    parameters:
        model model  :    pass in a model to use
        Item postdataItem   : pass in an Item instance containing the post body

    returns :
        list predictions  :  ids of recommendations
    """
    predictions = []
    tfidf = sfw_tfidf
    labels = sfw_labels
    item = postdataItem

    if model == ns_model:
        tfidf = ns_tfidf
        labels = ns_labels

    query = tfidf.transform([item.title + item.selftext])
    query_result = model.kneighbors(query.todense())
    for i in query_result[1][0]:
        predictions.append(labels[i])
        output = list(predictions)
    return predictions  # give us the list of predictions


@ router.post('/predict')
async def predict(item: Item):  # SFW
    return {'recommendations': list(set(pred(item)))[:5]}


@router.post('/nsfw_predict')
async def nsfw_predict(item: Item):
    """ WARNING: May return NSFW content! \n
    load, query the prediction model
    return : [5 best results]
    """
    return {'recommendations': pred(item, ns_model)}


@router.post('/test_predict')
async def dummy_predict(item: Item):
    """
    dummy_predict  to return some data for testing the API

    Parameter input :  str title, str selftext, int n_results

    Returns:
    dict recommendations : { 'recommendations:['reditt1',
                                                'reddit2',
                                                ...,
                                                'reddit5']}

    """
    predictions = ['HomeDepot', 'DunderMifflin', 'hometheater', 'EnterTheGungeon',
                   'cinematography', 'Tinder', 'LearnJapanese',
                   'futarp', 'OnePieceTC', 'Firefighting', 'fleshlight', 'lotr',
                   'knifeclub', 'sociopath', 'bleach', 'SCCM', 'GhostRecon',
                   'Ayahuasca', 'codes', 'preppers', 'grammar', 'NewSkaters',
                   'Truckers', 'southpark', 'Dreams', 'JUSTNOMIL',
                   'EternalCardGame', 'evangelion', 'mercedes_benz', 'Cuckold',
                   'writing', 'afinil', 'synology', 'thinkpad', 'MDMA', 'sailing',
                   'cfs', 'siacoin', 'ASUS', 'OccupationalTherapy', 'biology',
                   'thelastofus', 'lonely', 'swrpg', 'acting', 'transformers',
                   'vergecurrency', 'Beekeeping']

    recs = {}  # store in dict

    n_results = 5             # fix to 5 results

    recommendations = random.sample(predictions, n_results)
    return {'subreddits': recommendations}
