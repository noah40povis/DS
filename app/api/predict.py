import logging
import random
#import astropy.constants.tests.test_pickle

from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator

log = logging.getLogger(__name__)
router = APIRouter()


class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    title: str = Field(..., example="my hangnail is getting so painful")
    selftext: str = Field(..., example=(f"Comedian Nigel Ng to collaborate with BBC"
                                        f"host Hersha Patel on cooking video after Asian"
                                        f"netizens in uproar over controversial egg fried"
                                        f"rice tutorial."))
    n_results: int = Field(..., example=5)

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

    @ validator('n_results')
    def n_results_positive(cls, value):
        """Validate that n_results is a positive number."""
        assert value > 0, f'n_results == {value}, must be > 0'
        return value


@ router.post('/predict')
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
    n_results = item.n_results

    recomendations = random.sample(predictions, n_results)
    return {'subreddits': recommendations }
