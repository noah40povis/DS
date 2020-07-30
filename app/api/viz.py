from fastapi import APIRouter, HTTPException
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .predict import Item
from joblib import load

router = APIRouter()
sfw_model = load('nn_cleaned.joblib')
sfw_tfidf = load('tfidf_cleaned.joblib')

tfidf = sfw_tfidf
model = sfw_model
df = pd.read_csv('cleaned_subs.csv', usecols=[1])
subreddits = df['subreddit']

@router.post('/viz')
async def viz(postbody: Item):
    query = tfidf.transform([postbody.title+postbody.selftext])   #use sfw

    query_results= model.kneighbors(query.todense())
    preds = list(zip(query_results[1][0], query_results[0][0]))
    predictions = []
    values = []
    size = []
    

    for i in preds:
        if subreddits[i[0]] not in predictions:
            predictions.append(subreddits[i[0]])
            values.append(i[1])
            size.append((i[1]+1)*10)
        
    predictions = predictions[:6]
    values = values[:6]
    predictions.reverse()
    values.reverse()

    fig = go.Figure(data=[go.Scatter(
                x=values, y=predictions,
                mode='markers',
                marker=dict(
                            color=values,
                            size=size
                            )
                )])
 
    return fig.to_json()

