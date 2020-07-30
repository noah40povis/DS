from fastapi import APIRouter, HTTPException
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import stylecloud

from .predict import Item, pred 
from joblib import load

router = APIRouter()
sfw_model = load('nn_cleaned.joblib')
sfw_tfidf = load('tfidf_cleaned.joblib')
ns_model = load('subreddit_mvp.joblib')

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


@router.post('/wordclouds'):
async def wordclouds(postbody: Item):
    """generate word clouds
    """
    df = pd.read_csv('25325_subreddits.csv')
    list_in = pred(postbody, model= ns_model)['recommendations']
    data = {} #dict of serialized imgs
    for i, subreddit in enumerate(list_in):
        x = df[df['subreddit']== subreddit]
        y = x['text'].str.cat(sep=', ')
        filename = f'reddit{i}.png'
        stylecloud.gen_stylecloud(text = y,
                            icon_name='fab fa-reddit-alien',
                            palette='colorbrewer.diverging.Spectral_11',
                            background_color='black',
                            gradient='horizontal',
                            output_name=filename)
  
  
    with open(filename, mode='rb') as file:
        img = file.read()
    key = f"img{i}"
    data[key] = base64.encodebytes(img).decode("utf-8")
    
    return data

