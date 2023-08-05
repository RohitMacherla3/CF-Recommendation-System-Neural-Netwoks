import logging
logging.basicConfig(filename='fast_api.log', level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s',datefmt='%Y-%m-%d %H:%M:%S')

from fastapi import FastAPI
import asyncio
from os.path import dirname
import functools
from utils.content_based_utils_v1 import ContentBased1
from utils.content_based_utils_v3 import ContentBased3
from utils.content_based_utils_st import ContentBasedSentenceTransformer
from utils.hybrid_utils import HybridLightFM
from utils.cf_utils import COCluster
from utils.neural_network import NNRecModel
from utils.utils import get_user_scores, rerank
import json
import pandas as pd

app = FastAPI()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

us_df = get_user_scores("data/users_scores.csv")

#For user info
df_user_info = pd.read_csv("data/Survey_data_preprocessed.csv", usecols = ['UserId','What is your current age?','Country','What is your gender?'])
df_user_info.drop_duplicates(inplace=True)
df_user_info.set_index('UserId', inplace=True)
df_user_info.rename(columns={"What is your current age?": "Age", "What is your gender?": "Gender"}, inplace=True)

#For Adding Content Information if needed
df_content_info = pd.read_csv("data/Survey_data_preprocessed.csv", usecols = ['TitleId', 'Pillar'])
df_content_info.drop_duplicates(inplace=True)

cb1 = ContentBased1()
cb3 = ContentBased3()
cbst = ContentBasedSentenceTransformer()

hlfm = HybridLightFM()
cfcoc = COCluster()
nn = NNRecModel()

# Endpoint to get recommendations.
@app.get("/recommend")
async def recommend(data: dict):
    logger.info("Processing request: raw recommendations")
    
    recommendations = get_raw_recommendations(data)
    
    return recommendations.to_json(orient="table")

@app.get("/recommend_health_scores")
async def recommend_hs(data: dict):
    logger.info("Processing request: recommending by health scores")

    user_id = data['user_id']
    num_results = data['num_results']
    pref = data.get('pref', None)
    
    #Forcing to get all. Otherwise updated healthscore list may only show one Pillar
    data['number'] = -1
    recommendations = get_raw_recommendations(data)
    user_scores = us_df.loc[[user_id]].values[0].tolist()
    
    recommendations = rerank(user_scores, recommendations, num_results, pref)
    
    return recommendations.to_json(orient="table")

@app.get("/user_scores/{user_id}")
async def user_score(user_id: int):
    logger.info("Processing request: getting user scores")
    
    return dict(us_df.loc[user_id])

@app.get("/user_info/{user_id}")
async def user_info(user_id: int):
    logger.info("Processing request: getting user info")
    val = dict(df_user_info.loc[user_id])
    val['Age'] = int(val['Age']) 
    
    return val

def get_raw_recommendations(data: dict) -> pd.DataFrame:
    rec_type = data['rec_type']
    
    match rec_type:
        case 'content_based_v1':
            logger.info("Getting recommendations through Content Based Model v1")
            recommendations = cb1.get_recommendations(data)
        
        case 'content_based_v3':
            logger.info("Getting recommendations through Content Based Model v3")
            recommendations = cb3.get_recommendations(data)
        
        case 'content_based_sentence_transformer':
            logger.info("Getting recommendations through Content Based Model Sentence Transformer")
            recommendations = cbst.get_recommendations(data)
            
        case 'hybrid_light_fm':
            logger.info("Getting recommendations through Hybrid Model Light FM")
            user_id = data['query']
            recommendations = hlfm.get_all_predictions(user_id)
            
        case 'cf_cocluster':
            logger.info("Getting recommendations through Colaborative Filtering Model Co Cluster")
            user_id = data['query']
            num_recs = data['number']
            recommendations = cfcoc.get_recommendations(user_id, num_recs)
            recommendations = recommendations.merge(df_content_info, how='left', on='TitleId')
            
        case 'nn':
            logger.info("Getting recommendations through Neural Network Model")
            user_id = data['query']
            num_recs = data['number']
            recommendations = nn.get_recommendations(user_id, num_recs)
            recommendations = recommendations.merge(df_content_info, how='left', on='TitleId')
            
    return recommendations

@app.get("/titles_v1")
async def get_titles1():
    titles = cb1.get_titles().tolist()
    return {'titles': titles}

@app.get("/titles_v3")
async def get_titles3():
    titles = cb3.get_titles().tolist()
    return {'titles': titles}
    