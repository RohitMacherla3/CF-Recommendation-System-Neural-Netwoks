import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class RecommendationEngine:
    def __init__(self):
        self.final_model = self.load_model()
        self.ratings_data, self.titles_data = self.load_data()

        # convert user IDs to sequential numerical values
        user_ids = self.ratings_data['UserId'].unique()
        self.user_id_map = {user_id: i for i, user_id in enumerate(user_ids)}
        self.ratings_data['UserSeqId'] = self.ratings_data['UserId'].map(self.user_id_map)

        # convert title IDs to sequential numerical values
        title_ids = self.ratings_data['TitleId'].unique()
        self.title_id_map = {title_id: i for i, title_id in enumerate(title_ids)}
        self.ratings_data['TitleSeqId'] = self.ratings_data['TitleId'].map(self.title_id_map)

        # drop unnecessary columns
        self.titles_data = self.titles_data.drop('Unnamed: 0', axis=1)

        # join the rating and titles data
        self.final_data = pd.merge(self.ratings_data, self.titles_data, on='TitleId')

    def load_data(self):
        # Load the synthetic data
        ratings_data_syn = pd.read_csv('data/survey_ratings_synthetic.csv')
        ratings_data_syn = ratings_data_syn.rename(columns={'Review': 'Rating'})

        # load the crowdsourcing data
        ratings = pd.read_csv('data/survey_ratings.csv')

        # combine both synthetic and crowdsourcing data
        ratings = pd.concat([ratings, ratings_data_syn])

        # load the titles data
        titles = pd.read_csv('data/survey_titles.csv')
        
        return ratings, titles

    def load_model(self):
        model_architecture_path = 'Models/NN_Model1/neural_net_architecture_1.pkl'
        with open(model_architecture_path, 'rb') as f:
            model_architecture = pickle.load(f)

        model = tf.keras.models.model_from_json(model_architecture)

        model_weights_path = 'Models/NN_Model1/neural_net_weights_1.pkl'
        model.load_weights(model_weights_path)
        
        return model    

    def get_top_recommendations(self, user_id, n, thres=0):
        final_recommendation = []
        
        # get the user's sequential ID
        user_seq_id = self.user_id_map[user_id]
        
        num_users = len(self.user_id_map)
        num_titles = len(self.title_id_map)
        
        # get the inputs for the model
        title_seq_ids = np.arange(num_titles)
        user_seq_ids = np.repeat(user_seq_id, num_titles)
        
        # get the predictions from the neural network
        predictions = self.final_model.predict([user_seq_ids, title_seq_ids])
        
        # create a DataFrame with title IDs and predicted ratings
        recommendations_df = pd.DataFrame({'TitleSeqId': title_seq_ids, 'PredictedRating': predictions.flatten()})
        
        # remove the recommendations that are already seen by the user
        seen_title_ids = self.ratings_data[self.ratings_data['UserSeqId'] == user_seq_id]['TitleSeqId'].values
        recommendations_df = recommendations_df[~recommendations_df['TitleSeqId'].isin(seen_title_ids)]
        
        # Sort the recommendations by predicted rating in descending order and select the top N titles
        top_recommendations = recommendations_df[recommendations_df['PredictedRating'] >= thres].nlargest(n, 'PredictedRating')
        
        # add the recommendations and respective predicted ratings as a tuple to a list
        for _, row in top_recommendations.iterrows():
            title_seq_id = row['TitleSeqId']
            predicted_rating = row['PredictedRating']
            title_name = self.final_data[self.final_data['TitleSeqId'] == title_seq_id]['TitleName'].values[0]
            final_recommendation.append((title_name, predicted_rating))
        
        recomm_df = pd.DataFrame(final_recommendation, columns=['Title', 'Rating'])
        return recomm_df