import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from vipas import logger
from model_components import StockPredictionModel
logger_client = logger.LoggerClient(__name__)

# Load data and prepare encoders/mappings during module import
user_encoder, movie_encoder, movie_id_to_title = None, None, None

logger_client.info("Was able to load class")

import json

# Path to your JSON file
json_file_path = "movie_id_to_title_first_100.json"

# Load the JSON data
try:
    with open(json_file_path, "r") as file:
        data = json.load(file)
        logger_client.info("JSON data loaded successfully!")
        # Access the data as a Python dictionary
        logger_client.info(data)
except FileNotFoundError:
    logger_client.info(f"Error: The file '{json_file_path}' was not found.")
except json.JSONDecodeError as e:
    logger_client.info(f"Error decoding JSON: {e}")


# Pre-process Function: Generate Inputs for Model
def pre_process(user_id):
    try:
        if user_encoder is None or movie_encoder is None:
            raise Exception("Encoders not initialized. Failed to load dataset.")

        # Encode the user ID
        encoded_user_id = user_encoder.transform([user_id])[0]
        
        # Generate input arrays
        user_input = np.full((len(movie_encoder.classes_),), encoded_user_id)
        movie_ids = np.arange(len(movie_encoder.classes_))  # All movie IDs encoded
        
        return user_input, movie_ids
    except Exception as e:
        logger_client.error(f"Error in pre_process: {e}")
        return None, None


# Post-process Function: Interpret Model Output
def post_process(predictions):
    try:
        if movie_id_to_title is None:
            raise Exception("Movie ID-to-title mapping not initialized.")

        # Combine predictions with movie IDs
        movie_ids = movie_encoder.inverse_transform(np.arange(len(predictions)))
        movie_predictions = list(zip(movie_ids, predictions.flatten()))

        # Sort movies by predicted ratings in descending order
        movie_predictions = sorted(movie_predictions, key=lambda x: x[1], reverse=True)

        # Map movie IDs to titles and return with ratings
        top_movies = [(movie_id_to_title.get(mid, "Unknown Movie"), rating) for mid, rating in movie_predictions]

        return top_movies
    except Exception as e:
        logger_client.error(f"Error in post_process: {e}")
        return []
