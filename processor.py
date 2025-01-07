import numpy as np
import json
from sklearn.preprocessing import LabelEncoder

# Load user encoder classes
user_encoder = LabelEncoder()
user_encoder.classes_ = np.load("./utils/user_encoder_classes.npy")

# Load movie encoder classes
movie_encoder = LabelEncoder()
movie_encoder.classes_ = np.load("./utils/movie_encoder_classes.npy")

# Load movie ID to title mapping
with open("./utils/movie_id_to_title.json", "r") as f:
    movie_id_to_title = json.load(f)
    print("loaded successfully")

def pre_process(user_id):
    """
    Pre-process the input user_id to prepare it for model inference.
    Args:
        user_id (int): The user ID to be processed.
    Returns:
        tuple: Processed user vector and movie vector.
    """
    try:
        # Check if user_id exists in the dataset
        if user_id not in user_encoder.classes_:
            raise ValueError(f"User ID {user_id} not found in the dataset.")
        
        # Encode the user ID
        user_vector = np.full((len(movie_encoder.classes_),), user_encoder.transform([user_id])[0])
        
        # Prepare movie vector for all movies
        movie_vector = np.arange(len(movie_encoder.classes_))
        
        return user_vector, movie_vector
    except Exception as e:
        raise ValueError(f"Error in preprocessing: {e}")

def post_process(model_output):
    """
    Post-process the model output to generate movie recommendations.
    Args:
        model_output (np.ndarray): Model predictions for all movies for a user.
    Returns:
        list: Top 10 recommended movie titles.
    """
    try:
        # Get the top 10 movie indices with highest predictions
        top_indices = np.argsort(model_output.flatten())[::-1][:10]
        
        # Map movie indices back to original IDs
        recommended_movie_ids = movie_encoder.inverse_transform(top_indices)
        
        # Map IDs to titles
        recommended_movies = [movie_id_to_title.get(mid, "Unknown Movie") for mid in recommended_movie_ids]
        
        return recommended_movies
    except Exception as e:
        raise ValueError(f"Error in postprocessing: {e}")
