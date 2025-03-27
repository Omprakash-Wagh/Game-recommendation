import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_processed_data():
    """
    Load the processed training and testing data
    """
    try:
        logging.info("Loading processed data...")
        
        # Check if processed files exist
        train_path = 'data/processed_train.csv'
        test_path = 'data/processed_test.csv'
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            logging.error("Processed data files not found. Run data_processing.py first.")
            return None, None
            
        # Load data
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        logging.info(f"Processed train data shape: {train_data.shape}")
        logging.info(f"Processed test data shape: {test_data.shape}")
        
        return train_data, test_data
        
    except Exception as e:
        logging.error(f"Error loading processed data: {e}")
        return None, None

def extract_temporal_features(data):
    """
    Extract advanced temporal features from the data
    """
    logging.info("Extracting temporal features...")
    
    # Convert release_date to datetime if it's not already
    if data['release_date'].dtype == 'object':
        data['release_date'] = pd.to_datetime(data['release_date'])
    
    # Calculate days since release
    current_date = datetime.now()
    data['days_since_release'] = (current_date - data['release_date']).dt.days
    
    # Create age category
    data['game_age_category'] = pd.cut(
        data['days_since_release'],
        bins=[-1, 30, 90, 365, 730, float('inf')],
        labels=['new_release', 'recent', 'within_year', 'within_two_years', 'older']
    )
    
    # Create price-to-age ratio (value indicator)
    data['price_age_ratio'] = data['price'] / (data['days_since_release'] + 1)  # Add 1 to avoid division by zero
    
    logging.info("Temporal features added.")
    return data

def create_genre_features(data):
    """
    Create features related to game genres
    """
    logging.info("Creating genre-based features...")
    
    # Create one-hot encoded genre features
    genre_dummies = pd.get_dummies(data['genre'], prefix='genre')
    data = pd.concat([data, genre_dummies], axis=1)
    
    # Calculate genre popularity
    genre_popularity = data.groupby('genre')['popularity_score'].mean().reset_index()
    genre_popularity.columns = ['genre', 'genre_avg_popularity']
    
    # Merge genre popularity back to main data
    data = data.merge(genre_popularity, on='genre', how='left')
    
    # Calculate normalized interest by genre
    if 'interest_normalized' in data.columns and 'user_id' in data.columns:
        # Group by user and genre, get average interest
        user_genre_interest = data.groupby(['user_id', 'genre'])['interest_normalized'].mean().reset_index()
        user_genre_interest.columns = ['user_id', 'genre', 'user_genre_interest']
        
        # Merge back to main data
        data = data.merge(user_genre_interest, on=['user_id', 'genre'], how='left')
        
        # Fill missing values
        data['user_genre_interest'] = data['user_genre_interest'].fillna(data['interest_normalized'])
    
    logging.info("Genre features added.")
    return data

def create_price_features(data):
    """
    Create price-related features
    """
    logging.info("Creating price-related features...")
    
    # Create price categories
    data['price_category'] = pd.cut(
        data['price'],
        bins=[-1, 19.99, 39.99, 59.99, float('inf')],
        labels=['budget', 'standard', 'premium', 'deluxe']
    )
    
    # One-hot encode price categories
    price_dummies = pd.get_dummies(data['price_category'], prefix='price')
    data = pd.concat([data, price_dummies], axis=1)
    
    # Create price-to-rating ratio (value for money indicator)
    data['price_rating_ratio'] = data['price'] / data['rating']
    
    logging.info("Price features added.")
    return data

def create_user_features(data):
    """
    Create user-related features
    """
    logging.info("Creating user-related features...")
    
    if 'user_id' in data.columns:
        # Calculate user activity metrics
        user_activity = data.groupby('user_id').agg({
            'interest_score': ['count', 'mean', 'sum'],
            'clicked_game': 'sum',
            'purchased': 'sum'
        })
        
        # Flatten multi-level columns
        user_activity.columns = ['user_game_count', 'user_avg_interest', 'user_total_interest', 
                                'user_total_clicks', 'user_total_purchases']
        user_activity = user_activity.reset_index()
        
        # Create purchase rate
        user_activity['user_purchase_rate'] = user_activity['user_total_purchases'] / user_activity['user_total_clicks']
        user_activity['user_purchase_rate'] = user_activity['user_purchase_rate'].fillna(0)
        
        # Merge back to main data
        data = data.merge(user_activity, on='user_id', how='left')
        
        # Calculate relative interest (how much user likes this game vs. their average)
        data['relative_interest'] = data['interest_normalized'] / data['user_avg_interest']
        data['relative_interest'] = data['relative_interest'].fillna(1)  # Default to 1 if no other interests
    
    logging.info("User features added.")
    return data

def create_interaction_features(data):
    """
    Create interaction features between different variables
    """
    logging.info("Creating interaction features...")
    
    # Genre and price interaction
    data['genre_price_interaction'] = data['genre'] + '_' + data['price_category'].astype(str)
    
    # Convert to dummy variables if needed
    if len(data['genre_price_interaction'].unique()) < 20:  # Reasonable number of categories
        genre_price_dummies = pd.get_dummies(data['genre_price_interaction'], prefix='genre_price')
        data = pd.concat([data, genre_price_dummies], axis=1)
    
    # Game age and popularity interaction
    data['age_popularity'] = data['days_since_release'] * data['popularity_score']
    
    # Normalize this feature
    data['age_popularity_norm'] = (data['age_popularity'] - data['age_popularity'].min()) / \
                                 (data['age_popularity'].max() - data['age_popularity'].min())
    
    # User interest and game popularity
    if 'interest_normalized' in data.columns and 'popularity_score' in data.columns:
        data['interest_popularity'] = data['interest_normalized'] * data['popularity_score'] / 100
    
    logging.info("Interaction features added.")
    return data

def scale_numerical_features(train_data, test_data):
    """
    Scale numerical features
    """
    logging.info("Scaling numerical features...")
    
    # Identify numerical columns
    numerical_cols = [
        'price', 'rating', 'num_ratings', 'popularity_score', 'days_since_release',
        'price_age_ratio', 'price_rating_ratio', 'age_popularity_norm'
    ]
    
    # Only include columns that actually exist in the data
    numerical_cols = [col for col in numerical_cols if col in train_data.columns and col in test_data.columns]
    
    if len(numerical_cols) > 0:
        # Initialize scaler
        scaler = StandardScaler()
        
        # Fit on training data
        scaler.fit(train_data[numerical_cols])
        
        # Transform both training and test data
        train_data[numerical_cols] = scaler.transform(train_data[numerical_cols])
        test_data[numerical_cols] = scaler.transform(test_data[numerical_cols])
    
    logging.info("Numerical features scaled.")
    return train_data, test_data

def visualize_features(data, output_dir='plots'):
    """
    Create visualizations of key features
    """
    logging.info("Creating feature visualizations...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Genre Popularity
    plt.figure(figsize=(12, 6))
    sns.barplot(x='genre', y='popularity_score', data=data)
    plt.title('Game Popularity by Genre')
    plt.xlabel('Genre')
    plt.ylabel('Average Popularity Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/genre_popularity.png')
    plt.close()
    
    # 2. Price vs Rating
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='price', y='rating', hue='genre', data=data, alpha=0.7)
    plt.title('Game Price vs Rating by Genre')
    plt.xlabel('Price ($)')
    plt.ylabel('Rating (1-5)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/price_rating.png')
    plt.close()
    
    # 3. Interest by Game Age
    if 'interest_normalized' in data.columns and 'game_age_category' in data.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='game_age_category', y='interest_normalized', data=data)
        plt.title('User Interest by Game Age')
        plt.xlabel('Game Age Category')
        plt.ylabel('Normalized Interest Score')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/interest_by_age.png')
        plt.close()
    
    # 4. User Preferences (if we have enough users)
    if 'user_id' in data.columns and len(data['user_id'].unique()) < 100:
        # Get top users by activity
        top_users = data.groupby('user_id')['interest_score'].count().nlargest(10).index
        
        # Filter data for these users
        user_data = data[data['user_id'].isin(top_users)]
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x='user_id', y='interest_normalized', hue='genre', data=user_data)
        plt.title('Top User Preferences by Genre')
        plt.xlabel('User ID')
        plt.ylabel('Average Interest Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/user_preferences.png')
        plt.close()
    
    logging.info(f"Visualizations saved to {output_dir} directory.")

def engineer_features():
    """
    Main function to engineer features
    """
    try:
        # Load processed data
        train_data, test_data = load_processed_data()
        if train_data is None or test_data is None:
            return None, None
        
        # Create a copy of the data to avoid modifying the original
        train_data_fe = train_data.copy()
        test_data_fe = test_data.copy()
        
        # Extract temporal features
        train_data_fe = extract_temporal_features(train_data_fe)
        test_data_fe = extract_temporal_features(test_data_fe)
        
        # Create genre features
        train_data_fe = create_genre_features(train_data_fe)
        test_data_fe = create_genre_features(test_data_fe)
        
        # Create price features
        train_data_fe = create_price_features(train_data_fe)
        test_data_fe = create_price_features(test_data_fe)
        
        # Create user features
        train_data_fe = create_user_features(train_data_fe)
        test_data_fe = create_user_features(test_data_fe)
        
        # Create interaction features
        train_data_fe = create_interaction_features(train_data_fe)
        test_data_fe = create_interaction_features(test_data_fe)
        
        # Scale numerical features
        train_data_fe, test_data_fe = scale_numerical_features(train_data_fe, test_data_fe)
        
        # Create visualizations
        visualize_features(train_data_fe)
        
        # Save engineered data
        train_data_fe.to_csv('data/train_engineered.csv', index=False)
        test_data_fe.to_csv('data/test_engineered.csv', index=False)
        
        logging.info("Feature engineering completed successfully.")
        logging.info("Engineered data saved to data/train_engineered.csv and data/test_engineered.csv")
        
        return train_data_fe, test_data_fe
        
    except Exception as e:
        logging.error(f"Error in feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    engineer_features() 