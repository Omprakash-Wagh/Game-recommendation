import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """
    Load BestBuy search data and Xbox game metadata
    Returns:
        search_data: DataFrame containing user search data
        game_data: DataFrame containing Xbox game information
    """
    try:
        logging.info("Loading BestBuy search data and Xbox game metadata...")
        
        # Check if data directory exists, create if not
        if not os.path.exists('data'):
            os.makedirs('data')
            logging.warning("Data directory not found. Created new directory.")
            logging.warning("Please place bestbuy_searches.csv and game_metadata.csv in the data directory.")
            return None, None
        
        # Load search data
        search_data_path = 'data/bestbuy_searches.csv'
        if not os.path.exists(search_data_path):
            # Create sample data for demonstration if file doesn't exist
            logging.warning(f"File {search_data_path} not found. Creating sample data for demonstration.")
            search_data = create_sample_search_data()
        else:
            search_data = pd.read_csv(search_data_path)
            
        # Load game metadata
        game_data_path = 'data/game_metadata.csv'
        if not os.path.exists(game_data_path):
            # Create sample game data for demonstration if file doesn't exist
            logging.warning(f"File {game_data_path} not found. Creating sample data for demonstration.")
            game_data = create_sample_game_data()
        else:
            game_data = pd.read_csv(game_data_path)
        
        logging.info(f"Search data shape: {search_data.shape}")
        logging.info(f"Game data shape: {game_data.shape}")
        
        return search_data, game_data
    
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None, None

def create_sample_search_data(n_samples=10000):
    """
    Create sample search data for demonstration purposes
    """
    np.random.seed(42)
    
    # Create user IDs (1000 unique users)
    user_ids = np.random.randint(1, 1001, n_samples)
    
    # Create search timestamps in the last 90 days
    now = datetime.now()
    timestamps = [now - pd.Timedelta(days=np.random.randint(1, 90)) for _ in range(n_samples)]
    
    # Common Xbox game search terms
    game_terms = [
        "halo", "forza", "gears of war", "sea of thieves", "call of duty", 
        "assassin's creed", "fifa", "madden", "nba 2k", "minecraft", 
        "destiny", "elder scrolls", "fallout", "star wars", "battlefield",
        "xbox series x games", "best xbox games", "xbox exclusives",
        "racing games xbox", "shooter games xbox", "rpg games xbox"
    ]
    
    # Generate search queries
    search_queries = []
    for _ in range(n_samples):
        if np.random.random() < 0.7:  # 70% specific game searches
            query = np.random.choice(game_terms)
            if np.random.random() < 0.3:  # Sometimes add modifiers
                modifier = np.random.choice(["best price", "review", "gameplay", "buy", "digital code"])
                query = f"{query} {modifier}"
        else:  # 30% general searches
            category = np.random.choice(["xbox", "game", "console", "controller", "headset"])
            modifier = np.random.choice(["new", "best", "cheap", "top rated", "sale", "deal"])
            query = f"{modifier} {category}"
            
        search_queries.append(query)
    
    # Generate page views (1-15)
    page_views = np.random.randint(1, 16, n_samples)
    
    # Generate session duration in seconds (10s to 30min)
    session_duration = np.random.randint(10, 1800, n_samples)
    
    # Create clicks on game products (boolean)
    clicked_game = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    # Create purchase data (boolean, less frequent than clicks)
    purchased = np.zeros(n_samples)
    for i in range(n_samples):
        if clicked_game[i] == 1:
            purchased[i] = np.random.choice([0, 1], p=[0.8, 0.2])
    
    # Create DataFrame
    search_data = pd.DataFrame({
        'user_id': user_ids,
        'timestamp': timestamps,
        'search_query': search_queries,
        'page_views': page_views,
        'session_duration': session_duration,
        'clicked_game': clicked_game,
        'purchased': purchased
    })
    
    # Sort by user_id and timestamp
    search_data = search_data.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    # Save the sample data
    search_data.to_csv('data/bestbuy_searches.csv', index=False)
    logging.info("Created sample search data and saved to data/bestbuy_searches.csv")
    
    return search_data

def create_sample_game_data(n_games=100):
    """
    Create sample Xbox game metadata for demonstration purposes
    """
    np.random.seed(42)
    
    # Game titles - combination of real titles and generated ones
    real_titles = [
        "Halo Infinite", "Forza Horizon 5", "Gears 5", "Sea of Thieves",
        "Call of Duty: Modern Warfare", "Assassin's Creed Valhalla",
        "FIFA 23", "Madden NFL 23", "NBA 2K23", "Minecraft",
        "Destiny 2", "The Elder Scrolls V: Skyrim", "Fallout 4",
        "Star Wars Jedi: Fallen Order", "Battlefield 2042"
    ]
    
    # Generate additional titles
    adjectives = ["Ultimate", "Legendary", "Epic", "Extreme", "Super"]
    nouns = ["Quest", "Warriors", "Legends", "Champions", "Adventure"]
    more_titles = [f"{adj} {noun}" for adj in adjectives for noun in nouns]
    
    # Combine and select titles
    all_titles = real_titles + more_titles
    if n_games <= len(all_titles):
        game_titles = np.random.choice(all_titles, n_games, replace=False)
    else:
        # If we need more games than we have titles, we'll need to generate more or allow duplicates
        game_titles = np.random.choice(all_titles, n_games, replace=True)
    
    # Game IDs
    game_ids = [f"XBOX{i:05d}" for i in range(1, n_games+1)]
    
    # Genres
    genres = ["Action", "Adventure", "RPG", "Sports", "Racing", "Shooter", "Strategy", "Simulation", "Puzzle"]
    game_genres = np.random.choice(genres, n_games)
    
    # Release dates in the last 5 years
    release_dates = []
    for _ in range(n_games):
        year = np.random.randint(datetime.now().year - 5, datetime.now().year + 1)
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 29)  # Simplify to avoid month length issues
        release_dates.append(f"{year}-{month:02d}-{day:02d}")
    
    # Prices
    base_prices = np.random.choice([19.99, 29.99, 39.99, 49.99, 59.99, 69.99], n_games)
    
    # Ratings
    ratings = np.random.uniform(1, 5, n_games).round(1)
    
    # Number of ratings
    num_ratings = np.random.randint(10, 5000, n_games)
    
    # Popularity score (combination of ratings, number of ratings, and recency)
    popularity_scores = []
    for i in range(n_games):
        # Convert release date to datetime
        release_date = datetime.strptime(release_dates[i], "%Y-%m-%d")
        
        # Calculate days since release
        days_since_release = (datetime.now() - release_date).days
        
        # Calculate popularity based on ratings, number of ratings, and recency
        rating_factor = ratings[i] / 5.0
        rating_count_factor = min(num_ratings[i] / 1000, 1)  # Cap at 1
        recency_factor = max(1 - (days_since_release / 365), 0)  # Higher for newer games
        
        popularity = (rating_factor * 0.4 + rating_count_factor * 0.3 + recency_factor * 0.3) * 100
        popularity_scores.append(round(popularity, 2))
    
    # Create DataFrame
    game_data = pd.DataFrame({
        'game_id': game_ids,
        'title': game_titles,
        'genre': game_genres,
        'release_date': release_dates,
        'price': base_prices,
        'rating': ratings,
        'num_ratings': num_ratings,
        'popularity_score': popularity_scores
    })
    
    # Save the sample data
    game_data.to_csv('data/game_metadata.csv', index=False)
    logging.info("Created sample game data and saved to data/game_metadata.csv")
    
    return game_data

def clean_search_data(search_data):
    """
    Clean and preprocess search data
    """
    logging.info("Cleaning search data...")
    
    # Convert timestamp to datetime if it's not already
    if search_data['timestamp'].dtype == 'object':
        search_data['timestamp'] = pd.to_datetime(search_data['timestamp'])
    
    # Add derived time features
    search_data['hour'] = search_data['timestamp'].dt.hour
    search_data['day'] = search_data['timestamp'].dt.day
    search_data['day_of_week'] = search_data['timestamp'].dt.dayofweek
    search_data['month'] = search_data['timestamp'].dt.month
    search_data['is_weekend'] = search_data['day_of_week'].isin([5, 6]).astype(int)
    
    # Clean search queries (lowercase, remove special characters)
    search_data['search_query_clean'] = search_data['search_query'].str.lower()
    search_data['search_query_clean'] = search_data['search_query_clean'].apply(
        lambda x: re.sub(r'[^\w\s]', '', str(x))
    )
    
    # Handle missing values
    for col in search_data.columns:
        if search_data[col].isna().sum() > 0:
            if search_data[col].dtype == 'object':
                search_data[col] = search_data[col].fillna('')
            else:
                search_data[col] = search_data[col].fillna(search_data[col].median())
    
    logging.info(f"Cleaned search data shape: {search_data.shape}")
    return search_data

def link_searches_to_games(search_data, game_data):
    """
    Link search queries to specific Xbox games
    """
    logging.info("Linking search queries to games...")
    
    # Create a mapping of game titles to game IDs
    game_mapping = dict(zip(game_data['title'].str.lower(), game_data['game_id']))
    
    # Additional mapping for partial matches
    partial_mapping = {}
    for title, game_id in game_mapping.items():
        words = title.split()
        for word in words:
            if len(word) > 3:  # Only use meaningful words
                partial_mapping[word] = game_id
    
    # Function to find matching game for a search query
    def find_matching_game(query):
        query = query.lower()
        
        # Check for exact match
        for title, game_id in game_mapping.items():
            if title in query:
                return game_id
        
        # Check for partial matches
        for keyword, game_id in partial_mapping.items():
            if keyword in query:
                return game_id
                
        return None
    
    # Apply the matching function
    search_data['matched_game_id'] = search_data['search_query_clean'].apply(find_matching_game)
    
    # Calculate match rate
    match_rate = (search_data['matched_game_id'].notna().sum() / len(search_data)) * 100
    logging.info(f"Successfully matched {match_rate:.2f}% of searches to games")
    
    return search_data

def prepare_user_game_matrix(search_data, game_data):
    """
    Create a user-game interaction matrix for training
    """
    logging.info("Creating user-game interaction matrix...")
    
    # Get relevant columns
    user_game_data = search_data[['user_id', 'matched_game_id', 'clicked_game', 'purchased']]
    
    # Remove rows with no matched game
    user_game_data = user_game_data.dropna(subset=['matched_game_id'])
    
    # Group by user and game, aggregate interactions
    user_game_agg = user_game_data.groupby(['user_id', 'matched_game_id']).agg({
        'clicked_game': 'sum',
        'purchased': 'sum'
    }).reset_index()
    
    # Create interest score (purchases are weighted more heavily than clicks)
    user_game_agg['interest_score'] = user_game_agg['clicked_game'] + 5 * user_game_agg['purchased']
    
    # Normalize interest score
    max_score = user_game_agg['interest_score'].max()
    user_game_agg['interest_normalized'] = user_game_agg['interest_score'] / max_score
    
    # Merge game metadata
    user_game_matrix = user_game_agg.merge(game_data, left_on='matched_game_id', right_on='game_id', how='left')
    
    # For any games that were matched but not in our game metadata, use placeholder values
    placeholder_cols = ['title', 'genre', 'release_date', 'price', 'rating', 'num_ratings', 'popularity_score']
    for col in placeholder_cols:
        if col in user_game_matrix.columns:
            user_game_matrix[col] = user_game_matrix[col].fillna(user_game_matrix[col].mode()[0])
    
    logging.info(f"User-game matrix shape: {user_game_matrix.shape}")
    return user_game_matrix

def split_train_test(user_game_matrix, test_size=0.2):
    """
    Split the data into training and testing sets
    """
    logging.info(f"Splitting data with test_size={test_size}...")
    
    # Get unique users
    unique_users = user_game_matrix['user_id'].unique()
    
    # Randomly select users for test set
    np.random.seed(42)
    test_users = np.random.choice(
        unique_users, 
        size=int(len(unique_users) * test_size), 
        replace=False
    )
    
    # Split data
    test_data = user_game_matrix[user_game_matrix['user_id'].isin(test_users)]
    train_data = user_game_matrix[~user_game_matrix['user_id'].isin(test_users)]
    
    logging.info(f"Train data shape: {train_data.shape}")
    logging.info(f"Test data shape: {test_data.shape}")
    
    return train_data, test_data

def process_data():
    """
    Main function to process the data
    """
    try:
        # Load data
        search_data, game_data = load_data()
        if search_data is None or game_data is None:
            return
        
        # Clean search data
        search_data = clean_search_data(search_data)
        
        # Link searches to games
        search_data = link_searches_to_games(search_data, game_data)
        
        # Create user-game matrix
        user_game_matrix = prepare_user_game_matrix(search_data, game_data)
        
        # Split into train and test
        train_data, test_data = split_train_test(user_game_matrix)
        
        # Save processed data
        if not os.path.exists('data'):
            os.makedirs('data')
        
        train_data.to_csv('data/processed_train.csv', index=False)
        test_data.to_csv('data/processed_test.csv', index=False)
        
        logging.info("Data processing completed successfully.")
        logging.info("Processed data saved to data/processed_train.csv and data/processed_test.csv")
        
        return train_data, test_data
        
    except Exception as e:
        logging.error(f"Error in data processing: {e}")
        return None, None

if __name__ == "__main__":
    process_data() 