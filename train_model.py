import pandas as pd
import numpy as np
import xgboost as xgb
import os
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_engineered_data():
    """
    Load engineered training and testing data
    """
    try:
        logging.info("Loading engineered data...")
        
        # Check if engineered files exist
        train_path = 'data/train_engineered.csv'
        test_path = 'data/test_engineered.csv'
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            logging.error("Engineered data files not found. Run feature_engineering.py first.")
            return None, None
            
        # Load data
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        logging.info(f"Engineered train data shape: {train_data.shape}")
        logging.info(f"Engineered test data shape: {test_data.shape}")
        
        return train_data, test_data
        
    except Exception as e:
        logging.error(f"Error loading engineered data: {e}")
        return None, None

def prepare_training_data(train_data, target_col='interest_normalized', test_size=0.2):
    """
    Prepare data for model training
    """
    logging.info(f"Preparing training data with target column: {target_col}")
    
    # Check if target column exists
    if target_col not in train_data.columns:
        logging.error(f"Target column '{target_col}' not found in training data.")
        return None, None, None, None
    
    # Identify feature columns to use
    exclude_cols = [
        'user_id', 'matched_game_id', 'game_id', 'title', 'release_date',
        'genre', 'price_category', 'game_age_category', 'genre_price_interaction'
    ]
    
    # Only exclude columns that exist
    exclude_cols = [col for col in exclude_cols if col in train_data.columns]
    
    # Add target column to exclude list
    exclude_cols.append(target_col)
    
    # Select features (exclude non-numeric and identifier columns)
    feature_cols = [col for col in train_data.columns if col not in exclude_cols]
    
    # Check if we have features
    if len(feature_cols) == 0:
        logging.error("No valid feature columns found.")
        return None, None, None, None
    
    logging.info(f"Selected {len(feature_cols)} feature columns.")
    
    # Split features and target
    X = train_data[feature_cols]
    y = train_data[target_col]
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    logging.info(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"Validation set shape: X={X_val.shape}, y={y_val.shape}")
    
    return X_train, X_val, y_train, y_val, feature_cols

def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """
    Tune XGBoost hyperparameters
    """
    logging.info("Tuning hyperparameters...")
    
    # Parameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3, 5]
    }
    
    # Base XGBoost model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    )
    
    # Use smaller parameter combinations for faster execution
    simplified_param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'min_child_weight': [1]
    }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=simplified_param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    logging.info(f"Best hyperparameters: {best_params}")
    
    return best_params

def train_model(X_train, y_train, X_val, y_val, params=None):
    """
    Train the XGBoost model
    """
    logging.info("Training XGBoost model...")
    
    # Default parameters if none provided
    if params is None:
        params = {
            'max_depth': 5,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1
        }
    
    # Initialize model with parameters
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        eval_metric='rmse',
        **params
    )
    
    # Train the model without extra parameters that might not be supported in this version
    model.fit(X_train, y_train)
    
    # Make prediction on validation set
    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    mae = mean_absolute_error(y_val, val_preds)
    r2 = r2_score(y_val, val_preds)
    
    logging.info(f"Validation RMSE: {rmse:.4f}")
    logging.info(f"Validation MAE: {mae:.4f}")
    logging.info(f"Validation R²: {r2:.4f}")
    
    return model, rmse

def visualize_feature_importance(model, feature_names, output_dir='plots'):
    """
    Visualize feature importance
    """
    logging.info("Creating feature importance visualization...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png')
    plt.close()
    
    logging.info(f"Feature importance visualization saved to {output_dir}/feature_importance.png")

def generate_recommendations(model, test_data, feature_cols, output_path='data/recommendations.csv'):
    """
    Generate game recommendations for test users
    """
    logging.info("Generating game recommendations...")
    
    # Prepare test features
    X_test = test_data[feature_cols]
    
    # Generate interest score predictions
    test_data['predicted_interest'] = model.predict(X_test)
    
    # Get top recommendations for each user
    user_recommendations = []
    
    for user_id in test_data['user_id'].unique():
        # Get user's data
        user_data = test_data[test_data['user_id'] == user_id]
        
        # Sort by predicted interest
        user_top_games = user_data.sort_values('predicted_interest', ascending=False)
        
        # Take top 5 recommendations
        top_5 = user_top_games.head(5)[['user_id', 'game_id', 'title', 'genre', 'predicted_interest']]
        
        # Add to recommendations list
        user_recommendations.append(top_5)
    
    # Combine all recommendations
    all_recommendations = pd.concat(user_recommendations)
    
    # Save recommendations
    all_recommendations.to_csv(output_path, index=False)
    
    logging.info(f"Recommendations saved to {output_path}")
    logging.info(f"Generated recommendations for {len(test_data['user_id'].unique())} users")
    
    return all_recommendations

def evaluate_recommendations(recommendations, output_dir='plots'):
    """
    Evaluate and visualize recommendations
    """
    logging.info("Evaluating recommendations...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Analyze genre distribution in recommendations
    genre_dist = recommendations['genre'].value_counts().reset_index()
    genre_dist.columns = ['Genre', 'Count']
    
    # Create genre distribution plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Count', y='Genre', data=genre_dist)
    plt.title('Game Genre Distribution in Recommendations')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/recommendation_genres.png')
    plt.close()
    
    # Analyze interest score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(recommendations['predicted_interest'], kde=True)
    plt.title('Distribution of Predicted Interest Scores')
    plt.xlabel('Predicted Interest')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/recommendation_scores.png')
    plt.close()
    
    # User-based recommendation count
    user_rec_counts = recommendations.groupby('user_id').size().reset_index()
    user_rec_counts.columns = ['User ID', 'Recommendation Count']
    
    logging.info(f"Average recommendations per user: {user_rec_counts['Recommendation Count'].mean():.2f}")
    
    logging.info("Recommendation evaluation completed and visualizations saved.")

def save_model(model, filename='model/xgb_game_recommender.pkl'):
    """
    Save the trained model to disk
    """
    logging.info(f"Saving model to {filename}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save model
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    
    logging.info("Model saved successfully.")

def train():
    """
    Main function to train the model
    """
    try:
        # Load engineered data
        train_data, test_data = load_engineered_data()
        if train_data is None or test_data is None:
            return
        
        # Prepare training data
        X_train, X_val, y_train, y_val, feature_cols = prepare_training_data(
            train_data, target_col='interest_normalized'
        )
        if X_train is None:
            return
        
        # Tune hyperparameters
        best_params = tune_hyperparameters(X_train, y_train, X_val, y_val)
        
        # Train model with best parameters
        model, validation_rmse = train_model(X_train, y_train, X_val, y_val, best_params)
        
        # Visualize feature importance
        visualize_feature_importance(model, feature_cols)
        
        # Generate recommendations
        recommendations = generate_recommendations(model, test_data, feature_cols)
        
        # Evaluate recommendations
        evaluate_recommendations(recommendations)
        
        # Save model
        save_model(model)
        
        logging.info("Model training and recommendation generation completed successfully.")
        
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train() 