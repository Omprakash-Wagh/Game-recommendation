import pandas as pd
import numpy as np
import os
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
import traceback
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_processed_data(data_type="processed"):
    """
    Load the processed or engineered training and testing data
    
    Args:
        data_type (str): Type of data to load - "processed" or "engineered"
    """
    try:
        logging.info(f"Loading {data_type} data...")
        
        if data_type == "processed":
            train_path = 'data/processed_train.csv'
            test_path = 'data/processed_test.csv'
        elif data_type == "engineered":
            train_path = 'data/train_engineered.csv'
            test_path = 'data/test_engineered.csv'
        else:
            logging.error(f"Invalid data_type: {data_type}")
            return None, None
        
        # Check if files exist
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            logging.error(f"{data_type.capitalize()} data files not found.")
            if data_type == "processed":
                logging.error("Run data_processing.py first.")
            else:
                logging.error("Run feature_engineering.py first.")
            return None, None
            
        # Load data
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        logging.info(f"{data_type.capitalize()} train data shape: {train_data.shape}")
        logging.info(f"{data_type.capitalize()} test data shape: {test_data.shape}")
        
        return train_data, test_data
        
    except Exception as e:
        logging.error(f"Error loading {data_type} data: {e}")
        traceback.print_exc()
        return None, None

def add_interaction_features(data):
    """
    Add interaction features to improve model performance
    """
    logging.info("Adding interaction features...")
    
    # Create interaction features based on available columns
    if 'user_id' in data.columns and 'interest_score' in data.columns:
        data['user_interest'] = data['user_id'] * data['interest_score']
    
    if 'price' in data.columns and 'rating' in data.columns:
        data['price_rating'] = data['price'] * data['rating']
    
    if 'popularity_score' in data.columns and 'interest_normalized' in data.columns:
        data['popularity_interest'] = data['popularity_score'] * data['interest_normalized']
    
    # Create time-based features if release_date is available
    if 'release_date' in data.columns:
        # Convert to datetime if it's not already
        if data['release_date'].dtype == 'object':
            data['release_date'] = pd.to_datetime(data['release_date'])
            
        # Calculate days since release
        from datetime import datetime
        current_date = datetime.now()
        data['days_since_release'] = (current_date - data['release_date']).dt.days
        
        # Log transform for numerical features (handling negative values)
        data['days_log'] = np.log1p(data['days_since_release'])
    
    logging.info(f"Data shape after adding interaction features: {data.shape}")
    return data

def prepare_features(train_data, test_data, target_col='interest_normalized', add_interactions=False):
    """
    Prepare features and target variables for training
    
    Args:
        train_data: Training DataFrame
        test_data: Testing DataFrame
        target_col: Target column name
        add_interactions: Whether to add interaction features
    """
    try:
        logging.info(f"Preparing features with target column: {target_col}")
        
        # Check if target column exists
        if target_col not in train_data.columns:
            logging.error(f"Target column '{target_col}' not found in training data.")
            return None, None, None, None, None, None, None
        
        # Add interaction features if requested
        if add_interactions:
            train_data = add_interaction_features(train_data)
            test_data = add_interaction_features(test_data)
        
        # Identify feature columns to use (exclude non-numeric and identifier columns)
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
            return None, None, None, None, None, None, None
        
        logging.info(f"Selected {len(feature_cols)} feature columns.")
        
        # Split features and target
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        
        # Apply log transform to target if needed (for skewed distributions)
        if y_train.skew() > 1.0:
            logging.info(f"Target is skewed ({y_train.skew():.2f}), applying log transform.")
            y_train_sign = np.sign(y_train)
            y_train_log = np.log1p(np.abs(y_train))
            y_train_transformed = y_train_sign * y_train_log
            logging.info(f"Original target - min: {y_train.min()}, max: {y_train.max()}, mean: {y_train.mean()}")
            logging.info(f"Transformed target - min: {y_train_transformed.min()}, max: {y_train_transformed.max()}, mean: {y_train_transformed.mean()}")
        else:
            y_train_transformed = y_train
            y_train_sign = None
        
        # Prepare test data
        X_test = test_data[feature_cols]
        if target_col in test_data.columns:
            y_test = test_data[target_col]
            logging.info("Test data contains target column.")
        else:
            y_test = None
            logging.info("Test data does not contain target column.")
        
        # Split into training and validation sets
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train_transformed, test_size=0.2, random_state=42
        )
        
        logging.info(f"Training set shape: X={X_train_split.shape}, y={y_train_split.shape}")
        logging.info(f"Validation set shape: X={X_val.shape}, y={y_val.shape}")
        
        return X_train_split, X_val, y_train_split, y_val, X_test, y_test, feature_cols, y_train_sign
    
    except Exception as e:
        logging.error(f"Error preparing features: {e}")
        traceback.print_exc()
        return None, None, None, None, None, None, None, None

def tune_hyperparameters(X_train, y_train, X_val=None, y_val=None, cv=3, quick_mode=True):
    """
    Tune XGBoost hyperparameters
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        cv: Number of cross-validation folds
        quick_mode: Whether to use a smaller parameter grid for faster execution
    """
    logging.info("Tuning hyperparameters...")
    
    # Parameter grid
    if quick_mode:
        param_grid = {
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [100, 200],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'min_child_weight': [1]
        }
        logging.info("Using quick parameter grid for faster tuning.")
    else:
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5]
        }
        logging.info("Using full parameter grid for comprehensive tuning.")
    
    # Base XGBoost model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    )
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=cv,
        verbose=1,
        n_jobs=-1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    logging.info(f"Best hyperparameters: {best_params}")
    
    return best_params

def train_model(X_train, y_train, X_val, y_val, params=None, feature_names=None):
    """
    Train the XGBoost model
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        params: Model parameters (optional)
        feature_names: Feature names (optional)
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
    
    # Train the model
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # Make prediction on validation set
    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    mae = mean_absolute_error(y_val, val_preds)
    r2 = r2_score(y_val, val_preds)
    
    logging.info(f"Validation RMSE: {rmse:.4f}")
    logging.info(f"Validation MAE: {mae:.4f}")
    logging.info(f"Validation R²: {r2:.4f}")
    
    # Feature importance
    if feature_names is not None:
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        logging.info("\nTop 10 Feature Importance:")
        for i, (feature, importance) in enumerate(zip(feature_importance['Feature'].head(10), 
                                                    feature_importance['Importance'].head(10))):
            logging.info(f"{i+1}. {feature}: {importance:.4f}")
    
    return model, rmse

def make_predictions(model, X_test, y_train_sign=None):
    """
    Make predictions on test data
    
    Args:
        model: Trained model
        X_test: Test features
        y_train_sign: Sign of training target (for inverse transform)
    """
    logging.info("Making predictions on test data...")
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform if log transform was applied
    if y_train_sign is not None:
        logging.info("Applying inverse transform to predictions...")
        predictions_sign = np.sign(predictions)
        predictions_exp = np.expm1(np.abs(predictions))
        predictions = predictions_sign * predictions_exp
    
    logging.info(f"Prediction summary: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")
    return predictions

def visualize_feature_importance(model, feature_names, output_dir='plots'):
    """
    Visualize feature importance
    
    Args:
        model: Trained model
        feature_names: Feature names
        output_dir: Output directory for plots
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
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300)
    plt.close()
    
    logging.info(f"Feature importance plot saved to {output_dir}/feature_importance.png")

def generate_recommendations(model, test_data, feature_cols, output_path='data/recommendations.csv'):
    """
    Generate game recommendations for test users
    
    Args:
        model: Trained model
        test_data: Test data
        feature_cols: Feature columns
        output_path: Output path for recommendations
    """
    logging.info("Generating game recommendations...")
    
    # Get unique users
    if 'user_id' in test_data.columns:
        unique_users = test_data['user_id'].unique()
        logging.info(f"Generating recommendations for {len(unique_users)} users")
        
        # Create recommendations DataFrame
        recommendations = []
        
        # For each user, predict interest scores for all games
        for user_id in unique_users[:5]:  # Limit to first 5 users for demonstration
            user_data = test_data[test_data['user_id'] == user_id]
            
            # Get features
            X_user = user_data[feature_cols]
            
            # Predict interest scores
            interest_scores = model.predict(X_user)
            
            # Create user recommendations
            user_recs = pd.DataFrame({
                'user_id': user_id,
                'game_id': user_data['game_id'],
                'title': user_data['title'],
                'predicted_interest': interest_scores
            })
            
            # Sort by predicted interest
            user_recs = user_recs.sort_values('predicted_interest', ascending=False)
            
            # Add to recommendations
            recommendations.append(user_recs.head(5))  # Top 5 recommendations per user
        
        # Combine all recommendations
        if recommendations:
            all_recommendations = pd.concat(recommendations)
            
            # Save recommendations
            all_recommendations.to_csv(output_path, index=False)
            logging.info(f"Recommendations saved to {output_path}")
            
            return all_recommendations
        else:
            logging.warning("No recommendations generated.")
            return None
    else:
        logging.warning("No user_id column found in test data. Cannot generate personalized recommendations.")
        return None

def save_model(model, filename='model/unified_game_recommender.pkl'):
    """
    Save the trained model to disk
    
    Args:
        model: Trained model
        filename: Output filename
    """
    try:
        logging.info(f"Saving model to {filename}...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save model
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        
        logging.info("Model saved successfully.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        traceback.print_exc()

def train(data_type="processed", target_col="interest_normalized", tune_params=True, 
          add_interactions=True, quick_mode=True):
    """
    Main function to train the model
    
    Args:
        data_type: Type of data to use - "processed" or "engineered"
        target_col: Target column name
        tune_params: Whether to tune hyperparameters
        add_interactions: Whether to add interaction features
        quick_mode: Whether to use quick mode for hyperparameter tuning
    """
    try:
        # Load data
        train_data, test_data = load_processed_data(data_type)
        if train_data is None or test_data is None:
            return
        
        # Prepare features
        X_train, X_val, y_train, y_val, X_test, y_test, feature_cols, y_train_sign = prepare_features(
            train_data, test_data, target_col, add_interactions
        )
        if X_train is None:
            return
        
        # Tune hyperparameters if requested
        if tune_params:
            params = tune_hyperparameters(X_train, y_train, X_val, y_val, quick_mode=quick_mode)
        else:
            params = None
        
        # Train model
        model, rmse = train_model(X_train, y_train, X_val, y_val, params, feature_cols)
        
        # Visualize feature importance
        visualize_feature_importance(model, feature_cols)
        
        # Make predictions
        predictions = make_predictions(model, X_test, y_train_sign)
        
        # Generate recommendations
        recommendations = generate_recommendations(model, test_data, feature_cols)
        
        # Save model
        save_model(model)
        
        logging.info("Model training and recommendation generation completed successfully.")
        return model, predictions, recommendations
    
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train a game recommendation model')
    parser.add_argument('--data-type', choices=['processed', 'engineered'], default='processed',
                        help='Type of data to use (processed or engineered)')
    parser.add_argument('--target-col', default='interest_normalized',
                        help='Target column name')
    parser.add_argument('--no-tune', action='store_true',
                        help='Skip hyperparameter tuning')
    parser.add_argument('--no-interactions', action='store_true',
                        help='Skip adding interaction features')
    parser.add_argument('--full-tune', action='store_true',
                        help='Use full hyperparameter tuning (slower)')
    
    args = parser.parse_args()
    
    # Train model
    train(
        data_type=args.data_type,
        target_col=args.target_col,
        tune_params=not args.no_tune,
        add_interactions=not args.no_interactions,
        quick_mode=not args.full_tune
    )
