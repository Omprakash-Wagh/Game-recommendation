import pandas as pd
import numpy as np
import sys
import traceback
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import xgboost as xgb

def load_processed_data():
    """Load processed train and test data"""
    try:
        print("Loading processed data...")
        train_data = pd.read_csv('data/processed_train.csv')
        test_data = pd.read_csv('data/processed_test.csv')
        print(f"Processed train data shape: {train_data.shape}")
        print(f"Processed test data shape: {test_data.shape}")
        return train_data, test_data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run data_processing.py first to preprocess the data.")
        return None, None
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        traceback.print_exc()
        return None, None

def prepare_features(train_data, test_data):
    """Prepare features and target variables for training"""
    try:
        print("Preparing features...")
        
        # Separate features and target for train data
        X_train = train_data.drop(columns=['sku'])
        y_train = train_data['sku']
        
        # Prepare test data - make sure to check if sku column exists
        if 'sku' in test_data.columns:
            X_test = test_data.drop(columns=['sku'])
            y_test = test_data['sku']
            print("Test data contains target 'sku' column.")
        else:
            X_test = test_data
            y_test = None
            print("Test data does not contain target 'sku' column.")
        
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        
        # Split train data into train and validation sets
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        print(f"Train split shapes: X={X_train_split.shape}, y={y_train_split.shape}")
        print(f"Validation split shapes: X={X_val.shape}, y={y_val.shape}")
        
        return X_train_split, X_val, y_train_split, y_val, X_test, y_test
    except Exception as e:
        print(f"Error preparing features: {e}")
        traceback.print_exc()
        return None, None, None, None, None, None

def train_model(X_train, X_val, y_train, y_val):
    """Train XGBoost model"""
    try:
        print("Training XGBoost model...")
        
        # Create DMatrix datasets for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Define parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
        
        # Train the model with early stopping
        evals = [(dtrain, 'train'), (dval, 'validation')]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=50
        )
        
        # Evaluate the model
        val_preds = model.predict(dval)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"Validation RMSE: {val_rmse:.4f}")
        
        # Feature importance
        feature_importance = model.get_score(importance_type='gain')
        print("\nFeature Importance:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.2f}")
        
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        traceback.print_exc()
        return None

def make_predictions(model, X_test):
    """Make predictions on test data"""
    try:
        print("Making predictions on test data...")
        dtest = xgb.DMatrix(X_test)
        predictions = model.predict(dtest)
        print(f"Prediction summary: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")
        return predictions
    except Exception as e:
        print(f"Error making predictions: {e}")
        traceback.print_exc()
        return None

def save_model(model, filename='model/xgboost_model.pkl'):
    """Save model to disk"""
    try:
        print(f"Saving model to {filename}...")
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")
        traceback.print_exc()

def save_predictions(predictions, test_data, filename='data/predictions.csv'):
    """Save predictions to disk"""
    try:
        print(f"Saving predictions to {filename}...")
        result = test_data.copy()
        result['predicted_sku'] = predictions
        result.to_csv(filename, index=False)
        print("Predictions saved successfully.")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        # Load processed data
        train_data, test_data = load_processed_data()
        if train_data is None or test_data is None:
            sys.exit(1)
        
        # Prepare features
        X_train, X_val, y_train, y_val, X_test, y_test = prepare_features(train_data, test_data)
        if X_train is None:
            sys.exit(1)
        
        # Train model
        model = train_model(X_train, X_val, y_train, y_val)
        if model is None:
            sys.exit(1)
        
        # Make predictions
        predictions = make_predictions(model, X_test)
        if predictions is None:
            sys.exit(1)
        
        # Save model and predictions
        save_model(model)
        save_predictions(predictions, test_data)
        
        print("Model training and prediction completed successfully.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1) 