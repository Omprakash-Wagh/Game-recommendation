import pandas as pd
import numpy as np
import sys
import traceback
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import os

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

def add_interaction_features(data):
    """Add interaction features to improve model performance"""
    print("Adding interaction features...")
    
    # Create interaction features
    data['user_query'] = data['user'] * data['query']
    data['user_category'] = data['user'] * data['category']
    data['query_category'] = data['query'] * data['category']
    data['hour_day'] = data['hour'] * data['day']
    data['time_diff_abs'] = np.abs(data['time_diff'])
    data['time_diff_squared'] = data['time_diff'] ** 2
    
    # Log transforms for numerical features (handling negative values)
    data['time_diff_sign'] = np.sign(data['time_diff'])
    data['time_diff_log'] = np.log1p(np.abs(data['time_diff']))
    
    print(f"Data shape after adding interaction features: {data.shape}")
    return data

def prepare_features(train_data, test_data):
    """Prepare features and target variables for training with feature engineering"""
    try:
        print("Preparing features...")
        
        # Add interaction features
        train_data = add_interaction_features(train_data)
        test_data = add_interaction_features(test_data)
        
        # Separate features and target for train data
        X_train = train_data.drop(columns=['sku'])
        y_train = train_data['sku']
        
        # Apply log transform to target (handle negative values)
        y_train_sign = np.sign(y_train)
        y_train_log = np.log1p(np.abs(y_train))
        y_train_transformed = y_train_sign * y_train_log
        
        print(f"Target min: {y_train.min()}, max: {y_train.max()}, mean: {y_train.mean()}")
        print(f"Transformed target min: {y_train_transformed.min()}, max: {y_train_transformed.max()}, mean: {y_train_transformed.mean()}")
        
        # Prepare test data
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
            X_train, y_train_transformed, test_size=0.2, random_state=42
        )
        
        print(f"Train split shapes: X={X_train_split.shape}, y={y_train_split.shape}")
        print(f"Validation split shapes: X={X_val.shape}, y={y_val.shape}")
        
        feature_names = X_train.columns.tolist()
        print(f"Feature names: {feature_names}")
        
        return X_train_split, X_val, y_train_split, y_val, X_test, y_test, y_train_sign, feature_names
    except Exception as e:
        print(f"Error preparing features: {e}")
        traceback.print_exc()
        return None, None, None, None, None, None, None, None

def tune_hyperparameters(X_train, y_train, X_val, y_val, feature_names):
    """Tune hyperparameters for XGBoost model"""
    try:
        print("Tuning hyperparameters...")
        
        # Convert to DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        
        # Define parameter grid
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 6, 9],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5]
        }
        
        # Use a smaller grid for demonstration (comment this out for full tuning)
        param_grid = {
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 6],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'min_child_weight': [1]
        }
        
        best_params = {}
        best_score = float('inf')
        
        # Simple grid search (for demonstration)
        for eta in param_grid['learning_rate']:
            for depth in param_grid['max_depth']:
                for subsample in param_grid['subsample']:
                    for colsample in param_grid['colsample_bytree']:
                        for min_child in param_grid['min_child_weight']:
                            params = {
                                'objective': 'reg:squarederror',
                                'eval_metric': 'rmse',
                                'eta': eta,
                                'max_depth': depth,
                                'subsample': subsample,
                                'colsample_bytree': colsample,
                                'min_child_weight': min_child,
                                'seed': 42
                            }
                            
                            print(f"Testing parameters: eta={eta}, depth={depth}, subsample={subsample}, colsample={colsample}, min_child={min_child}")
                            
                            # Train with early stopping
                            evals = [(dtrain, 'train'), (dval, 'validation')]
                            model = xgb.train(
                                params,
                                dtrain,
                                num_boost_round=300,
                                evals=evals,
                                early_stopping_rounds=20,
                                verbose_eval=False
                            )
                            
                            # Get best score
                            score = model.best_score
                            if score < best_score:
                                best_score = score
                                best_params = params
                                best_model = model
                                
                            print(f"  Score: {score:.4f} (best: {best_score:.4f})")
        
        print("\nBest parameters:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        print(f"Best validation RMSE: {best_score:.4f}")
        
        return best_model, best_params
    except Exception as e:
        print(f"Error tuning hyperparameters: {e}")
        traceback.print_exc()
        return None, None

def train_model(X_train, X_val, y_train, y_val, feature_names, params=None):
    """Train XGBoost model with given parameters"""
    try:
        print("Training XGBoost model...")
        
        # Create DMatrix datasets for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        
        # Use default parameters if none provided
        if params is None:
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
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        xgb.plot_importance(model, max_num_features=20)
        plt.title('Feature Importance')
        
        # Create directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/feature_importance.png')
        plt.close()
        
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        traceback.print_exc()
        return None

def make_predictions(model, X_test, feature_names, y_train_sign=None):
    """Make predictions on test data with inverse transformation"""
    try:
        print("Making predictions on test data...")
        dtest = xgb.DMatrix(X_test, feature_names=feature_names)
        
        # Get predictions (these are in log space)
        predictions_log = model.predict(dtest)
        
        # Inverse transform
        if y_train_sign is not None:
            print("Applying inverse transform to predictions...")
            # Get the sign and apply expm1
            predictions = np.sign(predictions_log) * (np.exp(np.abs(predictions_log)) - 1)
        else:
            predictions = predictions_log
        
        print(f"Prediction summary: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")
        return predictions
    except Exception as e:
        print(f"Error making predictions: {e}")
        traceback.print_exc()
        return None

def save_model(model, filename='model/xgboost_model_improved.pkl'):
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

def save_predictions(predictions, test_data, filename='data/improved_predictions.csv'):
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
        
        # Prepare features with engineering
        X_train, X_val, y_train, y_val, X_test, y_test, y_train_sign, feature_names = prepare_features(train_data, test_data)
        if X_train is None:
            sys.exit(1)
        
        # Tune hyperparameters (comment out for faster execution)
        best_model, best_params = tune_hyperparameters(X_train, y_train, X_val, y_val, feature_names)
        if best_model is None:
            # Train with default parameters
            print("Hyperparameter tuning failed, training with default parameters...")
            model = train_model(X_train, X_val, y_train, y_val, feature_names)
        else:
            # Use the best model from tuning
            model = best_model
        
        if model is None:
            sys.exit(1)
        
        # Make predictions
        predictions = make_predictions(model, X_test, feature_names, y_train_sign)
        if predictions is None:
            sys.exit(1)
        
        # Save model and predictions
        save_model(model)
        save_predictions(predictions, test_data)
        
        print("Improved model training and prediction completed successfully.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1) 