import pandas as pd
import numpy as np
import pickle
import matplotlib
# Check if we're running in a GUI environment
import os
if 'DISPLAY' not in os.environ and os.name != 'nt':
    # Use non-interactive backend if no display available
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys
import logging
from sklearn.metrics import precision_score, recall_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_predictions():
    """Load original and improved predictions"""
    try:
        print("Loading prediction files...")
        
        # Check if prediction files exist
        original_pred_path = 'data/predictions.csv'
        improved_pred_path = 'data/improved_predictions.csv'
        
        if not os.path.exists(original_pred_path):
            print(f"Error: {original_pred_path} not found. Run model_training.py first.")
            return None, None
            
        if not os.path.exists(improved_pred_path):
            print(f"Error: {improved_pred_path} not found. Run improved_model.py first.")
            return None, None
        
        # Load both prediction files
        original_preds = pd.read_csv(original_pred_path)
        improved_preds = pd.read_csv(improved_pred_path)
        
        print(f"Original predictions shape: {original_preds.shape}")
        print(f"Improved predictions shape: {improved_preds.shape}")
        
        return original_preds, improved_preds
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return None, None

def compare_predictions(original_preds, improved_preds):
    """Compare the original and improved model predictions"""
    try:
        print("\nComparing prediction statistics...")
        
        # Extract the prediction columns
        if 'predicted_sku' in original_preds.columns and 'predicted_sku' in improved_preds.columns:
            orig_pred_vals = original_preds['predicted_sku']
            impr_pred_vals = improved_preds['predicted_sku']
            
            # Create statistics summary
            stats_df = pd.DataFrame({
                'Original': [orig_pred_vals.min(), orig_pred_vals.max(), 
                             orig_pred_vals.mean(), orig_pred_vals.std()],
                'Improved': [impr_pred_vals.min(), impr_pred_vals.max(), 
                             impr_pred_vals.mean(), impr_pred_vals.std()]
            }, index=['Min', 'Max', 'Mean', 'Std'])
            
            print("\nPrediction Statistics:")
            print(stats_df)
            
            # Calculate correlation between predictions
            corr = np.corrcoef(orig_pred_vals, impr_pred_vals)[0, 1]
            print(f"\nCorrelation between original and improved predictions: {corr:.4f}")
            
            return orig_pred_vals, impr_pred_vals
        else:
            print("Error: 'predicted_sku' column not found in one or both prediction files")
            return None, None
    except Exception as e:
        print(f"Error comparing predictions: {e}")
        return None, None

def evaluate_if_true_values_exist(predictions_df):
    """Evaluate model performance against true values if they exist"""
    try:
        # Check if 'sku' column exists (true values)
        if 'sku' in predictions_df.columns and 'predicted_sku' in predictions_df.columns:
            print("\nTrue values found. Calculating performance metrics...")
            
            y_true = predictions_df['sku']
            y_pred = predictions_df['predicted_sku']
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R² Score: {r2:.4f}")
            
            # Calculate percentage error
            pct_error = np.abs(y_true - y_pred) / y_true * 100
            mean_pct_error = np.mean(pct_error)
            median_pct_error = np.median(pct_error)
            
            print(f"Mean Percentage Error: {mean_pct_error:.2f}%")
            print(f"Median Percentage Error: {median_pct_error:.2f}%")
            
            return True
        else:
            print("\nNo true values found for evaluation.")
            return False
    except Exception as e:
        print(f"Error evaluating against true values: {e}")
        return False

def plot_prediction_comparison(orig_pred, impr_pred):
    """Create visualization comparing original and improved predictions"""
    try:
        print("\nGenerating visualizations...")
        
        # Create directory for plots if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        # Check if display is available
        has_display = True
        if os.name != 'nt':  # Not Windows
            has_display = 'DISPLAY' in os.environ
        
        # Create a random sample of predictions to visualize
        # (using full dataset might be too large for visualization)
        sample_size = min(1000, len(orig_pred))
        sample_indices = np.random.choice(len(orig_pred), sample_size, replace=False)
        
        # Create a dataframe with both predictions
        plot_df = pd.DataFrame({
            'Original': orig_pred.iloc[sample_indices],
            'Improved': impr_pred.iloc[sample_indices]
        })
        
        # 1. Scatterplot comparing original and improved predictions
        plt.figure(figsize=(10, 8))
        plt.scatter(plot_df['Original'], plot_df['Improved'], alpha=0.5)
        
        # Add identity line
        max_val = max(plot_df['Original'].max(), plot_df['Improved'].max())
        min_val = min(plot_df['Original'].min(), plot_df['Improved'].min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('Original vs Improved Model Predictions')
        plt.xlabel('Original Model Predictions')
        plt.ylabel('Improved Model Predictions')
        plt.grid(True, alpha=0.3)
        plt.savefig('plots/prediction_comparison_scatter.png', dpi=300)
        if has_display:
            print("Displaying scatter plot... (close window to continue)")
            plt.show()  # Display the plot if possible
        plt.close()
        
        # 2. Distribution of predictions
        plt.figure(figsize=(12, 6))
        
        # Use log scale for better visualization if values have large range
        log_scale = plot_df['Original'].max() / plot_df['Original'].min() > 100
        
        if log_scale:
            bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
            plt.xscale('log')
        else:
            bins = 50
            
        plt.hist(plot_df['Original'], bins=bins, alpha=0.5, label='Original')
        plt.hist(plot_df['Improved'], bins=bins, alpha=0.5, label='Improved')
        
        plt.title('Distribution of Predictions')
        plt.xlabel('Prediction Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('plots/prediction_distribution.png', dpi=300)
        if has_display:
            print("Displaying distribution plot... (close window to continue)")
            plt.show()  # Display the plot if possible
        plt.close()
        
        # 3. Box plot for comparison
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=plot_df)
        plt.title('Prediction Distribution Comparison')
        plt.ylabel('Prediction Value')
        if log_scale:
            plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig('plots/prediction_boxplot.png', dpi=300)
        if has_display:
            print("Displaying boxplot... (close window to continue)")
            plt.show()  # Display the plot if possible
        plt.close()
        
        print("Visualizations saved to 'plots' directory with high resolution (300 DPI).")
        if not has_display:
            print("No display detected. Plots saved to files but not shown on screen.")
            print("You can view them in the 'plots' directory.")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def load_recommendations():
    """
    Load recommendation data
    """
    try:
        logging.info("Loading recommendation data...")
        
        # Check if recommendations file exists
        rec_path = 'data/recommendations.csv'
        
        if not os.path.exists(rec_path):
            logging.error("Recommendations file not found. Run train_model.py first.")
            return None
            
        # Load recommendations
        recommendations = pd.read_csv(rec_path)
        
        logging.info(f"Recommendations shape: {recommendations.shape}")
        logging.info(f"Recommendations for {recommendations['user_id'].nunique()} unique users")
        
        return recommendations
        
    except Exception as e:
        logging.error(f"Error loading recommendations: {e}")
        return None

def load_game_data():
    """
    Load game metadata
    """
    try:
        logging.info("Loading game metadata...")
        
        # Check if game metadata file exists
        game_path = 'data/game_metadata.csv'
        
        if not os.path.exists(game_path):
            logging.error("Game metadata file not found.")
            return None
            
        # Load game data
        game_data = pd.read_csv(game_path)
        
        logging.info(f"Game data shape: {game_data.shape}")
        
        return game_data
        
    except Exception as e:
        logging.error(f"Error loading game data: {e}")
        return None

def load_model():
    """
    Load the trained model
    """
    try:
        logging.info("Loading trained model...")
        
        # Check if model file exists
        model_path = 'model/xgb_game_recommender.pkl'
        
        if not os.path.exists(model_path):
            logging.error("Model file not found. Run train_model.py first.")
            return None
            
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logging.info("Model loaded successfully.")
        
        return model
        
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

def calculate_recommendation_metrics(recommendations):
    """Calculate recommendation quality metrics"""
    try:
        print("\nCalculating recommendation metrics...")
        
        # Check if we have the necessary columns
        required_cols = ['user_id', 'game_id', 'predicted_interest']
        missing_cols = [col for col in required_cols if col not in recommendations.columns]
        
        if missing_cols:
            print(f"Error: Missing required columns in recommendations: {missing_cols}")
            return
        
        # Number of unique users and games
        unique_users = recommendations['user_id'].nunique()
        unique_games = recommendations['game_id'].nunique()
        
        print(f"Unique users with recommendations: {unique_users}")
        print(f"Unique games recommended: {unique_games}")
        
        # Average number of recommendations per user
        recs_per_user = recommendations.groupby('user_id').size().mean()
        print(f"Average recommendations per user: {recs_per_user:.2f}")
        
        # Diversity metrics - only calculate if genre column exists
        if 'genre' in recommendations.columns:
            # Genre diversity
            genre_diversity = recommendations['genre'].nunique() / unique_games
            print(f"Genre diversity: {genre_diversity:.2f}")
            
            # Calculate entropy of genre distribution
            genre_entropy = calculate_entropy(recommendations['genre'])
            print(f"Genre entropy (higher = more diverse): {genre_entropy:.4f}")
        else:
            print("Genre column not found in recommendations. Skipping genre diversity metrics.")
        
        # Interest score distribution
        interest_mean = recommendations['predicted_interest'].mean()
        interest_std = recommendations['predicted_interest'].std()
        interest_min = recommendations['predicted_interest'].min()
        interest_max = recommendations['predicted_interest'].max()
        
        print(f"\nPredicted interest statistics:")
        print(f"Mean: {interest_mean:.4f}")
        print(f"Std Dev: {interest_std:.4f}")
        print(f"Min: {interest_min:.4f}")
        print(f"Max: {interest_max:.4f}")
        
        return True
    
    except Exception as e:
        print(f"Error in recommendation metrics calculation: {e}")
        return False

def calculate_entropy(series):
    """
    Calculate entropy (diversity) of a series
    """
    value_counts = series.value_counts(normalize=True)
    return -np.sum(value_counts * np.log2(value_counts))

def visualize_recommendations(recommendations, game_data, output_dir='plots'):
    """
    Create visualizations for recommendation evaluation
    """
    logging.info("Creating recommendation visualizations...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if display is available
    has_display = True
    if os.name != 'nt':  # Not Windows
        has_display = 'DISPLAY' in os.environ
        
    # 1. Genre Distribution
    plt.figure(figsize=(12, 6))
    genre_counts = recommendations['genre'].value_counts()
    sns.barplot(x=genre_counts.values, y=genre_counts.index)
    plt.title('Genre Distribution in Recommendations')
    plt.xlabel('Number of Recommendations')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/genre_distribution.png', dpi=300)
    if has_display:
        plt.show()
    plt.close()
    
    # 2. Interest Score Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(recommendations['predicted_interest'], kde=True)
    plt.title('Distribution of Predicted Interest Scores')
    plt.xlabel('Predicted Interest Score')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/interest_score_distribution.png', dpi=300)
    if has_display:
        plt.show()
    plt.close()
    
    # 3. Recommendation count by user
    plt.figure(figsize=(10, 6))
    user_rec_counts = recommendations.groupby('user_id').size()
    sns.histplot(user_rec_counts, kde=True)
    plt.title('Number of Recommendations per User')
    plt.xlabel('Number of Recommendations')
    plt.ylabel('Count of Users')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/recs_per_user.png', dpi=300)
    if has_display:
        plt.show()
    plt.close()
    
    # 4. If game data includes ratings, plot rating vs. predicted interest
    if game_data is not None and 'rating' in game_data.columns:
        # Merge recommendations with game data
        merged_data = recommendations.merge(
            game_data[['game_id', 'rating', 'popularity_score']], 
            on='game_id', 
            how='left'
        )
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='rating', y='predicted_interest', data=merged_data, alpha=0.6)
        plt.title('Game Rating vs. Predicted Interest')
        plt.xlabel('Game Rating')
        plt.ylabel('Predicted Interest Score')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/rating_vs_interest.png', dpi=300)
        if has_display:
            plt.show()
        plt.close()
        
        # 5. Popularity vs. predicted interest
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='popularity_score', y='predicted_interest', data=merged_data, alpha=0.6)
        plt.title('Game Popularity vs. Predicted Interest')
        plt.xlabel('Game Popularity Score')
        plt.ylabel('Predicted Interest Score')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/popularity_vs_interest.png', dpi=300)
        if has_display:
            plt.show()
        plt.close()
    
    logging.info(f"Visualizations saved to {output_dir} directory with high resolution (300 DPI).")
    if not has_display:
        logging.info("No display detected. Plots saved to files but not shown on screen.")

def simulate_user_satisfaction(recommendations, simulation_rounds=1000):
    """
    Simulate user satisfaction with recommendations
    """
    logging.info("Simulating user satisfaction...")
    
    # Parameters that influence satisfaction
    interest_weight = 0.7     # How much predicted interest influences satisfaction
    diversity_weight = 0.3    # How much genre diversity influences satisfaction
    
    # Aggregate recommendations by user
    user_recs = recommendations.groupby('user_id')
    
    # Store satisfaction scores
    satisfaction_scores = []
    
    # For each user
    for user_id, user_data in user_recs:
        # Calculate average interest score
        avg_interest = user_data['predicted_interest'].mean()
        
        # Calculate genre diversity
        genre_entropy = calculate_entropy(user_data['genre'])
        
        # Normalize interest between 0-1 (assuming interest is between 0-1)
        normalized_interest = min(max(avg_interest, 0), 1)
        
        # Normalize entropy (max entropy = log2(number of all possible genres))
        max_entropy = np.log2(9)  # Assuming 9 genres as in the sample data
        normalized_diversity = min(genre_entropy / max_entropy, 1)
        
        # Calculate satisfaction score
        satisfaction = (interest_weight * normalized_interest + 
                        diversity_weight * normalized_diversity)
        
        # Scale to 1-5 range
        satisfaction_5pt = 1 + satisfaction * 4
        
        satisfaction_scores.append(satisfaction_5pt)
    
    # Calculate overall satisfaction metrics
    avg_satisfaction = np.mean(satisfaction_scores)
    min_satisfaction = np.min(satisfaction_scores)
    max_satisfaction = np.max(satisfaction_scores)
    
    logging.info(f"Simulated user satisfaction (1-5 scale):")
    logging.info(f"  Average: {avg_satisfaction:.2f}")
    logging.info(f"  Min: {min_satisfaction:.2f}")
    logging.info(f"  Max: {max_satisfaction:.2f}")
    
    # Visualization of satisfaction distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(satisfaction_scores, kde=True)
    plt.title('Distribution of Simulated User Satisfaction Scores')
    plt.xlabel('Satisfaction Score (1-5)')
    plt.ylabel('Count of Users')
    plt.axvline(avg_satisfaction, color='red', linestyle='--', 
                label=f'Average: {avg_satisfaction:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/user_satisfaction.png', dpi=300)
    
    # Check if display is available
    has_display = True
    if os.name != 'nt':  # Not Windows
        has_display = 'DISPLAY' in os.environ
    if has_display:
        plt.show()
    plt.close()
    
    return {'avg_satisfaction': avg_satisfaction, 
            'min_satisfaction': min_satisfaction,
            'max_satisfaction': max_satisfaction}

def evaluate_model():
    """
    Main function to evaluate the model and recommendations
    """
    try:
        # Load recommendations
        recommendations = load_recommendations()
        if recommendations is None:
            return
        
        # Load game data
        game_data = load_game_data()
        if game_data is None:
            return
        
        # Load model
        model = load_model()
        if model is None:
            return
        
        # Merge game data with recommendations to get genre information
        if 'game_id' in recommendations.columns and 'game_id' in game_data.columns:
            recommendations = recommendations.merge(
                game_data[['game_id', 'genre', 'price', 'rating', 'popularity_score']], 
                on='game_id', 
                how='left'
            )
            logging.info("Added game metadata to recommendations.")
        
        # Calculate recommendation metrics
        calculate_recommendation_metrics(recommendations)
        
        # Visualize recommendations
        visualize_recommendations(recommendations, game_data)
        
        # Simulate user satisfaction
        satisfaction = simulate_user_satisfaction(recommendations)
        
        # Print summary
        logging.info("="*50)
        logging.info("RECOMMENDATION SYSTEM EVALUATION SUMMARY")
        logging.info("="*50)
        logging.info(f"Generated recommendations for {recommendations['user_id'].nunique()} users")
        logging.info(f"Total recommendations: {len(recommendations)}")
        logging.info(f"Average interest score: {recommendations['predicted_interest'].mean():.4f}")
        
        # Only calculate genre entropy if genre column exists
        if 'genre' in recommendations.columns:
            logging.info(f"Genre diversity (entropy): {calculate_entropy(recommendations['genre']):.4f}")
        
        if satisfaction and 'avg_satisfaction' in satisfaction:
            logging.info(f"Simulated user satisfaction: {satisfaction['avg_satisfaction']:.2f}/5.0")
        logging.info("="*50)
        
        return True
    
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    evaluate_model() 