import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pickle
import webbrowser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """
    Load necessary data for the dashboard
    """
    try:
        logging.info("Loading data for dashboard...")
        
        # Load game metadata
        game_data_path = 'data/game_metadata.csv'
        if not os.path.exists(game_data_path):
            logging.error(f"Game metadata file not found: {game_data_path}")
            return None, None, None
        
        game_data = pd.read_csv(game_data_path)
        logging.info(f"Loaded game data: {game_data.shape}")
        
        # Load recommendations
        recs_path = 'data/recommendations.csv'
        if not os.path.exists(recs_path):
            logging.error(f"Recommendations file not found: {recs_path}")
            return game_data, None, None
        
        recommendations = pd.read_csv(recs_path)
        logging.info(f"Loaded recommendations: {recommendations.shape}")
        
        # Load processed data to get user interactions
        processed_path = 'data/processed_train.csv'
        if not os.path.exists(processed_path):
            logging.error(f"Processed data file not found: {processed_path}")
            return game_data, recommendations, None
        
        user_interactions = pd.read_csv(processed_path)
        logging.info(f"Loaded user interactions: {user_interactions.shape}")
        
        return game_data, recommendations, user_interactions
        
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def simulate_sales_data(game_data):
    """
    Simulate which games are currently on sale
    """
    logging.info("Simulating sales data...")
    
    # Random seed for reproducibility
    np.random.seed(42)
    
    # Create a copy of game data
    games_with_sales = game_data.copy()
    
    # Randomly select 20% of games to be on sale
    sale_mask = np.random.choice([True, False], size=len(games_with_sales), p=[0.2, 0.8])
    games_with_sales['on_sale'] = sale_mask
    
    # Calculate sale price (10-50% off)
    discount_rates = np.random.uniform(0.1, 0.5, size=len(games_with_sales))
    games_with_sales['discount_rate'] = 0.0
    games_with_sales.loc[sale_mask, 'discount_rate'] = discount_rates[sale_mask]
    
    games_with_sales['sale_price'] = games_with_sales['price'] * (1 - games_with_sales['discount_rate'])
    games_with_sales['sale_price'] = games_with_sales['sale_price'].round(2)
    
    # Generate random sale end dates (1-14 days from now)
    today = datetime.now()
    sale_end_dates = []
    for is_on_sale in sale_mask:
        if is_on_sale:
            days = np.random.randint(1, 15)
            end_date = (today + timedelta(days=days)).strftime('%Y-%m-%d')
            sale_end_dates.append(end_date)
        else:
            sale_end_dates.append(None)
    
    games_with_sales['sale_end_date'] = sale_end_dates
    
    logging.info(f"Created sales data. {sale_mask.sum()} games are on sale.")
    return games_with_sales

def identify_sale_candidates(game_data, user_interactions):
    """
    Identify games that should get a sale based on popularity, price, and user interest
    """
    logging.info("Identifying games that should get a sale...")
    
    # Create a copy of game data
    sale_candidates = game_data.copy()
    
    # Calculate metrics from user interactions
    if user_interactions is not None and 'matched_game_id' in user_interactions.columns:
        # Calculate view-to-purchase ratio
        game_metrics = user_interactions.groupby('matched_game_id').agg({
            'clicked_game': 'sum',
            'purchased': 'sum'
        }).reset_index()
        
        # Calculate conversion rate (purchases / clicks)
        game_metrics['conversion_rate'] = game_metrics['purchased'] / game_metrics['clicked_game']
        game_metrics['conversion_rate'] = game_metrics['conversion_rate'].fillna(0)
        
        # Merge with game data
        sale_candidates = sale_candidates.merge(
            game_metrics, left_on='game_id', right_on='matched_game_id', how='left'
        )
    else:
        # If no user interaction data, use randomly generated conversion rates
        sale_candidates['clicked_game'] = np.random.randint(10, 100, size=len(sale_candidates))
        sale_candidates['purchased'] = np.random.binomial(sale_candidates['clicked_game'], 0.2)
        sale_candidates['conversion_rate'] = sale_candidates['purchased'] / sale_candidates['clicked_game']
    
    # Calculate a "sale worthiness" score
    # High popularity, low conversion rate, higher price = good candidate
    sale_candidates['popularity_norm'] = (sale_candidates['popularity_score'] - sale_candidates['popularity_score'].min()) / \
                                        (sale_candidates['popularity_score'].max() - sale_candidates['popularity_score'].min())
    
    sale_candidates['price_norm'] = (sale_candidates['price'] - sale_candidates['price'].min()) / \
                                   (sale_candidates['price'].max() - sale_candidates['price'].min())
    
    sale_candidates['conversion_norm'] = 1 - (sale_candidates['conversion_rate'] - sale_candidates['conversion_rate'].min()) / \
                                        (sale_candidates['conversion_rate'].max() - sale_candidates['conversion_rate'].min())
    
    # Calculate sale worthiness score (weighted sum)
    sale_candidates['sale_worthiness'] = (
        0.4 * sale_candidates['popularity_norm'] + 
        0.4 * sale_candidates['conversion_norm'] + 
        0.2 * sale_candidates['price_norm']
    )
    
    # Exclude games already on sale
    if 'on_sale' in sale_candidates.columns:
        sale_candidates = sale_candidates[~sale_candidates['on_sale']]
    
    # Sort by sale worthiness
    sale_candidates = sale_candidates.sort_values('sale_worthiness', ascending=False)
    
    # Calculate recommended discount based on worthiness
    sale_candidates['recommended_discount'] = 0.1 + (sale_candidates['sale_worthiness'] * 0.3)
    sale_candidates['recommended_discount'] = sale_candidates['recommended_discount'].round(2)
    
    # Calculate recommended sale price
    sale_candidates['recommended_sale_price'] = sale_candidates['price'] * (1 - sale_candidates['recommended_discount'])
    sale_candidates['recommended_sale_price'] = sale_candidates['recommended_sale_price'].round(2)
    
    logging.info("Identified sale candidates and calculated sale worthiness scores.")
    return sale_candidates

def get_popular_games(game_data, user_interactions, n=20):
    """
    Get the most popular games based on a combined metric of ratings, popularity, and user interactions
    """
    logging.info("Identifying popular games...")
    
    # Create a copy of game data
    popular_games = game_data.copy()
    
    # Normalize ratings
    popular_games['rating_norm'] = (popular_games['rating'] - popular_games['rating'].min()) / \
                                  (popular_games['rating'].max() - popular_games['rating'].min())
    
    # Normalize popularity score
    popular_games['popularity_norm'] = (popular_games['popularity_score'] - popular_games['popularity_score'].min()) / \
                                      (popular_games['popularity_score'].max() - popular_games['popularity_score'].min())
    
    # Normalize number of ratings (as a measure of popularity)
    popular_games['num_ratings_norm'] = (popular_games['num_ratings'] - popular_games['num_ratings'].min()) / \
                                       (popular_games['num_ratings'].max() - popular_games['num_ratings'].min())
    
    # Add user interaction data if available
    if user_interactions is not None and 'matched_game_id' in user_interactions.columns:
        # Count total interactions per game
        game_interactions = user_interactions.groupby('matched_game_id').agg({
            'clicked_game': 'sum',
            'purchased': 'sum',
            'interest_score': 'mean'
        }).reset_index()
        
        # Merge with popular games
        popular_games = popular_games.merge(
            game_interactions, left_on='game_id', right_on='matched_game_id', how='left'
        )
        
        # Normalize interaction metrics
        for col in ['clicked_game', 'purchased', 'interest_score']:
            if col in popular_games.columns:
                popular_games[f'{col}_norm'] = (popular_games[col] - popular_games[col].min()) / \
                                             (popular_games[col].max() - popular_games[col].min())
                popular_games[f'{col}_norm'] = popular_games[f'{col}_norm'].fillna(0)
        
        # Calculate combined popularity score
        popular_games['combined_popularity'] = (
            0.3 * popular_games['popularity_norm'] +
            0.2 * popular_games['rating_norm'] +
            0.1 * popular_games['num_ratings_norm'] +
            0.2 * popular_games.get('clicked_game_norm', 0) +
            0.2 * popular_games.get('purchased_norm', 0)
        )
    else:
        # Without user interaction data
        popular_games['combined_popularity'] = (
            0.4 * popular_games['popularity_norm'] +
            0.4 * popular_games['rating_norm'] +
            0.2 * popular_games['num_ratings_norm']
        )
    
    # Sort by combined popularity
    popular_games = popular_games.sort_values('combined_popularity', ascending=False)
    
    logging.info("Identified and ranked popular games based on combined metrics.")
    return popular_games.head(n)

def get_personalized_recommendations(recommendations, game_data, user_id):
    """
    Get personalized recommendations for a specific user
    """
    if recommendations is None or len(recommendations) == 0:
        logging.warning(f"No recommendations available for user {user_id}")
        return pd.DataFrame()
    
    # Filter recommendations for this user
    user_recs = recommendations[recommendations['user_id'] == user_id]
    
    if len(user_recs) == 0:
        logging.warning(f"No recommendations found for user {user_id}")
        return pd.DataFrame()
    
    # Sort by predicted interest
    user_recs = user_recs.sort_values('predicted_interest', ascending=False)
    
    # Merge with game data to get additional info
    if game_data is not None:
        user_recs = user_recs.merge(
            game_data[['game_id', 'price', 'rating', 'popularity_score', 'release_date']],
            on='game_id',
            how='left'
        )
    
    # Add any sale info if available
    if 'on_sale' in game_data.columns:
        sale_info = game_data[['game_id', 'on_sale', 'sale_price', 'discount_rate', 'sale_end_date']]
        user_recs = user_recs.merge(sale_info, on='game_id', how='left')
    
    logging.info(f"Generated personalized recommendations for user {user_id}: {len(user_recs)} games")
    return user_recs

def generate_dashboard_tables(output_dir='dashboard'):
    """
    Generate all dashboard tables
    """
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        
        # Load data
        game_data, recommendations, user_interactions = load_data()
        if game_data is None:
            return
        
        # Add sales data
        games_with_sales = simulate_sales_data(game_data)
        
        # 1. Generate table of games on sale
        games_on_sale = games_with_sales[games_with_sales['on_sale']].copy()
        games_on_sale = games_on_sale[[
            'game_id', 'title', 'genre', 'price', 'sale_price', 'discount_rate', 'sale_end_date',
            'rating', 'popularity_score'
        ]]
        games_on_sale['discount_percentage'] = (games_on_sale['discount_rate'].fillna(0) * 100).round(0).astype(int)
        games_on_sale['savings'] = (games_on_sale['price'] - games_on_sale['sale_price']).round(2)
        
        # Save to CSV
        games_on_sale.to_csv(f'{output_dir}/games_on_sale.csv', index=False)
        logging.info(f"Saved games on sale to {output_dir}/games_on_sale.csv")
        
        # 2. Generate table of sale candidates
        sale_candidates = identify_sale_candidates(games_with_sales, user_interactions)
        sale_candidates_table = sale_candidates[[
            'game_id', 'title', 'genre', 'price', 'recommended_sale_price', 'recommended_discount',
            'popularity_score', 'rating', 'conversion_rate', 'sale_worthiness'
        ]].head(20)
        sale_candidates_table['recommended_discount_percentage'] = (sale_candidates_table['recommended_discount'].fillna(0) * 100).round(0).astype(int)
        
        # Save to CSV
        sale_candidates_table.to_csv(f'{output_dir}/sale_candidates.csv', index=False)
        logging.info(f"Saved sale candidates to {output_dir}/sale_candidates.csv")
        
        # 3. Generate table of popular games
        popular_games = get_popular_games(games_with_sales, user_interactions)
        popular_games_table = popular_games[[
            'game_id', 'title', 'genre', 'price', 'rating', 'num_ratings', 'popularity_score', 
            'combined_popularity', 'on_sale', 'sale_price'
        ]].copy()
        
        # Add "sale status" column for easier reading
        popular_games_table['sale_status'] = 'Regular Price'
        popular_games_table.loc[popular_games_table['on_sale'].fillna(False), 'sale_status'] = 'On Sale'
        
        # Save to CSV
        popular_games_table.to_csv(f'{output_dir}/popular_games.csv', index=False)
        logging.info(f"Saved popular games to {output_dir}/popular_games.csv")
        
        # 4. Generate personalized recommendations
        if recommendations is not None:
            # Get unique users
            users = recommendations['user_id'].unique()
            
            # Sample 5 users for demonstration
            sample_users = np.random.choice(users, min(5, len(users)), replace=False)
            
            # Generate recommendations for each sample user
            for user_id in sample_users:
                user_recs = get_personalized_recommendations(recommendations, games_with_sales, user_id)
                
                if len(user_recs) > 0:
                    user_recs_table = user_recs[[
                        'game_id', 'title', 'genre', 'predicted_interest', 'price', 'rating', 
                        'popularity_score', 'on_sale', 'sale_price', 'discount_rate'
                    ]].copy()
                    
                    # Add "sale status" column for easier reading
                    user_recs_table['sale_status'] = 'Regular Price'
                    user_recs_table.loc[user_recs_table['on_sale'].fillna(False) == True, 'sale_status'] = 'On Sale'
                    
                    # Save to CSV
                    user_recs_table.to_csv(f'{output_dir}/user_{user_id}_recommendations.csv', index=False)
                    logging.info(f"Saved personalized recommendations for user {user_id}")
        
        # Create a summary table with key stats
        summary = {
            'Total Games': len(game_data),
            'Games On Sale': len(games_on_sale),
            'Average Discount': f"{(games_on_sale['discount_rate'].mean() * 100):.1f}%",
            'Average Rating of Sale Games': f"{games_on_sale['rating'].mean():.1f}/5.0",
            'Top Sale Candidate': sale_candidates_table.iloc[0]['title'],
            'Top Popular Game': popular_games_table.iloc[0]['title'],
            'Most Popular Genre': popular_games_table['genre'].value_counts().index[0],
            'Total Users with Recommendations': len(recommendations['user_id'].unique()) if recommendations is not None else 0
        }
        
        # Convert to DataFrame for easier saving
        summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
        summary_df.to_csv(f'{output_dir}/summary_stats.csv', index=False)
        logging.info(f"Saved summary statistics to {output_dir}/summary_stats.csv")
        
        return {
            'games_on_sale': games_on_sale,
            'sale_candidates': sale_candidates_table,
            'popular_games': popular_games_table,
            'summary': summary
        }
        
    except Exception as e:
        logging.error(f"Error generating dashboard tables: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_dashboard_visualizations(dashboard_data, output_dir='dashboard'):
    """
    Create visualizations for the dashboard
    """
    try:
        if dashboard_data is None:
            logging.error("No dashboard data to visualize")
            return
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 1. Games on sale by genre
        if 'games_on_sale' in dashboard_data:
            games_on_sale = dashboard_data['games_on_sale']
            plt.figure(figsize=(10, 6))
            sns.countplot(y='genre', data=games_on_sale, order=games_on_sale['genre'].value_counts().index)
            plt.title('Games on Sale by Genre')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/sales_by_genre.png', dpi=300)
            plt.close()
            
            # Discount distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(games_on_sale['discount_percentage'], bins=10, kde=True)
            plt.title('Distribution of Discount Percentages')
            plt.xlabel('Discount (%)')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/discount_distribution.png', dpi=300)
            plt.close()
            
            # Interactive Plotly visualizations
            # Games on sale scatter plot
            fig1 = px.scatter(games_on_sale, x='price', y='discount_percentage',
                             color='genre', size='popularity_score',
                             hover_data=['title', 'sale_price'],
                             title='Games on Sale: Price vs Discount by Genre')
            fig1.write_html(f'{output_dir}/interactive_sales.html')
            
            # Discount by genre boxplot
            fig2 = px.box(games_on_sale, x='genre', y='discount_percentage',
                         title='Discount Distribution by Genre')
            fig2.write_html(f'{output_dir}/interactive_discounts.html')
        
        # 2. Sale worthiness factors
        if 'sale_candidates' in dashboard_data:
            sale_candidates = dashboard_data['sale_candidates']
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='conversion_rate', y='popularity_score', size='price', 
                           hue='sale_worthiness', data=sale_candidates)
            plt.title('Sale Worthiness Factors')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/sale_worthiness.png', dpi=300)
            plt.close()
            
            # Interactive sale worthiness visualization
            fig3 = px.scatter(sale_candidates, x='conversion_rate', y='popularity_score',
                             size='price', color='sale_worthiness',
                             hover_data=['title', 'recommended_discount', 'recommended_sale_price'],
                             title='Sale Worthiness Analysis')
            fig3.write_html(f'{output_dir}/interactive_sale_worthiness.html')
            
            # Top sale candidates
            fig4 = px.bar(sale_candidates.head(15), x='sale_worthiness', y='title',
                         hover_data=['price', 'recommended_discount', 'recommended_sale_price'],
                         title='Top 15 Sale Candidates',
                         orientation='h')
            fig4.write_html(f'{output_dir}/interactive_top_candidates.html')
        
        # 3. Popular games distribution
        if 'popular_games' in dashboard_data:
            popular_games = dashboard_data['popular_games']
            plt.figure(figsize=(10, 6))
            sns.barplot(y='title', x='combined_popularity', data=popular_games.head(10))
            plt.title('Top 10 Most Popular Games')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/popular_games_top10.png', dpi=300)
            plt.close()
            
            # Genre distribution of popular games
            plt.figure(figsize=(10, 6))
            sns.countplot(y='genre', data=popular_games, order=popular_games['genre'].value_counts().index)
            plt.title('Popular Games by Genre')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/popular_genres.png', dpi=300)
            plt.close()
            
            # Interactive popular games visualization
            fig5 = px.bar(popular_games.head(15), x='combined_popularity', y='title',
                         color='genre', hover_data=['rating', 'price', 'release_date'],
                         title='Top 15 Most Popular Games',
                         orientation='h')
            fig5.write_html(f'{output_dir}/interactive_popular_games.html')
            
            # Popularity factors
            fig6 = make_subplots(rows=1, cols=2, 
                                subplot_titles=('Rating vs Popularity', 'Price vs Popularity'))
            
            fig6.add_trace(
                go.Scatter(x=popular_games['rating'], y=popular_games['combined_popularity'],
                          mode='markers', marker=dict(color=popular_games['price'], 
                                                     colorscale='Viridis', showscale=True),
                          text=popular_games['title']),
                row=1, col=1
            )
            
            fig6.add_trace(
                go.Scatter(x=popular_games['price'], y=popular_games['combined_popularity'],
                          mode='markers', marker=dict(color=popular_games['rating'], 
                                                     colorscale='Plasma', showscale=True),
                          text=popular_games['title']),
                row=1, col=2
            )
            
            fig6.update_layout(title_text='Factors Affecting Game Popularity', height=600)
            fig6.write_html(f'{output_dir}/interactive_popularity_factors.html')
        
        logging.info(f"Created dashboard visualizations in {output_dir}")
        
    except Exception as e:
        logging.error(f"Error creating dashboard visualizations: {e}")
        import traceback
        traceback.print_exc()

def display_interactive_dashboard(output_dir='dashboard'):
    """
    Display interactive visualizations in the browser
    """
    try:
        logging.info("Displaying interactive dashboard...")
        
        # Check if the output directory exists
        if not os.path.exists(output_dir):
            logging.error(f"Dashboard directory not found: {output_dir}")
            return
        
        # Find all HTML files in the output directory
        html_files = [f for f in os.listdir(output_dir) if f.endswith('.html')]
        
        if not html_files:
            logging.warning("No interactive visualizations found in the dashboard directory")
            return
        
        # Create a dashboard index file
        with open(f'{output_dir}/dashboard_index.html', 'w') as f:
            f.write('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Game Recommendation Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #333366; }
                    .dashboard-container { display: flex; flex-wrap: wrap; }
                    .dashboard-item { 
                        width: 100%; 
                        margin-bottom: 20px;
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        overflow: hidden;
                    }
                    .dashboard-item h2 {
                        background-color: #f0f0f0;
                        margin: 0;
                        padding: 10px;
                        font-size: 16px;
                    }
                    iframe {
                        width: 100%;
                        height: 500px;
                        border: none;
                    }
                </style>
            </head>
            <body>
                <h1>Game Recommendation Dashboard</h1>
                <div class="dashboard-container">
            ''')
            
            # Add each visualization to the dashboard
            for html_file in html_files:
                if html_file == 'dashboard_index.html':
                    continue
                    
                # Create a nice title from the filename
                title = html_file.replace('interactive_', '').replace('.html', '').replace('_', ' ').title()
                
                f.write(f'''
                <div class="dashboard-item">
                    <h2>{title}</h2>
                    <iframe src="{html_file}"></iframe>
                </div>
                ''')
            
            f.write('''
                </div>
            </body>
            </html>
            ''')
        
        # Open the dashboard in the default browser
        dashboard_path = os.path.abspath(f'{output_dir}/dashboard_index.html')
        webbrowser.open(f'file://{dashboard_path}')
        
        logging.info(f"Interactive dashboard opened in browser: {dashboard_path}")
        
    except Exception as e:
        logging.error(f"Error displaying interactive dashboard: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function to run the dashboard generation
    """
    try:
        logging.info("Starting dashboard generation...")
        
        # Load data
        game_data, recommendations, user_interactions = load_data()
        
        if game_data is None:
            logging.error("Failed to load game data. Exiting.")
            return
        
        # Generate dashboard data
        dashboard_data = generate_dashboard_tables()
        
        # Create visualizations
        create_dashboard_visualizations(dashboard_data)
        
        # Open the dashboard in the browser
        dashboard_path = os.path.join('dashboard', 'consolidated_dashboard.html')
        if os.path.exists(dashboard_path):
            webbrowser.open('file://' + os.path.abspath(dashboard_path))
            logging.info("Dashboard opened in browser.")
        else:
            dashboard_path = os.path.join('dashboard', 'dashboard_index.html')
            if os.path.exists(dashboard_path):
                webbrowser.open('file://' + os.path.abspath(dashboard_path))
                logging.info("Dashboard opened in browser.")
        
        logging.info("Dashboard generation complete.")
        
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()