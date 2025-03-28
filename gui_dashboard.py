import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pickle
import threading
import webbrowser
from matplotlib.figure import Figure
from matplotlib.cm import get_cmap

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GameRecommendationDashboard:
    """
    GUI Dashboard for Game Recommendation System
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Game Recommendation Dashboard")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Set application icon if available
        try:
            self.root.iconbitmap("assets/icon.ico")
        except:
            pass
        
        # Initialize data containers
        self.game_data = None
        self.recommendations = None
        self.user_interactions = None
        self.games_on_sale = None
        self.sale_candidates = None
        self.popular_games = None
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create header
        self.create_header()
        
        # Create tab control
        self.tab_control = ttk.Notebook(self.main_frame)
        self.tab_control.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.summary_tab = ttk.Frame(self.tab_control)
        self.popular_games_tab = ttk.Frame(self.tab_control)
        self.sales_tab = ttk.Frame(self.tab_control)
        self.genres_tab = ttk.Frame(self.tab_control)
        self.recommendations_tab = ttk.Frame(self.tab_control)
        
        # Add tabs to notebook
        self.tab_control.add(self.summary_tab, text="Summary")
        self.tab_control.add(self.popular_games_tab, text="Popular Games")
        self.tab_control.add(self.sales_tab, text="Sales Analysis")
        self.tab_control.add(self.genres_tab, text="Genre Distribution")
        self.tab_control.add(self.recommendations_tab, text="Recommendations")
        
        # Create footer
        self.create_footer()
        
        # Load data in background
        self.show_loading_message()
        threading.Thread(target=self.load_all_data, daemon=True).start()
    
    def create_menu_bar(self):
        """Create menu bar with options"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Create File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Reload Data", command=self.load_all_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Create View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Open HTML Dashboard", command=self.open_html_dashboard)
        view_menu.add_command(label="Switch to GUI Dashboard", command=self.switch_to_gui)
        
        # Create Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_header(self):
        """Create dashboard header"""
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(
            header_frame, 
            text="Game Recommendation Dashboard", 
            font=("Segoe UI", 16, "bold")
        )
        title_label.pack(pady=5)
        
        subtitle_label = ttk.Label(
            header_frame,
            text="Interactive visualization of game recommendations and analytics",
            font=("Segoe UI", 10)
        )
        subtitle_label.pack(pady=2)
        
        # Add separator
        separator = ttk.Separator(header_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=5)
    
    def create_footer(self):
        """Create dashboard footer"""
        footer_frame = ttk.Frame(self.main_frame)
        footer_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Add separator
        separator = ttk.Separator(footer_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=5)
        
        # Add footer text
        footer_label = ttk.Label(
            footer_frame,
            text=" 2025 Game Recommendation System",
            font=("Segoe UI", 8)
        )
        footer_label.pack(pady=5)
    
    def show_loading_message(self):
        """Show loading message while data is being loaded"""
        for tab in [self.summary_tab, self.popular_games_tab, self.sales_tab, 
                   self.genres_tab, self.recommendations_tab]:
            loading_label = ttk.Label(
                tab,
                text="Loading data, please wait...",
                font=("Segoe UI", 12)
            )
            loading_label.pack(expand=True)
    
    def load_all_data(self):
        """Load all necessary data for the dashboard"""
        try:
            # Load data from recommendation_dashboard.py functions
            from recommendation_dashboard import load_data, simulate_sales_data, identify_sale_candidates, get_popular_games
            
            # Load base data
            self.game_data, self.recommendations, self.user_interactions = load_data()
            
            if self.game_data is None:
                self.show_error("Failed to load game data. Please check the logs.")
                return
            
            # Generate sales data
            self.games_on_sale = simulate_sales_data(self.game_data)
            
            # Identify sale candidates
            self.sale_candidates = identify_sale_candidates(self.game_data, self.user_interactions)
            
            # Get popular games
            self.popular_games = get_popular_games(self.game_data, self.user_interactions)
            
            # Check if recommendations are empty and create sample data if needed
            if self.recommendations is None or len(self.recommendations) == 0:
                logging.info("No recommendations found. Creating sample recommendation data.")
                self.create_sample_recommendations()
            
            # Update UI with loaded data
            self.root.after(0, self.update_ui_with_data)
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            self.show_error(f"Error loading data: {str(e)}")
    
    def create_sample_recommendations(self):
        """Create sample recommendation data if real data is not available"""
        # Create sample user IDs
        user_ids = [68, 178, 369, 434, 441]
        
        # Create sample game data if not available
        if self.game_data is None or len(self.game_data) == 0:
            logging.warning("Game data not available for sample recommendations. Creating minimal game data.")
            self.game_data = pd.DataFrame({
                'game_id': range(1, 21),
                'title': [
                    'Call of Duty', 'FIFA 2025', 'Minecraft', 'Forza Horizon', 'Battlefield 2042',
                    'Grand Theft Auto', 'Halo Infinite', 'Epic Champions', 'Super Champions', 'Legendary Adventure',
                    'Assassin\'s Creed', 'Red Dead Redemption', 'The Witcher', 'Cyberpunk 2077', 'Destiny 2',
                    'Elder Scrolls', 'Fallout', 'Mass Effect', 'Far Cry', 'Rainbow Six'
                ],
                'genre': ['Action', 'Sports', 'Adventure', 'Racing', 'Action', 
                         'Action', 'Shooter', 'Sports', 'Sports', 'RPG',
                         'Action', 'Action', 'RPG', 'RPG', 'Shooter',
                         'RPG', 'RPG', 'RPG', 'Action', 'Shooter'],
                'rating': [4.5, 4.2, 4.8, 4.3, 3.9, 4.7, 4.1, 3.8, 3.7, 4.0,
                          4.6, 4.7, 4.9, 3.5, 4.2, 4.5, 4.3, 4.4, 4.0, 4.1],
                'price': [59.99, 49.99, 29.99, 39.99, 59.99, 
                         59.99, 49.99, 19.99, 24.99, 39.99,
                         49.99, 39.99, 29.99, 49.99, 39.99,
                         59.99, 39.99, 29.99, 49.99, 39.99],
                'popularity_score': [9.5, 9.2, 9.8, 8.7, 8.3, 
                                   9.7, 8.9, 7.5, 7.2, 8.1,
                                   9.3, 9.4, 9.6, 8.2, 8.8,
                                   9.1, 8.9, 8.7, 8.5, 8.4]
            })
        
        # Create sample recommendations
        sample_recs = []
        
        for user_id in user_ids:
            # Get 10 random games for each user
            game_indices = np.random.choice(len(self.game_data), 10, replace=False)
            games = self.game_data.iloc[game_indices]
            
            # Create recommendations with random interest scores
            for _, game in games.iterrows():
                sample_recs.append({
                    'user_id': user_id,
                    'game_id': game['game_id'],
                    'predicted_interest': np.random.uniform(0.7, 0.95)
                })
        
        # Create DataFrame
        self.recommendations = pd.DataFrame(sample_recs)
        logging.info(f"Created {len(self.recommendations)} sample recommendations for {len(user_ids)} users")
    
    def update_ui_with_data(self):
        """Update UI with loaded data"""
        # Clear loading messages
        for tab in [self.summary_tab, self.popular_games_tab, self.sales_tab, 
                   self.genres_tab, self.recommendations_tab]:
            for widget in tab.winfo_children():
                widget.destroy()
        
        # Initialize tabs with data
        self.initialize_summary_tab()
        self.initialize_popular_games_tab()
        self.initialize_sales_tab()
        self.initialize_genres_tab()
        self.initialize_recommendations_tab()
    
    def show_error(self, message):
        """Show error message"""
        messagebox.showerror("Error", message)
    
    def initialize_summary_tab(self):
        """Initialize summary tab with data"""
        # Create a frame for summary statistics
        stats_frame = ttk.Frame(self.summary_tab)
        stats_frame.pack(fill=tk.X, pady=10)
        
        # Calculate summary statistics
        total_games = len(self.game_data) if self.game_data is not None else 0
        games_on_sale = sum(self.games_on_sale['on_sale']) if self.games_on_sale is not None else 0
        avg_discount = self.games_on_sale.loc[self.games_on_sale['on_sale'], 'discount_rate'].mean() * 100 if self.games_on_sale is not None else 0
        
        # Create stat cards
        self.create_stat_card(stats_frame, "Total Games", str(total_games), 0)
        self.create_stat_card(stats_frame, "Games on Sale", str(games_on_sale), 1)
        self.create_stat_card(stats_frame, "Average Discount", f"{avg_discount:.1f}%", 2)
        self.create_stat_card(stats_frame, "Top Genre", self.get_top_genre(), 3)
        
        # Add a frame for charts
        charts_frame = ttk.Frame(self.summary_tab)
        charts_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create a notebook for summary charts
        charts_notebook = ttk.Notebook(charts_frame)
        charts_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create frames for each chart
        price_chart_frame = ttk.Frame(charts_notebook)
        genre_chart_frame = ttk.Frame(charts_notebook)
        discount_chart_frame = ttk.Frame(charts_notebook)
        
        charts_notebook.add(price_chart_frame, text="Price Distribution")
        charts_notebook.add(genre_chart_frame, text="Genre Distribution")
        charts_notebook.add(discount_chart_frame, text="Discount Analysis")
        
        # Create price distribution chart
        self.create_price_distribution_chart(price_chart_frame)
        
        # Create genre distribution chart
        self.create_genre_distribution_chart(genre_chart_frame)
        
        # Create discount analysis chart
        self.create_discount_analysis_chart(discount_chart_frame)
    
    def create_stat_card(self, parent, label, value, column):
        """Create a statistics card"""
        card_frame = ttk.Frame(parent, borderwidth=1, relief="solid")
        card_frame.grid(row=0, column=column, padx=10, pady=10, sticky="nsew")
        
        # Configure grid weights
        parent.columnconfigure(column, weight=1)
        
        # Add label
        label_widget = ttk.Label(
            card_frame,
            text=label,
            font=("Segoe UI", 10)
        )
        label_widget.pack(pady=(10, 5))
        
        # Add value
        value_widget = ttk.Label(
            card_frame,
            text=value,
            font=("Segoe UI", 14, "bold")
        )
        value_widget.pack(pady=(5, 10))
    
    def get_top_genre(self):
        """Get the top genre from game data"""
        if self.game_data is not None and 'genre' in self.game_data.columns:
            genre_counts = self.game_data['genre'].value_counts()
            if not genre_counts.empty:
                return genre_counts.index[0]
        return "Unknown"
    
    def initialize_popular_games_tab(self):
        """Initialize popular games tab with data"""
        if self.popular_games is None:
            self.show_placeholder(self.popular_games_tab, "Popular games data not available")
            return
        
        # Create main frame
        main_frame = ttk.Frame(self.popular_games_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create top frame for chart
        chart_frame = ttk.Frame(main_frame)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create bottom frame for table
        table_frame = ttk.Frame(main_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create popular games chart
        self.create_popular_games_chart(chart_frame)
        
        # Create popular games table
        self.create_popular_games_table(table_frame)
    
    def create_popular_games_chart(self, parent):
        """Create chart showing top popular games"""
        if self.popular_games is None:
            return
        
        # Get top 10 popular games
        top_games = self.popular_games.head(10).copy()
        
        # Create figure
        fig = Figure(figsize=(10, 5), dpi=100)
        ax = fig.add_subplot(111)
        
        # Sort by popularity score
        top_games = top_games.sort_values('popularity_score', ascending=False)
        
        # Create bar chart with color gradient
        titles = top_games['title']
        scores = top_games['popularity_score']
        
        # Create color gradient similar to Plotly's Plasma colorscale
        plasma_cmap = get_cmap('plasma')
        colors = [plasma_cmap(i/len(scores)) for i in range(len(scores))]
        
        # Create vertical bar chart (to match HTML version)
        bars = ax.bar(titles, scores, color=colors)
        
        # Set titles and labels
        ax.set_title('Top 10 Most Popular Games')
        ax.set_xlabel('Game')
        ax.set_ylabel('Popularity Score')
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
        
        # Adjust layout
        fig.tight_layout()
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_popular_games_table(self, parent):
        """Create table showing popular games details"""
        if self.popular_games is None:
            return
        
        # Get top 20 popular games
        top_games = self.popular_games.head(20).copy()
        
        # Create frame for table
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(table_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create treeview
        columns = ['title', 'genre', 'rating', 'price', 'popularity_score']
        column_headings = ['Title', 'Genre', 'Rating', 'Price ($)', 'Popularity']
        
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', yscrollcommand=scrollbar.set)
        
        # Configure scrollbar
        scrollbar.config(command=tree.yview)
        
        # Set column headings
        for col, heading in zip(columns, column_headings):
            tree.heading(col, text=heading)
            
            # Set column widths
            if col == 'title':
                tree.column(col, width=250, minwidth=200)
            elif col == 'genre':
                tree.column(col, width=150, minwidth=100)
            else:
                tree.column(col, width=100, minwidth=80, anchor=tk.CENTER)
        
        # Insert data
        for i, row in top_games.iterrows():
            values = [
                row['title'],
                row['genre'],
                f"{row['rating']:.1f}",
                f"{row['price']:.2f}",
                f"{row['popularity_score']:.2f}"
            ]
            tree.insert('', tk.END, values=values)
        
        # Pack treeview
        tree.pack(fill=tk.BOTH, expand=True)
    
    def initialize_sales_tab(self):
        """Initialize sales tab with data"""
        if self.games_on_sale is None or self.sale_candidates is None:
            self.show_placeholder(self.sales_tab, "Sales data not available")
            return
        
        # Create notebook for sales tabs
        sales_notebook = ttk.Notebook(self.sales_tab)
        sales_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create frames for each sales analysis
        current_sales_frame = ttk.Frame(sales_notebook)
        sale_candidates_frame = ttk.Frame(sales_notebook)
        discount_analysis_frame = ttk.Frame(sales_notebook)
        
        # Add frames to notebook
        sales_notebook.add(current_sales_frame, text="Current Sales")
        sales_notebook.add(sale_candidates_frame, text="Sale Candidates")
        sales_notebook.add(discount_analysis_frame, text="Discount Analysis")
        
        # Populate current sales tab
        self.create_current_sales_view(current_sales_frame)
        
        # Populate sale candidates tab
        self.create_sale_candidates_view(sale_candidates_frame)
        
        # Populate discount analysis tab
        self.create_discount_analysis_view(discount_analysis_frame)
    
    def create_current_sales_view(self, parent):
        """Create view for current sales"""
        if self.games_on_sale is None:
            return
        
        # Filter games on sale
        on_sale = self.games_on_sale[self.games_on_sale['on_sale']].copy()
        
        if len(on_sale) == 0:
            self.show_placeholder(parent, "No games currently on sale")
            return
        
        # Create frame for chart and table
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create top frame for chart
        chart_frame = ttk.Frame(main_frame)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create bottom frame for table
        table_frame = ttk.Frame(main_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create chart
        fig = Figure(figsize=(10, 5), dpi=100)
        ax = fig.add_subplot(111)
        
        # Sort by discount rate and take top 10
        on_sale['discount_rate_pct'] = on_sale['discount_rate'] * 100
        top_discounts = on_sale.sort_values('discount_rate', ascending=False).head(10)
        
        # Create bar chart with color gradient
        titles = top_discounts['title']
        discounts = top_discounts['discount_rate_pct']
        
        # Create color gradient similar to Plotly's Viridis colorscale
        viridis_cmap = get_cmap('viridis')
        colors = [viridis_cmap(i/len(discounts)) for i in range(len(discounts))]
        
        # Create vertical bar chart (to match HTML version)
        bars = ax.bar(titles, discounts, color=colors)
        
        # Set titles and labels
        ax.set_title('Top 10 Games with Highest Discounts')
        ax.set_xlabel('Game')
        ax.set_ylabel('Discount %')
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
        
        # Adjust layout
        fig.tight_layout()
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create table
        # Create scrollbar
        scrollbar = ttk.Scrollbar(table_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create treeview
        columns = ['title', 'genre', 'price', 'discount_rate', 'sale_price', 'sale_end_date']
        column_headings = ['Title', 'Genre', 'Original Price ($)', 'Discount (%)', 'Sale Price ($)', 'Sale Ends']
        
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', yscrollcommand=scrollbar.set)
        
        # Configure scrollbar
        scrollbar.config(command=tree.yview)
        
        # Set column headings
        for col, heading in zip(columns, column_headings):
            tree.heading(col, text=heading)
            
            # Set column widths
            if col == 'title':
                tree.column(col, width=250, minwidth=200)
            elif col == 'genre':
                tree.column(col, width=150, minwidth=100)
            elif col == 'sale_end_date':
                tree.column(col, width=150, minwidth=100, anchor=tk.CENTER)
            else:
                tree.column(col, width=100, minwidth=80, anchor=tk.CENTER)
        
        # Insert data
        for i, row in on_sale.iterrows():
            values = [
                row['title'],
                row['genre'],
                f"{row['price']:.2f}",
                f"{row['discount_rate']*100:.1f}%",
                f"{row['sale_price']:.2f}",
                row['sale_end_date'] if pd.notna(row['sale_end_date']) else "N/A"
            ]
            tree.insert('', tk.END, values=values)
        
        # Pack treeview
        tree.pack(fill=tk.BOTH, expand=True)
    
    def create_sale_candidates_view(self, parent):
        """Create view for sale candidates"""
        if self.sale_candidates is None:
            return
        
        if len(self.sale_candidates) == 0:
            self.show_placeholder(parent, "No sale candidates available")
            return
        
        # Create frame for chart and table
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create top frame for chart
        chart_frame = ttk.Frame(main_frame)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create bottom frame for table
        table_frame = ttk.Frame(main_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create chart
        fig = Figure(figsize=(10, 5), dpi=100)
        ax = fig.add_subplot(111)
        
        # Get top candidates and sort by sale worthiness
        top_candidates = self.sale_candidates.head(10).copy()
        top_candidates = top_candidates.sort_values('sale_worthiness', ascending=False)
        
        # Scale sale worthiness to percentage for consistency with HTML
        top_candidates['sale_worthiness_pct'] = top_candidates['sale_worthiness'] * 100
        
        # Create bar chart with color gradient
        titles = top_candidates['title']
        worthiness = top_candidates['sale_worthiness_pct']
        
        # Create color gradient similar to Plotly's Turbo colorscale
        turbo_cmap = get_cmap('turbo')
        colors = [turbo_cmap(i/len(worthiness)) for i in range(len(worthiness))]
        
        # Create vertical bar chart (to match HTML version)
        bars = ax.bar(titles, worthiness, color=colors)
        
        # Set titles and labels
        ax.set_title('Top 10 Games Recommended for Sale')
        ax.set_xlabel('Game')
        ax.set_ylabel('Sale Worthiness Score')
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
        
        # Adjust layout
        fig.tight_layout()
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create table
        # Create scrollbar
        scrollbar = ttk.Scrollbar(table_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create treeview
        columns = ['title', 'genre', 'price', 'recommended_discount', 'recommended_sale_price', 'sale_worthiness']
        column_headings = ['Title', 'Genre', 'Current Price ($)', 'Rec. Discount (%)', 'Rec. Sale Price ($)', 'Worthiness']
        
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', yscrollcommand=scrollbar.set)
        
        # Configure scrollbar
        scrollbar.config(command=tree.yview)
        
        # Set column headings
        for col, heading in zip(columns, column_headings):
            tree.heading(col, text=heading)
            
            # Set column widths
            if col == 'title':
                tree.column(col, width=250, minwidth=200)
            elif col == 'genre':
                tree.column(col, width=150, minwidth=100)
            else:
                tree.column(col, width=120, minwidth=80, anchor=tk.CENTER)
        
        # Insert data
        for i, row in self.sale_candidates.head(20).iterrows():
            values = [
                row['title'],
                row['genre'],
                f"{row['price']:.2f}",
                f"{row['recommended_discount']*100:.1f}%",
                f"{row['recommended_sale_price']:.2f}",
                f"{row['sale_worthiness']:.3f}"
            ]
            tree.insert('', tk.END, values=values)
        
        # Pack treeview
        tree.pack(fill=tk.BOTH, expand=True)
    
    def create_discount_analysis_view(self, parent):
        """Create view for discount analysis"""
        if self.games_on_sale is None:
            return
        
        # Create frame
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create chart
        fig = plt.Figure(figsize=(10, 8), dpi=100)
        
        # Create subplots
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Filter games on sale
        on_sale = self.games_on_sale[self.games_on_sale['on_sale']]
        
        if len(on_sale) == 0:
            self.show_placeholder(parent, "No games currently on sale")
            return
        
        # Create discount distribution histogram
        sns.histplot(on_sale['discount_rate'] * 100, bins=20, kde=True, ax=ax1)
        ax1.set_title('Discount Rate Distribution')
        ax1.set_xlabel('Discount Rate (%)')
        ax1.set_ylabel('Number of Games')
        
        # Create genre vs discount boxplot if genre column exists
        if 'genre' in on_sale.columns:
            # Get top 8 genres by count
            top_genres = on_sale['genre'].value_counts().head(8).index.tolist()
            genre_data = on_sale[on_sale['genre'].isin(top_genres)]
            
            # Create boxplot
            sns.boxplot(x='genre', y='discount_rate', data=genre_data, ax=ax2, palette='Set3')
            ax2.set_title('Discount Rates by Genre')
            ax2.set_xlabel('Genre')
            ax2.set_ylabel('Discount Rate')
            ax2.set_ylim(0, on_sale['discount_rate'].max() * 1.1)
            ax2.tick_params(axis='x', rotation=45)
            
            # Convert y-axis to percentage
            ax2.set_yticklabels([f'{x*100:.0f}%' for x in ax2.get_yticks()])
        
        # Adjust layout
        fig.tight_layout()
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_price_distribution_chart(self, parent):
        """Create price distribution chart"""
        if self.game_data is None or 'price' not in self.game_data.columns:
            self.show_placeholder(parent, "Price data not available")
            return
        
        # Create figure
        fig = plt.Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Create histogram
        sns.histplot(self.game_data['price'], bins=20, kde=True, ax=ax)
        
        # Set titles and labels
        ax.set_title('Game Price Distribution')
        ax.set_xlabel('Price ($)')
        ax.set_ylabel('Number of Games')
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_genre_distribution_chart(self, parent):
        """Create genre distribution chart"""
        if self.game_data is None or 'genre' not in self.game_data.columns:
            self.show_placeholder(parent, "Genre data not available")
            return
        
        # Get genre counts
        genre_counts = self.game_data['genre'].value_counts().head(10)
        
        # Create figure
        fig = plt.Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Create bar chart
        genre_counts.plot(kind='bar', ax=ax)
        
        # Set titles and labels
        ax.set_title('Top 10 Game Genres')
        ax.set_xlabel('Genre')
        ax.set_ylabel('Number of Games')
        ax.tick_params(axis='x', rotation=45)
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_discount_analysis_chart(self, parent):
        """Create discount analysis chart"""
        if self.games_on_sale is None or 'discount_rate' not in self.games_on_sale.columns:
            self.show_placeholder(parent, "Discount data not available")
            return
        
        # Filter games on sale
        on_sale = self.games_on_sale[self.games_on_sale['on_sale']]
        
        if len(on_sale) == 0:
            self.show_placeholder(parent, "No games currently on sale")
            return
        
        # Create figure
        fig = plt.Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Create scatter plot
        ax.scatter(on_sale['price'], on_sale['discount_rate'] * 100, alpha=0.6)
        
        # Set titles and labels
        ax.set_title('Discount Rate vs Original Price')
        ax.set_xlabel('Original Price ($)')
        ax.set_ylabel('Discount Rate (%)')
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_placeholder(self, parent, message):
        """Show placeholder message when data is not available"""
        placeholder_label = ttk.Label(
            parent,
            text=message,
            font=("Segoe UI", 12)
        )
        placeholder_label.pack(expand=True)

    def initialize_genres_tab(self):
        """Initialize genres tab with data"""
        if self.game_data is None or 'genre' not in self.game_data.columns:
            self.show_placeholder(self.genres_tab, "Genre data not available")
            return
        
        # Create main frame
        main_frame = ttk.Frame(self.genres_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for genre analysis
        genre_notebook = ttk.Notebook(main_frame)
        genre_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create frames for each analysis
        distribution_frame = ttk.Frame(genre_notebook)
        price_analysis_frame = ttk.Frame(genre_notebook)
        popularity_frame = ttk.Frame(genre_notebook)
        
        # Add frames to notebook
        genre_notebook.add(distribution_frame, text="Genre Distribution")
        genre_notebook.add(price_analysis_frame, text="Price by Genre")
        genre_notebook.add(popularity_frame, text="Popularity by Genre")
        
        # Create genre distribution visualization
        self.create_genre_distribution_visualization(distribution_frame)
        
        # Create price by genre visualization
        self.create_price_by_genre_visualization(price_analysis_frame)
        
        # Create popularity by genre visualization
        self.create_popularity_by_genre_visualization(popularity_frame)
    
    def create_genre_distribution_visualization(self, parent):
        """Create genre distribution visualization"""
        if self.game_data is None or 'genre' not in self.game_data.columns:
            return
        
        # Get genre counts
        genre_counts = self.game_data['genre'].value_counts().head(10)
        
        # Create figure
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Define colors similar to HTML version
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', 
                 '#1abc9c', '#34495e', '#d35400', '#7f8c8d', '#2980b9']
        
        # Create pie chart with a hole in the center (donut chart)
        wedges, texts, autotexts = ax.pie(
            genre_counts, 
            labels=genre_counts.index, 
            autopct='%1.1f%%',
            textprops={'fontsize': 9},
            startangle=90,
            colors=colors,
            wedgeprops=dict(width=0.5)  # Creates a donut chart with a hole
        )
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        ax.set_title('Game Distribution by Genre')
        
        # Create legend
        ax.legend(
            wedges, 
            genre_counts.index,
            title="Genres",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
        
        # Adjust layout
        fig.tight_layout()
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_price_by_genre_visualization(self, parent):
        """Create price by genre visualization"""
        if self.game_data is None or 'genre' not in self.game_data.columns or 'price' not in self.game_data.columns:
            return
        
        # Get top genres by count
        top_genres = self.game_data['genre'].value_counts().head(10).index.tolist()
        genre_data = self.game_data[self.game_data['genre'].isin(top_genres)]
        
        # Create figure
        fig = plt.Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Create boxplot
        sns.boxplot(x='genre', y='price', data=genre_data, ax=ax, palette='viridis')
        
        # Set titles and labels
        ax.set_title('Price Distribution by Genre')
        ax.set_xlabel('Genre')
        ax.set_ylabel('Price ($)')
        ax.tick_params(axis='x', rotation=45)
        
        # Adjust layout
        fig.tight_layout()
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_popularity_by_genre_visualization(self, parent):
        """Create popularity by genre visualization"""
        if self.game_data is None or 'genre' not in self.game_data.columns or 'popularity_score' not in self.game_data.columns:
            return
        
        # Get top genres by count
        top_genres = self.game_data['genre'].value_counts().head(10).index.tolist()
        genre_data = self.game_data[self.game_data['genre'].isin(top_genres)]
        
        # Create figure
        fig = plt.Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Calculate average popularity by genre
        genre_popularity = genre_data.groupby('genre')['popularity_score'].mean().sort_values(ascending=False)
        
        # Create bar chart
        bars = ax.bar(genre_popularity.index, genre_popularity.values, color='purple')
        
        # Set titles and labels
        ax.set_title('Average Popularity Score by Genre')
        ax.set_xlabel('Genre')
        ax.set_ylabel('Average Popularity Score')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Adjust layout
        fig.tight_layout()
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def initialize_recommendations_tab(self):
        """Initialize recommendations tab with data"""
        if self.recommendations is None:
            self.show_placeholder(self.recommendations_tab, "Recommendation data not available")
            return
        
        # Create main frame
        main_frame = ttk.Frame(self.recommendations_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create user selection frame
        selection_frame = ttk.Frame(main_frame)
        selection_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create label
        user_label = ttk.Label(selection_frame, text="Select User ID:", font=("Segoe UI", 10))
        user_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Get unique user IDs
        user_ids = sorted(self.recommendations['user_id'].unique())
        
        # Create combobox for user selection
        self.user_id_var = tk.StringVar()
        user_combobox = ttk.Combobox(selection_frame, textvariable=self.user_id_var, values=user_ids, state="readonly")
        user_combobox.pack(side=tk.LEFT)
        
        # Set default value
        if len(user_ids) > 0:
            user_combobox.current(0)
        
        # Create button to load recommendations
        load_button = ttk.Button(selection_frame, text="Load Recommendations", command=self.load_user_recommendations)
        load_button.pack(side=tk.LEFT, padx=10)
        
        # Create frame for recommendations
        self.recommendations_frame = ttk.Frame(main_frame)
        self.recommendations_frame.pack(fill=tk.BOTH, expand=True)
        
        # Load initial recommendations if user IDs exist
        if len(user_ids) > 0:
            self.load_user_recommendations()
    
    def load_user_recommendations(self):
        """Load recommendations for selected user"""
        # Clear previous recommendations
        for widget in self.recommendations_frame.winfo_children():
            widget.destroy()
        
        # Get selected user ID
        user_id = self.user_id_var.get()
        
        if not user_id:
            self.show_placeholder(self.recommendations_frame, "Please select a user ID")
            return
        
        # Filter recommendations for selected user
        user_recs = self.recommendations[self.recommendations['user_id'] == int(user_id)]
        
        if len(user_recs) == 0:
            self.show_placeholder(self.recommendations_frame, f"No recommendations found for user {user_id}")
            return
        
        # Create notebook for recommendation views
        recs_notebook = ttk.Notebook(self.recommendations_frame)
        recs_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create frames for different views
        chart_frame = ttk.Frame(recs_notebook)
        table_frame = ttk.Frame(recs_notebook)
        
        # Add frames to notebook
        recs_notebook.add(chart_frame, text="Visualization")
        recs_notebook.add(table_frame, text="Detailed List")
        
        # Create visualization
        self.create_recommendations_visualization(chart_frame, user_recs)
        
        # Create table
        self.create_recommendations_table(table_frame, user_recs)
    
    def create_recommendations_visualization(self, parent, user_recs):
        """Create visualization for user recommendations"""
        if len(user_recs) == 0:
            return
        
        # Check if title is already in user_recs
        if 'title' not in user_recs.columns:
            # Merge with game data to get additional information
            if self.game_data is not None:
                merged_recs = user_recs.merge(
                    self.game_data[['game_id', 'title', 'genre', 'rating', 'price']],
                    left_on='game_id',
                    right_on='game_id',
                    how='left'
                )
            else:
                # If game_data is not available, use game_id as title
                merged_recs = user_recs.copy()
                merged_recs['title'] = merged_recs['game_id'].apply(lambda x: f"Game {x}")
        else:
            # Title is already in the recommendations
            merged_recs = user_recs
        
        # Sort by predicted interest
        sorted_recs = merged_recs.sort_values('predicted_interest', ascending=False)
        
        # Get the actual number of recommendations available
        num_recs = len(sorted_recs)
        max_to_show = min(5, num_recs)
        
        # Get top recommendations to display
        display_recs = sorted_recs.head(max_to_show)
        
        # Scale predicted interest to percentage for consistency with HTML
        display_recs['predicted_interest_pct'] = display_recs['predicted_interest'] * 100
        
        # Create figure
        fig = plt.figure(figsize=(12, 10), dpi=100)
        ax = fig.add_subplot(111)
        
        # Create a vibrant color palette
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        
        # Create bubble chart with equal-sized bubbles
        bubble_size = 30000  # You can change this value to adjust bubble size
        
        # Calculate bubble radius in data coordinates
        # This is an approximation as the exact conversion depends on the figure size and DPI
        bubble_radius_approx = np.sqrt(bubble_size / np.pi) / 100
        min_distance = bubble_radius_approx * 2.2  # Ensure bubbles don't overlap
        
        # Generate non-overlapping positions
        x_positions = []
        y_positions = []
        
        # Define the area where bubbles can be placed
        # Add padding to prevent bubbles from being cut off at the edges
        padding = bubble_radius_approx * 1.5  # Increased padding factor for larger bubbles
        x_min, x_max = 0.0 + padding, 1.0 - padding
        y_min, y_max = 0.0 + padding, 1.0 - padding
        
        # Place bubbles using a simple algorithm to avoid overlap
        for i in range(max_to_show):
            # Try to find a position that doesn't overlap with existing bubbles
            max_attempts = 100
            for attempt in range(max_attempts):
                # Generate a random position
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                
                # Check if this position overlaps with any existing bubble
                overlap = False
                for j in range(len(x_positions)):
                    distance = np.sqrt((x - x_positions[j])**2 + (y - y_positions[j])**2)
                    if distance < min_distance:
                        overlap = True
                        break
                
                if not overlap:
                    # Found a good position
                    x_positions.append(x)
                    y_positions.append(y)
                    break
            
            # If we couldn't find a non-overlapping position, use a grid-based approach
            if attempt == max_attempts - 1:
                # Calculate grid positions
                cols = int(np.ceil(np.sqrt(max_to_show)))
                rows = int(np.ceil(max_to_show / cols))
                
                # Clear previous positions and use grid
                x_positions = []
                y_positions = []
                
                for idx in range(max_to_show):
                    row = idx // cols
                    col = idx % cols
                    
                    # Calculate grid position
                    x = x_min + (x_max - x_min) * col / (cols - 1 if cols > 1 else 1)
                    y = y_min + (y_max - y_min) * row / (rows - 1 if rows > 1 else 1)
                    
                    x_positions.append(x)
                    y_positions.append(y)
                
                break
        
        # Plot bubbles
        for i in range(max_to_show):
            ax.scatter(x_positions[i], y_positions[i], s=bubble_size, 
                      color=colors[i], alpha=0.7, edgecolors='white', linewidth=2)
            
            # Add game title in the center of each bubble
            ax.text(x_positions[i], y_positions[i], display_recs.iloc[i]['title'], 
                   ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        
        # Remove axes and spines for a cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Set title with increased padding to prevent overlap with bubbles
        ax.set_title(f'Top {max_to_show} Recommendations for User {user_recs.iloc[0]["user_id"]}', 
                    fontsize=14, pad=40)  # Increased pad value from 20 to 40
        
        # Add subtle grid for visual reference
        ax.grid(True, linestyle='--', alpha=0.2)
        
        # Set axis limits with some padding
        extra_margin = padding * 0.8  # Increased margin factor
        top_margin = padding * 1.2  # Extra margin at the top for the title
        ax.set_xlim(-extra_margin, 1 + extra_margin)
        ax.set_ylim(-extra_margin, 1 + top_margin)  # More space at the top
        
        # Adjust layout
        fig.tight_layout()
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # If fewer than 5 recommendations, add a note
        if num_recs < 5:
            note_frame = ttk.Frame(parent)
            note_frame.pack(fill=tk.X, pady=(5, 0))
            
            note_label = ttk.Label(
                note_frame,
                text=f"Note: Only {num_recs} recommendations available for this user.",
                font=("Segoe UI", 9, "italic"),
                foreground="#555555"
            )
            note_label.pack(pady=5)
    
    def create_recommendations_table(self, parent, user_recs):
        """Create table for user recommendations"""
        if len(user_recs) == 0:
            return
        
        # Check if title is already in user_recs
        if 'title' not in user_recs.columns:
            # Merge with game data to get additional information
            if self.game_data is not None:
                merged_recs = user_recs.merge(
                    self.game_data[['game_id', 'title', 'genre', 'rating', 'price', 'popularity_score']],
                    left_on='game_id',
                    right_on='game_id',
                    how='left'
                )
            else:
                # If game_data is not available, use game_id as title
                merged_recs = user_recs.copy()
                merged_recs['title'] = merged_recs['game_id'].apply(lambda x: f"Game {x}")
                merged_recs['genre'] = "Unknown"
                merged_recs['rating'] = np.nan
                merged_recs['price'] = np.nan
                merged_recs['popularity_score'] = np.nan
        else:
            # Title is already in the recommendations
            merged_recs = user_recs
        
        # Sort by predicted interest
        sorted_recs = merged_recs.sort_values('predicted_interest', ascending=False)
        
        # Get number of recommendations
        num_recs = len(sorted_recs)
        
        # Create table frame
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(table_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create treeview
        columns = ['title', 'genre', 'rating', 'price', 'popularity_score', 'predicted_interest']
        column_headings = ['Title', 'Genre', 'Rating', 'Price ($)', 'Popularity', 'Predicted Interest']
        
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', yscrollcommand=scrollbar.set)
        
        # Configure scrollbar
        scrollbar.config(command=tree.yview)
        
        # Set column headings
        for col, heading in zip(columns, column_headings):
            tree.heading(col, text=heading)
            
            # Set column widths
            if col == 'title':
                tree.column(col, width=250, minwidth=200)
            elif col == 'genre':
                tree.column(col, width=150, minwidth=100)
            else:
                tree.column(col, width=100, minwidth=80, anchor=tk.CENTER)
        
        # Insert data
        for i, row in sorted_recs.iterrows():
            values = [
                row['title'] if 'title' in row else f"Game {row['game_id']}",
                row['genre'] if 'genre' in row and pd.notna(row['genre']) else "Unknown",
                f"{row['rating']:.1f}" if 'rating' in row and pd.notna(row['rating']) else "N/A",
                f"{row['price']:.2f}" if 'price' in row and pd.notna(row['price']) else "N/A",
                f"{row['popularity_score']:.2f}" if 'popularity_score' in row and pd.notna(row['popularity_score']) else "N/A",
                f"{row['predicted_interest']:.3f}"
            ]
            tree.insert('', tk.END, values=values)
        
        # Pack treeview
        tree.pack(fill=tk.BOTH, expand=True)
        
        # If fewer than 5 recommendations, add a note
        if num_recs < 5:
            note_frame = ttk.Frame(parent)
            note_frame.pack(fill=tk.X, pady=(0, 5))
            
            note_label = ttk.Label(
                note_frame,
                text=f"Note: Only {num_recs} recommendations available for this user. Consider generating more recommendations.",
                font=("Segoe UI", 9, "italic"),
                foreground="#555555"
            )
            note_label.pack(pady=5)
    
    def open_html_dashboard(self):
        """Open the HTML dashboard in the default web browser"""
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               "dashboard", "consolidated_dashboard.html")
        
        if os.path.exists(html_path):
            # Open in default browser
            webbrowser.open(f"file://{html_path}")
            logging.info(f"Opening HTML dashboard: {html_path}")
        else:
            messagebox.showerror("Error", f"HTML dashboard not found at: {html_path}")
            logging.error(f"HTML dashboard not found at: {html_path}")
    
    def switch_to_gui(self):
        """Switch to GUI dashboard (refresh current view)"""
        # This is already the GUI, so just refresh
        self.refresh_dashboard()
        messagebox.showinfo("Information", "You are currently using the GUI Dashboard")
    
    def refresh_dashboard(self):
        """Refresh the dashboard"""
        # Clear all tabs
        for tab_id in range(self.tab_control.index("end")):
            tab = self.tab_control.nametowidget(self.tab_control.tabs()[tab_id])
            for widget in tab.winfo_children():
                widget.destroy()
        
        # Reload data and update UI
        self.load_all_data()
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
        Game Recommendation Dashboard
        
        A GUI application for game recommendations and analytics.
        
        Features:
        - Summary statistics
        - Popular games analysis
        - Sales analysis
        - Genre distribution
        - User recommendations
        
        You can switch between this GUI dashboard and the HTML dashboard
        using the View menu.
        """
        messagebox.showinfo("About", about_text)

# Main function to run the dashboard
def main():
    root = tk.Tk()
    app = GameRecommendationDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    main()