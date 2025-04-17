import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import datetime
import os
import random
from tqdm import tqdm
from run import SimpleMovieLensKnowledgeGraph
from forgetting import ForgettingMechanism
from evaluation import EvaluationMetrics

class MovieLensEvaluator:
    def __init__(self, ratings_file='ml-100k/u.data', movies_file='ml-100k/u.item', user_info_file='ml-100k/u.user'):
        """
        Initialize the MovieLensEvaluator with MovieLens 100K dataset.
        
        Args:
            ratings_file: Path to the ratings file
            movies_file: Path to the movies file
            user_info_file: Path to the user info file
        """
        # Check if files exist
        for file_path in [ratings_file, movies_file, user_info_file]:
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found. Please ensure MovieLens 100K dataset is properly downloaded.")
        
        self.load_data(ratings_file, movies_file, user_info_file)
        self.setup_knowledge_graph()
        self.initialize_forgetting_mechanism()
        self.initialize_evaluation_metrics()
    
    def load_data(self, ratings_file, movies_file, user_info_file):
        """Load MovieLens 100K dataset."""
        # Parse the ratings file
        try:
            self.ratings_df = pd.read_csv(ratings_file, 
                                        sep='\t', 
                                        names=['user_id', 'movie_id', 'rating', 'timestamp'])
        except:
            print(f"Failed to load ratings file. Creating sample data instead.")
            self.create_sample_data()
            return
        
        # Parse the movies file
        try:
            # MovieLens 100K format has | as separator and encoding issues with some titles
            self.movies_df = pd.read_csv(movies_file, 
                                        sep='|', 
                                        encoding='latin-1',
                                        names=['movie_id', 'title', 'release_date', 'video_release_date',
                                            'IMDb_URL'] + [f'genre_{i}' for i in range(19)])
            
            # Extract genre information
            genre_columns = [col for col in self.movies_df.columns if col.startswith('genre_')]
            self.movies_df['genres'] = self.movies_df[genre_columns].values.tolist()
        except:
            print(f"Failed to load movies file. Movie titles may not be available.")
            # Create a simple mapping of movie_id to title
            unique_movie_ids = self.ratings_df['movie_id'].unique()
            self.movies_df = pd.DataFrame({
                'movie_id': unique_movie_ids,
                'title': [f"Movie {movie_id}" for movie_id in unique_movie_ids]
            })
        
        # Parse the user info file if provided
        try:
            self.users_df = pd.read_csv(user_info_file,
                                       sep='|',
                                       names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
        except:
            print(f"Failed to load user info file. User demographics will not be available.")
            unique_user_ids = self.ratings_df['user_id'].unique()
            self.users_df = pd.DataFrame({
                'user_id': unique_user_ids,
                'age': [random.randint(18, 65) for _ in unique_user_ids],
                'gender': [random.choice(['M', 'F']) for _ in unique_user_ids],
                'occupation': ['unknown' for _ in unique_user_ids],
                'zip_code': ['00000' for _ in unique_user_ids]
            })
    
    def create_sample_data(self):
        """Create sample data if data files are not available."""
        # Sample users and movies
        user_ids = range(1, 101)  # 100 users
        movie_ids = range(1, 1001)  # 1000 movies
        
        # Create ratings data
        data = []
        current_time = datetime.datetime.now().timestamp()
        
        for user_id in user_ids:
            # Each user rates 20-50 movies
            num_ratings = np.random.randint(20, 51)
            user_movies = np.random.choice(movie_ids, num_ratings, replace=False)
            
            for movie_id in user_movies:
                # Create a rating between 1 and 5
                rating = np.random.randint(1, 6)
                # Create a timestamp within the last 180 days
                days_ago = np.random.randint(1, 180)
                timestamp = current_time - (days_ago * 24 * 60 * 60)
                
                data.append({
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'rating': rating,
                    'timestamp': timestamp
                })
        
        self.ratings_df = pd.DataFrame(data)
        
        # Create movie data
        movie_data = []
        for movie_id in movie_ids:
            movie_data.append({
                'movie_id': movie_id,
                'title': f"Movie {movie_id}",
                'genres': np.random.choice([0, 1], size=19).tolist()  # Random genre vector
            })
        
        self.movies_df = pd.DataFrame(movie_data)
        
        # Create user data
        user_data = []
        for user_id in user_ids:
            user_data.append({
                'user_id': user_id,
                'age': np.random.randint(18, 65),
                'gender': np.random.choice(['M', 'F']),
                'occupation': np.random.choice(['student', 'programmer', 'teacher', 'other']),
                'zip_code': f"{np.random.randint(10000, 99999)}"
            })
        
        self.users_df = pd.DataFrame(user_data)
    
    def setup_knowledge_graph(self):
        """Set up knowledge graph with the loaded data."""
        
        self.kg = SimpleMovieLensKnowledgeGraph(ratings_file=None)  # We'll replace its data
        self.kg.ratings_df = self.ratings_df
        
        # Create movie features (genre vectors)
        self.kg.movie_features = {}
        
        for _, movie in self.movies_df.iterrows():
            movie_id = movie['movie_id']
            
            if 'genres' in movie and isinstance(movie['genres'], list):
                # Use provided genre vector
                genre_vector = np.array(movie['genres'])
            else:
                # Create genre vector from genre_X columns if available
                genre_columns = [col for col in self.movies_df.columns if col.startswith('genre_')]
                if genre_columns:
                    genre_vector = movie[genre_columns].values
                else:
                    # Create random genre vector as fallback
                    genre_vector = np.zeros(19)
                    genre_vector[np.random.choice(range(19), np.random.randint(1, 4), replace=False)] = 1
            
            self.kg.movie_features[movie_id] = genre_vector
        
        # Create user profiles
        self.kg.user_profiles = {}
        for user_id in self.ratings_df['user_id'].unique():
            user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
            rated_movies = set(user_ratings['movie_id'].values)
            
            # Calculate genre preferences
            if rated_movies:
                genre_vectors = [self.kg.movie_features.get(mid, np.zeros(19)) for mid in rated_movies]
                genre_preferences = np.mean(genre_vectors, axis=0)
            else:
                genre_preferences = np.zeros(19)
                
            self.kg.user_profiles[user_id] = {
                'rated_movies': rated_movies,
                'genre_preferences': genre_preferences
            }
    
    def initialize_forgetting_mechanism(self):
        """Initialize the forgetting mechanism."""
        self.fm = ForgettingMechanism(self.kg)
    
    def initialize_evaluation_metrics(self):
        """Initialize evaluation metrics."""
        self.evaluator = EvaluationMetrics(self.kg, self.fm)
    
    def create_train_test_splits(self, test_ratio=0.2, seed=42):
        """
        Create train-test splits for evaluation.
        
        Args:
            test_ratio: Ratio of ratings to use as test set
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary mapping user_id to set of test movie_ids
        """
        np.random.seed(seed)
        train_ratings = []
        test_ratings = []
        
        self.test_data = {}
        
        # Split by user
        for user_id in self.ratings_df['user_id'].unique():
            user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
            
            # Only include users with at least 5 ratings
            if len(user_ratings) >= 5:
                # Randomly select test ratings
                test_size = max(1, int(len(user_ratings) * test_ratio))
                test_indices = np.random.choice(user_ratings.index, test_size, replace=False)
                
                user_test = user_ratings.loc[test_indices]
                user_train = user_ratings.drop(test_indices)
                
                train_ratings.append(user_train)
                test_ratings.append(user_test)
                
                # Store test movie IDs for evaluation
                self.test_data[user_id] = set(user_test['movie_id'].values)
        
        # Combine all train and test ratings
        self.train_ratings_df = pd.concat(train_ratings)
        self.test_ratings_df = pd.concat(test_ratings)
        
        # Update the knowledge graph with training data only
        self.kg.ratings_df = self.train_ratings_df
        
        # Rebuild user profiles with training data only
        self.kg.user_profiles = {}
        for user_id in self.train_ratings_df['user_id'].unique():
            user_ratings = self.train_ratings_df[self.train_ratings_df['user_id'] == user_id]
            rated_movies = set(user_ratings['movie_id'].values)
            
            # Calculate genre preferences
            if rated_movies:
                genre_vectors = [self.kg.movie_features.get(mid, np.zeros(19)) for mid in rated_movies]
                genre_preferences = np.mean(genre_vectors, axis=0)
            else:
                genre_preferences = np.zeros(19)
                
            self.kg.user_profiles[user_id] = {
                'rated_movies': rated_movies,
                'genre_preferences': genre_preferences
            }
        
        return self.test_data
    
    def evaluate_forgetting_strategies(self, user_ids=None, k_values=[5, 10, 20]):
        """
        Evaluate different forgetting strategies on selected users.
        
        Args:
            user_ids: List of user IDs to evaluate (if None, select a random sample)
            k_values: List of k values for evaluation metrics
            
        Returns:
            DataFrame with evaluation results
        """
        # If no user_ids provided, select a random sample
        if user_ids is None:
            if len(self.test_data) <= 50:
                user_ids = list(self.test_data.keys())
            else:
                user_ids = random.sample(list(self.test_data.keys()), 50)
        
        # Filter to users in test data
        user_ids = [u for u in user_ids if u in self.test_data]
        
        # Create recommendation function without forgetting
        def get_recommendations_baseline(user_id, n=20):
            return self.kg.get_personalized_recommendations(user_id, n=n)
        
        # Define forgetting strategies to evaluate
        strategies = {
            'Baseline': lambda user_id: None,  # No forgetting applied
            'Time-based': lambda user_id: self.fm.implement_time_based_decay(user_id, decay_parameter=0.1),
            'Usage-based': lambda user_id: self.fm.implement_usage_based_decay(user_id, interaction_threshold=3),
            'Hybrid': lambda user_id: self.fm.create_hybrid_decay_function(
                user_id, time_weight=0.4, usage_weight=0.3, novelty_weight=0.3
            ),
            'Personalized': lambda user_id: self.fm.create_hybrid_decay_function(
                user_id, 
                time_weight=self.fm.personalize_forgetting_parameters(user_id).get('time_weight', 0.4),
                usage_weight=self.fm.personalize_forgetting_parameters(user_id).get('usage_weight', 0.3),
                novelty_weight=self.fm.personalize_forgetting_parameters(user_id).get('novelty_weight', 0.3)
            )
        }
        
        # Create recommendation functions for each strategy
        recommendation_functions = {
            'Baseline': get_recommendations_baseline,
            'Time-based': lambda user_id, n=20: self.fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
                'personalized', {
                    'decay_parameter': 0.1,
                    'time_weight': 0.6,
                    'usage_weight': 0.2,
                    'novelty_weight': 0.2,
                    'forgetting_factor': 0.5
                }
            )(user_id, n=n),
            'Usage-based': lambda user_id, n=20: self.fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
                'personalized', {
                    'interaction_threshold': 3,
                    'time_weight': 0.2,
                    'usage_weight': 0.6,
                    'novelty_weight': 0.2,
                    'forgetting_factor': 0.5
                }
            )(user_id, n=n),
            'Hybrid': lambda user_id, n=20: self.fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
                'personalized', {
                    'time_weight': 0.4,
                    'usage_weight': 0.3,
                    'novelty_weight': 0.3,
                    'forgetting_factor': 0.5
                }
            )(user_id, n=n),
            'Personalized': lambda user_id, n=20: self.fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
                'personalized', {
                    'time_weight': self.fm.personalize_forgetting_parameters(user_id).get('time_weight', 0.4),
                    'usage_weight': self.fm.personalize_forgetting_parameters(user_id).get('usage_weight', 0.3),
                    'novelty_weight': self.fm.personalize_forgetting_parameters(user_id).get('novelty_weight', 0.3),
                    'forgetting_factor': 0.5
                }
            )(user_id, n=n)
        }
        
        results_all = []
        
        # For each user, evaluate all strategies
        for user_id in tqdm(user_ids, desc="Evaluating users"):
            if user_id not in self.test_data:
                continue
                
            test_items = self.test_data[user_id]
            
            # Store original memory strengths for this user
            original_memory_strengths = {}
            for key, value in self.fm.memory_strength.items():
                if key[0] == user_id:
                    original_memory_strengths[key] = value
            
            # Evaluate each strategy
            for strategy_name, apply_strategy in strategies.items():
                # Reset memory strengths to original values
                for key, value in original_memory_strengths.items():
                    self.fm.memory_strength[key] = value
                
                # Apply the strategy (except for baseline)
                if strategy_name != 'Baseline':
                    apply_strategy(user_id)
                
                # Get recommendations
                recommendations = recommendation_functions[strategy_name](user_id, n=max(k_values))
                
                # Calculate metrics
                for k in k_values:
                    hit_rate = self.evaluator.calculate_hit_rate_at_k(test_items, recommendations, k)
                    precision = self.evaluator.calculate_precision_at_k(test_items, recommendations, k)
                    recall = self.evaluator.calculate_recall_at_k(test_items, recommendations, k)
                    
                    results_all.append({
                        'user_id': user_id,
                        'strategy': strategy_name,
                        'k': k,
                        'hit_rate': hit_rate,
                        'precision': precision,
                        'recall': recall
                    })
                
                # Calculate MRR
                mrr = self.evaluator.calculate_mrr(test_items, recommendations)
                
                results_all.append({
                    'user_id': user_id,
                    'strategy': strategy_name,
                    'k': 'MRR',
                    'value': mrr
                })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results_all)
        
        return results_df
    
    def visualize_results(self, results_df):
        """
        Visualize the evaluation results.
        
        Args:
            results_df: DataFrame with evaluation results from evaluate_forgetting_strategies
        """
        # Create separate dataframes for different metrics
        metric_dfs = {}
        
        # Process hit_rate, precision, recall
        for metric in ['hit_rate', 'precision', 'recall']:
            metric_data = results_df[results_df[metric].notna()].copy()
            # Keep only numeric k values
            metric_data = metric_data[metric_data['k'].apply(lambda x: isinstance(x, (int, float)))]
            metric_dfs[metric] = metric_data
        
        # Process MRR separately
        mrr_data = results_df[results_df['k'] == 'MRR'].copy()
        metric_dfs['mrr'] = mrr_data
        
        # Create visualizations
        metrics = ['hit_rate', 'precision', 'recall', 'mrr']
        metric_names = ['Hit Rate', 'Precision', 'Recall', 'MRR']
        
        # 1. Compare strategies by k value for hit_rate, precision, recall
        for i, metric in enumerate(['hit_rate', 'precision', 'recall']):
            df = metric_dfs[metric]
            
            plt.figure(figsize=(12, 6))
            
            # Group by strategy and k, then calculate mean
            pivot_df = df.pivot_table(index='k', columns='strategy', values=metric, aggfunc='mean')
            
            # Plot
            ax = pivot_df.plot(marker='o', markersize=8, linewidth=2)
            
            plt.title(f'Average {metric_names[i]} by Strategy and k')
            plt.xlabel('k')
            plt.ylabel(metric_names[i])
            plt.grid(True, alpha=0.3)
            plt.legend(title='Strategy')
            
            # Add value labels
            for line in ax.lines:
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                for x, y in zip(xdata, ydata):
                    ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                               xytext=(0, 5), ha='center')
            
            plt.tight_layout()
            plt.savefig(f'{metric}_by_k.png')
            plt.show()
        
        # 2. Compare strategies for MRR
        mrr_df = metric_dfs['mrr']
        plt.figure(figsize=(10, 6))
        
        # Group by strategy and calculate mean
        mrr_means = mrr_df.groupby('strategy')['value'].mean().sort_values(ascending=False)
        
        # Plot
        bars = plt.bar(mrr_means.index, mrr_means.values, color='skyblue')
        
        plt.title('Average MRR by Strategy')
        plt.xlabel('Strategy')
        plt.ylabel('MRR')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('mrr_by_strategy.png')
        plt.show()
        
        # 3. Create boxplots to show distribution of metrics across users
        for i, metric in enumerate(['hit_rate', 'precision', 'recall']):
            df = metric_dfs[metric]
            
            # Select a specific k value for comparison (e.g., k=10)
            if 10 in df['k'].unique():
                k_value = 10
            else:
                k_value = df['k'].unique()[0]  # Use first available k
                
            k_df = df[df['k'] == k_value]
            
            plt.figure(figsize=(12, 6))
            
            # Create boxplot
            sns.boxplot(x='strategy', y=metric, data=k_df)
            
            plt.title(f'Distribution of {metric_names[i]} at k={k_value} Across Users')
            plt.xlabel('Strategy')
            plt.ylabel(metric_names[i])
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(f'{metric}_k{k_value}_distribution.png')
            plt.show()
        
        # 4. Create boxplot for MRR distribution
        plt.figure(figsize=(12, 6))
        
        # Create boxplot
        sns.boxplot(x='strategy', y='value', data=mrr_df)
        
        plt.title('Distribution of MRR Across Users')
        plt.xlabel('Strategy')
        plt.ylabel('MRR')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('mrr_distribution.png')
        plt.show()
        
        # 5. Summarize results in a table
        print("\n===== Summary of Results =====")
        for metric in metrics:
            if metric == 'mrr':
                # For MRR
                summary = mrr_df.groupby('strategy')['value'].agg(['mean', 'std', 'min', 'max'])
                print(f"\n{metric_names[3]} by Strategy:")
            else:
                # For other metrics, use k=10
                if 10 in metric_dfs[metric]['k'].unique():
                    k_value = 10
                else:
                    k_value = metric_dfs[metric]['k'].unique()[0]
                
                summary = metric_dfs[metric][metric_dfs[metric]['k'] == k_value].groupby('strategy')[metric].agg(['mean', 'std', 'min', 'max'])
                print(f"\n{metric_names[metrics.index(metric)]} at k={k_value} by Strategy:")
            
            print(summary.round(4))

def main():
    """Main function to run the evaluation."""
    # Initialize the evaluator
    print("Initializing MovieLens evaluator...")
    evaluator = MovieLensEvaluator()
    
    # Create train-test splits
    print("Creating train-test splits...")
    test_data = evaluator.create_train_test_splits(test_ratio=0.2)
    print(f"Created test data for {len(test_data)} users")
    
    # Evaluate forgetting strategies
    print("Evaluating forgetting strategies...")
    results = evaluator.evaluate_forgetting_strategies(k_values=[5, 10, 20])
    
    # Visualize results
    print("Visualizing results...")
    evaluator.visualize_results(results)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()