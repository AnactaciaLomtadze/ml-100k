import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from collections import defaultdict
from evaluation import EvaluationMetrics 
from forgetting import ForgettingMechanism



class SimpleMovieLensKnowledgeGraph:
    def __init__(self, ratings_file=None):
        """Simple knowledge graph for testing purposes."""
        self.movie_features = {}  # Maps movie_id to genre features
        self.user_profiles = {}   # Maps user_id to user profile
        
        # Create sample data if no ratings file provided
        if ratings_file:
            self.ratings_df = pd.read_csv(ratings_file)
        else:
            # Create sample ratings data
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data for testing."""
        # Sample users and movies
        users = range(1, 121)  # 10 users
        movies = range(101, 221)  # 20 movies
        
        # Create ratings data
        data = []
        current_time = datetime.datetime.now().timestamp()
        
        for user_id in users:
            # Each user rates 5-10 movies
            num_ratings = np.random.randint(5, 11)
            user_movies = np.random.choice(movies, num_ratings, replace=False)
            
            for movie_id in user_movies:
                # Create a rating between 1 and 5
                rating = np.random.randint(1, 6)
                # Create a timestamp within the last 90 days
                days_ago = np.random.randint(1, 90)
                timestamp = current_time - (days_ago * 24 * 60 * 60)
                
                data.append({
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'rating': rating,
                    'timestamp': timestamp
                })
        
        self.ratings_df = pd.DataFrame(data)
        
        # Create movie features (19 genres as in the original code)
        for movie_id in movies:
            # Randomly assign 1-3 genres to each movie
            num_genres = np.random.randint(1, 4)
            genre_indices = np.random.choice(range(19), num_genres, replace=False)
            genre_vector = np.zeros(19)
            genre_vector[genre_indices] = 1
            self.movie_features[movie_id] = genre_vector
        
        # Create user profiles
        for user_id in users:
            user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
            rated_movies = set(user_ratings['movie_id'].values)
            
            # Calculate genre preferences
            if rated_movies:
                genre_vectors = [self.movie_features.get(mid, np.zeros(19)) for mid in rated_movies]
                genre_preferences = np.mean(genre_vectors, axis=0)
            else:
                genre_preferences = np.zeros(19)
                
            self.user_profiles[user_id] = {
                'rated_movies': rated_movies,
                'genre_preferences': genre_preferences
            }
    
    def get_personalized_recommendations(self, user_id, n=10):
        """Simple personalized recommendation function."""
        # Find unrated movies
        rated_movies = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['movie_id'].values)
        all_movies = set(self.movie_features.keys())
        unrated_movies = all_movies - rated_movies
        
        if not unrated_movies:
            return []
        
        # Get user genre preferences
        if user_id in self.user_profiles:
            user_prefs = self.user_profiles[user_id]['genre_preferences']
        else:
            user_prefs = np.ones(19) / 19  # Uniform distribution
        
        # Calculate similarity scores
        scores = {}
        for movie_id in unrated_movies:
            if movie_id in self.movie_features:
                # Simple dot product similarity
                score = np.dot(user_prefs, self.movie_features[movie_id])
                scores[movie_id] = score
        
        # Sort and return top n
        sorted_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in sorted_movies[:n]]
    
    def get_graph_based_recommendations(self, user_id, n=10):
        """Simple graph-based recommendation function."""
        # For this demo, we'll use a simpler version that just returns different results
        # In a real system, this would use graph algorithms
        
        # Get user ratings
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        rated_movies = set(user_ratings['movie_id'].values)
        
        # Get all movies rated by users who rated the same movies
        similar_users = set()
        for movie_id in rated_movies:
            movie_raters = set(self.ratings_df[self.ratings_df['movie_id'] == movie_id]['user_id'].values)
            similar_users.update(movie_raters)
        
        # Remove the current user
        similar_users.discard(user_id)
        
        # Get movies rated by similar users
        candidate_movies = set()
        for sim_user in similar_users:
            sim_user_movies = set(self.ratings_df[self.ratings_df['user_id'] == sim_user]['movie_id'].values)
            candidate_movies.update(sim_user_movies)
        
        # Remove movies already rated by user
        candidate_movies -= rated_movies
        
        # Score based on frequency among similar users
        scores = defaultdict(int)
        for movie_id in candidate_movies:
            movie_ratings = self.ratings_df[self.ratings_df['movie_id'] == movie_id]
            movie_users = set(movie_ratings['user_id'].values)
            scores[movie_id] = len(movie_users.intersection(similar_users))
        
        # Sort and return top n
        sorted_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in sorted_movies[:n]]


def test_forgetting_and_evaluation():
    """Test the forgetting mechanism and evaluation metrics."""
    # Create knowledge graph with sample data
    print("Creating knowledge graph...")
    kg = SimpleMovieLensKnowledgeGraph()
    
    # Initialize forgetting mechanism
    print("Initializing forgetting mechanism...")
    fm = ForgettingMechanism(kg)
    
    # Initialize evaluation metrics
    print("Initializing evaluation metrics...")
    evaluator = EvaluationMetrics(kg, fm)
    
    # Select a test user
    test_user_id = 1
    
    

    # Test forgetting mechanism
  
    print(f"\nTesting forgetting mechanism for user {test_user_id}...")
    
    # Get recommendations before forgetting
    recommendations_before = kg.get_personalized_recommendations(test_user_id, n=10)
    print(f"Recommendations before forgetting: {recommendations_before}")
    
    # Apply time-based decay
    print("\nApplying time-based decay...")
    time_decay_memories = fm.implement_time_based_decay(test_user_id, decay_parameter=0.15)
    print(f"Memory strengths after time decay (sample): {dict(list(time_decay_memories.items())[:3])}...")
    
    # Apply usage-based decay
    print("\nApplying usage-based decay...")
    usage_decay_memories = fm.implement_usage_based_decay(test_user_id, interaction_threshold=2)
    print(f"Memory strengths after usage decay (sample): {dict(list(usage_decay_memories.items())[:3])}...")
    
    # Apply hybrid decay
    print("\nApplying hybrid decay...")
    hybrid_decay_memories = fm.create_hybrid_decay_function(test_user_id)
    print(f"Memory strengths after hybrid decay (sample): {dict(list(hybrid_decay_memories.items())[:3])}...")
    
    # Get personalized parameters
    print("\nGetting personalized forgetting parameters...")
    personalized_params = fm.personalize_forgetting_parameters(test_user_id)
    print(f"Personalized parameters: {personalized_params}")
    
    # Get dynamic half-life adjustments
    print("\nCalculating dynamic half-life adjustments...")
    half_lives = fm.dynamic_half_life_adjustment(test_user_id)
    print(f"Dynamic half-lives (sample): {dict(list(half_lives.items())[:3])}...")
    
    # Get recommendations after forgetting
    print("\nGetting recommendations after forgetting...")
    recommendation_algorithm = lambda user_id: {movie_id: (20 - i)/20 for i, movie_id in 
                                               enumerate(kg.get_personalized_recommendations(user_id, n=20))}
    
    forgetting_aware_recommender = fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
        recommendation_algorithm, personalized_params)
    
    recommendations_after = forgetting_aware_recommender(test_user_id)
    print(f"Recommendations after forgetting: {recommendations_after}")
  
    # Test evaluation metrics

    print("\nTesting evaluation metrics...")

    # Measure recommendation diversity
    print("\nMeasuring recommendation diversity...")
    diversity_metrics = evaluator.measure_recommendation_diversity_after_forgetting(
        recommendations_before, recommendations_after)
    print(f"Diversity metrics: {diversity_metrics}")
    
    # Calculate temporal relevance
    print("\nCalculating temporal relevance...")
    # Create current interests (latest genre preferences)
    current_interests = {i: pref for i, pref in 
                         enumerate(kg.user_profiles[test_user_id]['genre_preferences']) if pref > 0}
    
    relevance_before = evaluator.calculate_temporal_relevance_score(
        recommendations_before, current_interests)
    relevance_after = evaluator.calculate_temporal_relevance_score(
        recommendations_after, current_interests)
    print(f"Temporal relevance before: {relevance_before:.4f}")
    print(f"Temporal relevance after: {relevance_after:.4f}")
    
    # Evaluate catastrophic forgetting
    print("\nEvaluating catastrophic forgetting impact...")
    # Simulate performance timeline
    timeline = [(datetime.datetime.now() - datetime.timedelta(days=i), 0.8 - 0.02*i) 
                for i in range(10)]
    
    forgetting_impact = evaluator.evaluate_catastrophic_forgetting_impact(timeline)
    print(f"Forgetting impact: {forgetting_impact}")
    
    # Compute memory efficiency
    print("\nComputing memory efficiency...")
    # Simulate graph size reduction
    graph_size_before = (1000, 5000)  # (nodes, edges)
    graph_size_after = (800, 3500)    # After forgetting
    
    memory_efficiency = evaluator.compute_memory_efficiency_metrics(
        graph_size_before, graph_size_after)
    print(f"Memory efficiency: {memory_efficiency}")
    
    # Return all results for visualization
   
    return {
        'user_id': test_user_id,
        'recommendations_before': recommendations_before,
        'recommendations_after': recommendations_after,
        'diversity_metrics': diversity_metrics,
        'temporal_relevance_before': relevance_before,
        'temporal_relevance_after': relevance_after,
        'forgetting_impact': forgetting_impact,
        'memory_efficiency': memory_efficiency,
        'personalized_params': personalized_params,
        'memory_strengths': hybrid_decay_memories,
    }

def visualize_evaluation_results(results):
    """Visualize the evaluation results."""
    # Create a 2x2 grid of plots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Recommendation Diversity Comparison
    ax = axs[0, 0]
    diversity_data = [
        results['diversity_metrics']['genre_diversity_before'],
        results['diversity_metrics']['genre_diversity_after']
    ]
    
    bars = ax.bar(['Before Forgetting', 'After Forgetting'], diversity_data, color=['blue', 'orange'])
    ax.set_ylabel('Genre Diversity Score')
    ax.set_title('Impact of Forgetting on Recommendation Diversity')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 2. Temporal Relevance Comparison
    ax = axs[0, 1]
    relevance_data = [
        results['temporal_relevance_before'],
        results['temporal_relevance_after']
    ]
    
    bars = ax.bar(['Before Forgetting', 'After Forgetting'], relevance_data, color=['blue', 'orange'])
    ax.set_ylabel('Temporal Relevance Score')
    ax.set_title('Impact of Forgetting on Temporal Relevance')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 3. Jaccard Similarity & New Item Percentage
    ax = axs[1, 0]
    similarity_data = [
        results['diversity_metrics']['jaccard_similarity'],
        results['diversity_metrics']['new_item_percentage']
    ]
    
    bars = ax.bar(['Jaccard Similarity', 'New Item %'], similarity_data, color=['green', 'purple'])
    ax.set_ylabel('Score')
    ax.set_title('Recommendation Set Changes')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 4. Memory Efficiency Metrics
    ax = axs[1, 1]
    efficiency_data = [
        results['memory_efficiency']['node_reduction_ratio'],
        results['memory_efficiency']['edge_reduction_ratio'],
        results['memory_efficiency']['memory_efficiency_gain']
    ]
    
    bars = ax.bar(['Node Reduction', 'Edge Reduction', 'Memory Efficiency'], 
                 efficiency_data, color=['red', 'purple', 'brown'])
    ax.set_ylabel('Ratio')
    ax.set_title('Memory Efficiency Metrics')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('forgetting_evaluation_results.png')
    plt.show()
    
    # Additional visualization: Memory strength distribution
    plt.figure(figsize=(10, 6))
    memory_values = list(results['memory_strengths'].values())
    sns.histplot(memory_values, bins=20, kde=True)
    plt.title(f"Memory Strength Distribution for User {results['user_id']}")
    plt.xlabel("Memory Strength")
    plt.ylabel("Count")
    plt.savefig('memory_strength_distribution.png')
    plt.show()
    
    # Show personalized parameters
    plt.figure(figsize=(8, 6))
    params = results['personalized_params']
    param_names = list(params.keys())
    param_values = list(params.values())
    
    plt.barh(param_names, param_values, color='teal')
    plt.xlabel('Parameter Value')
    plt.title(f"Personalized Forgetting Parameters for User {results['user_id']}")
    plt.tight_layout()
    plt.savefig('personalized_parameters.png')
    plt.show()



if __name__ == "__main__":
    # Run the test
    print("Starting test of forgetting mechanism and evaluation metrics...")
    results = test_forgetting_and_evaluation()
    
    # Visualize the results
    print("\nVisualizing evaluation results...")
    visualize_evaluation_results(results)
    
    print("\nTest completed successfully!")