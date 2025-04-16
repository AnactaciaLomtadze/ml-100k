import numpy as np
import datetime
import math
from collections import defaultdict

class ForgettingMechanism:
    def __init__(self, knowledge_graph):
        """
        Initialize the forgetting mechanism for a knowledge graph.
        
        Args:
            knowledge_graph: The MovieLensKnowledgeGraph instance
        """
        self.kg = knowledge_graph
        self.memory_strength = {}  # Maps (user_id, movie_id) to memory strength
        self.last_interaction_time = {}  # Maps (user_id, movie_id) to last interaction timestamp
        self.interaction_counts = defaultdict(int)  # Maps (user_id, movie_id) to interaction count
        self.user_activity_patterns = {}  # Maps user_id to activity pattern metrics
        
        # Initialize memory strengths from existing graph data
        self._initialize_memory_strengths()
    
    def _initialize_memory_strengths(self):
        """Initialize memory strengths from existing ratings data."""
        for _, rating in self.kg.ratings_df.iterrows():
            user_id = rating['user_id']
            movie_id = rating['movie_id']
            rating_value = rating['rating']
            timestamp = rating['timestamp']
            
            # Initial memory strength is based on the rating value (normalized to [0,1])
            memory_strength = rating_value / 5.0
            
            self.memory_strength[(user_id, movie_id)] = memory_strength
            self.last_interaction_time[(user_id, movie_id)] = timestamp
            self.interaction_counts[(user_id, movie_id)] += 1
    
    def implement_time_based_decay(self, user_id, decay_parameter=0.1):
        """
        Implement time-based decay for a user's memories.
        
        Args:
            user_id: The user ID to apply decay to
            decay_parameter: Controls how quickly memories decay (smaller values = slower decay)
            
        Returns:
            Dictionary of updated memory strengths for the user
        """
        current_time = datetime.datetime.now().timestamp()
        user_memories = {}
        
        # Apply decay to all movies the user has interacted with
        for (u_id, movie_id), strength in self.memory_strength.items():
            if u_id == user_id:
                last_time = self.last_interaction_time.get((u_id, movie_id), 0)
                time_diff = current_time - last_time
                
                # Exponential decay formula: strength * e^(-decay_parameter * time_diff)
                # Time difference is in seconds, convert to days for more reasonable decay
                days_diff = time_diff / (24 * 60 * 60)
                decayed_strength = strength * math.exp(-decay_parameter * days_diff)
                
                # Update memory strength
                self.memory_strength[(u_id, movie_id)] = max(0.001, decayed_strength)  # Prevent complete forgetting
                user_memories[movie_id] = self.memory_strength[(u_id, movie_id)]
        
        return user_memories
    
    def implement_usage_based_decay(self, user_id, interaction_threshold=3):
        """
        Implement usage-based decay where less frequently accessed items decay faster.
        
        Args:
            user_id: The user ID to apply decay to
            interaction_threshold: Number of interactions below which memory decays faster
            
        Returns:
            Dictionary of updated memory strengths for the user
        """
        user_memories = {}
        
        for (u_id, movie_id), strength in self.memory_strength.items():
            if u_id == user_id:
                interaction_count = self.interaction_counts.get((u_id, movie_id), 0)
                
                # Apply stronger decay to less frequently accessed items
                if interaction_count < interaction_threshold:
                    usage_decay_factor = 0.8  # Stronger decay for less used items
                else:
                    usage_decay_factor = 0.95  # Weaker decay for frequently used items
                
                # Apply usage-based decay
                decayed_strength = strength * usage_decay_factor
                self.memory_strength[(u_id, movie_id)] = max(0.001, decayed_strength)
                user_memories[movie_id] = self.memory_strength[(u_id, movie_id)]
        
        return user_memories
    
    
    
    def create_hybrid_decay_function(self, user_id, time_weight=0.4, usage_weight=0.3, novelty_weight=0.3):
        """
        Create a hybrid decay function that combines time-based, usage-based, and novelty-based decay.
        
        Args:
            user_id: The user ID to apply decay to
            time_weight: Weight for time-based decay
            usage_weight: Weight for usage-based decay
            novelty_weight: Weight for novelty-based decay
            
        Returns:
            Dictionary of updated memory strengths for the user
        """
        current_time = datetime.datetime.now().timestamp()
        user_memories = {}
        
        # Get all movies the user has interacted with
        user_movies = [(m_id, strength) for (u_id, m_id), strength in self.memory_strength.items() if u_id == user_id]
        
        # Calculate average rating time to determine movie novelty
        avg_timestamp = sum(self.last_interaction_time.get((user_id, m_id), 0) for m_id, _ in user_movies) / max(1, len(user_movies))
        
        for movie_id, strength in user_movies:
            # Time-based component
            last_time = self.last_interaction_time.get((user_id, movie_id), 0)
            time_diff = current_time - last_time
            days_diff = time_diff / (24 * 60 * 60)
            time_decay = math.exp(-0.05 * days_diff)  # Slower decay rate
            
            # Usage-based component
            interaction_count = self.interaction_counts.get((user_id, movie_id), 0)
            usage_factor = min(1.0, interaction_count / 5.0)  # Normalize to [0,1]
            
            # Novelty-based component
            movie_timestamp = self.last_interaction_time.get((user_id, movie_id), 0)
            novelty_factor = 1.0 if movie_timestamp > avg_timestamp else 0.8
            
            # Combine all factors
            hybrid_factor = (time_weight * time_decay + 
                             usage_weight * usage_factor + 
                             novelty_weight * novelty_factor)
            
            # Apply decay
            new_strength = strength * hybrid_factor
            self.memory_strength[(user_id, movie_id)] = max(0.001, min(1.0, new_strength))
            user_memories[movie_id] = self.memory_strength[(user_id, movie_id)]
        
        return user_memories
    
    def personalize_forgetting_parameters(self, user_id, activity_pattern=None):
        """
        Personalize forgetting parameters based on user activity patterns.
        
        Args:
            user_id: The user ID
            activity_pattern: Optional activity pattern dictionary. If None, it will be calculated.
            
        Returns:
            Dictionary of personalized forgetting parameters
        """
        if activity_pattern is None:
            # Calculate activity pattern from user data
            user_ratings = self.kg.ratings_df[self.kg.ratings_df['user_id'] == user_id]
            
            if user_ratings.empty:
                return {
                    'time_decay_rate': 0.1,  # Default values
                    'usage_threshold': 3,
                    'time_weight': 0.4,
                    'usage_weight': 0.3,
                    'novelty_weight': 0.3
                }
            
            # Calculate average time between ratings
            timestamps = sorted(user_ratings['timestamp'].values)
            if len(timestamps) > 1:
                time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                avg_time_diff = sum(time_diffs) / len(time_diffs)
                time_variability = np.std(time_diffs) / max(1, avg_time_diff)  # Normalized std dev
            else:
                time_variability = 0.5  # Default if only one rating
            
            # Calculate rating diversity
            rating_diversity = user_ratings['rating'].std() / 2.5  # Normalized by half the rating scale
            
            # Calculate genre diversity
            user_movies = user_ratings['movie_id'].values
            if len(user_movies) > 0:
                genre_vectors = [self.kg.movie_features.get(mid, np.zeros(19)) for mid in user_movies if mid in self.kg.movie_features]
                if genre_vectors:
                    genre_avg = np.mean(genre_vectors, axis=0)
                    genre_diversity = np.sum(genre_avg * (1 - genre_avg))  # Higher when user watches diverse genres
                else:
                    genre_diversity = 0.5
            else:
                genre_diversity = 0.5
            
            # Store activity pattern
            activity_pattern = {
                'time_variability': time_variability,
                'rating_diversity': rating_diversity,
                'genre_diversity': genre_diversity,
                'rating_count': len(user_ratings)
            }
            self.user_activity_patterns[user_id] = activity_pattern
        
        # Adjust parameters based on activity pattern
        time_decay_rate = 0.05 + 0.1 * activity_pattern['time_variability']  # [0.05, 0.15]
        usage_threshold = max(1, round(3 * activity_pattern['rating_count'] / 50))  # Adjust based on user activity
        
        # Users with diverse tastes should have slower time decay but faster novelty considerations
        time_weight = 0.5 - 0.2 * activity_pattern['genre_diversity']  # [0.3, 0.5]
        novelty_weight = 0.2 + 0.2 * activity_pattern['genre_diversity']  # [0.2, 0.4]
        usage_weight = 1.0 - time_weight - novelty_weight
        
        return {
            'time_decay_rate': time_decay_rate,
            'usage_threshold': usage_threshold,
            'time_weight': time_weight,
            'usage_weight': usage_weight,
            'novelty_weight': novelty_weight
        }
    
    def dynamic_half_life_adjustment(self, user_profile):
        """
        Dynamically adjust the half-life of the forgetting curve based on user profile.
        
        Args:
            user_profile: The user profile dictionary from KG or a user_id
            
        Returns:
            Dictionary mapping movie_id to adjusted half-life values
        """
        if isinstance(user_profile, int):
            user_id = user_profile
            if user_id in self.kg.user_profiles:
                user_profile = self.kg.user_profiles[user_id]
            else:
                return {}
        
        # Get genre preferences from user profile
        genre_preferences = user_profile.get('genre_preferences', np.zeros(19))
        
        # Scale the genre preferences to get positive values
        scaled_preferences = (genre_preferences - np.min(genre_preferences)) / (np.max(genre_preferences) - np.min(genre_preferences) + 1e-10)
        
        half_lives = {}
        
        # For each movie the user has rated
        for movie_id in user_profile.get('rated_movies', set()):
            if movie_id in self.kg.movie_features:
                # Get movie genre features
                movie_genres = self.kg.movie_features[movie_id]
                
                # Calculate relevance to user preferences
                genre_match = np.sum(scaled_preferences * movie_genres) / (np.sum(movie_genres) + 1e-10)
                
                # Adjust half-life based on genre match
                # Movies that match user preferences have longer half-lives
                base_half_life = 30  # Base half-life in days
                adjusted_half_life = base_half_life * (1 + genre_match)
                
                half_lives[movie_id] = adjusted_half_life
        
        return half_lives
    
    def apply_forgetting_to_recommendations(self, user_id, recommendation_scores, forgetting_factor=0.5):
        """
        Apply forgetting mechanism to adjust recommendation scores.
        
        Args:
            user_id: The user ID
            recommendation_scores: Dictionary mapping movie_id to recommendation score
            forgetting_factor: How strongly forgetting affects recommendations (0-1)
            
        Returns:
            Dictionary of adjusted recommendation scores
        """
        adjusted_scores = {}
        
        for movie_id, score in recommendation_scores.items():
            memory_strength = self.memory_strength.get((user_id, movie_id), 1.0)
            
            # Adjust score based on memory strength
            # Items with lower memory strength get boosted (novel items)
            # Items with high memory strength get slightly reduced (familiar items)
            novelty_boost = (1.0 - memory_strength) * forgetting_factor
            
            adjusted_scores[movie_id] = score * (1.0 + novelty_boost)
        
        return adjusted_scores
    
    def integrate_forgetting_mechanism_into_recommendation_pipeline(self, recommendation_algorithm, forgetting_parameters):
        """
        Integrate forgetting mechanism into the recommendation pipeline.
        
        Args:
            recommendation_algorithm: Function that returns recommendation scores for a user
            forgetting_parameters: Dictionary of forgetting parameters
            
        Returns:
            Function that generates recommendations with forgetting mechanism applied
        """
        def forgetting_aware_recommendations(user_id, n=10):
            # Get personalized forgetting parameters if not provided
            if user_id not in self.user_activity_patterns:
                params = self.personalize_forgetting_parameters(user_id)
            else:
                params = forgetting_parameters
            
            # Apply hybrid decay to update memory strengths
            self.create_hybrid_decay_function(
                user_id, 
                time_weight=params['time_weight'],
                usage_weight=params['usage_weight'],
                novelty_weight=params['novelty_weight']
            )
            
            # Get base recommendations
            if recommendation_algorithm == 'personalized':
                movie_ids = self.kg.get_personalized_recommendations(user_id, n=n*2)
                
                # Create scores dictionary (normalized to 0-1)
                scores = {mid: (n*2 - i) / (n*2) for i, mid in enumerate(movie_ids)}
                
            elif recommendation_algorithm == 'graph_based':
                movie_ids = self.kg.get_graph_based_recommendations(user_id, n=n*2)
                scores = {mid: (n*2 - i) / (n*2) for i, mid in enumerate(movie_ids)}
                
            else:
                # Custom recommendation algorithm that returns scores
                scores = recommendation_algorithm(user_id)
            
            # Apply forgetting mechanism to adjust scores
            adjusted_scores = self.apply_forgetting_to_recommendations(
                user_id, 
                scores, 
                forgetting_factor=params.get('forgetting_factor', 0.5)
            )
            
            # Sort by adjusted scores and return top n
            sorted_recommendations = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)[:n]
            return [movie_id for movie_id, _ in sorted_recommendations]
        
        return forgetting_aware_recommendations
    
    def serialize_and_store_memory_state(self, file_path, compression_level=0):
        """
        Serialize and store the current memory state.
        
        Args:
            file_path: Path to store the memory state
            compression_level: 0-9 compression level (0=none, 9=max)
            
        Returns:
            True if successful, False otherwise
        """
        import pickle
        import gzip
        
        try:
            data = {
                'memory_strength': self.memory_strength,
                'last_interaction_time': self.last_interaction_time,
                'interaction_counts': self.interaction_counts,
                'user_activity_patterns': self.user_activity_patterns
            }
            
            if compression_level > 0:
                with gzip.open(file_path, 'wb', compresslevel=compression_level) as f:
                    pickle.dump(data, f)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            
            return True
        except Exception as e:
            print(f"Error storing memory state: {e}")
            return False
    
    def load_and_restore_memory_state(self, file_path, validation_check=True):
        """
        Load and restore a previously saved memory state.
        
        Args:
            file_path: Path to the stored memory state
            validation_check: Whether to validate the loaded data
            
        Returns:
            True if successful, False otherwise
        """
        import pickle
        import gzip
        
        try:
            # Try to load as gzipped first
            try:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
            except:
                # If not gzipped, try normal pickle
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            # Validation check
            if validation_check:
                required_keys = ['memory_strength', 'last_interaction_time', 
                                'interaction_counts', 'user_activity_patterns']
                
                if not all(key in data for key in required_keys):
                    print("Invalid memory state file: missing required data")
                    return False
            
            # Restore state
            self.memory_strength = data['memory_strength']
            self.last_interaction_time = data['last_interaction_time']
            self.interaction_counts = data['interaction_counts']
            self.user_activity_patterns = data['user_activity_patterns']
            
            return True
        except Exception as e:
            print(f"Error loading memory state: {e}")
            return False