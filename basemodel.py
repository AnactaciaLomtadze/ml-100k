import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import datetime
import random

class MovieLensKnowledgeGraph:
    def __init__(self, data_path='./ml-100k'):
        """
        Initialize the MovieLens Knowledge Graph.
        
        Args:
            data_path: Path to the MovieLens dataset
        """
        self.data_path = data_path
        self.G = nx.Graph()  
        self.user_profiles = {}
        self.movie_features = {}
        self.ratings_df = None
        self.users_df = None
        self.movies_df = None
        self.similarity_matrix = None
        
    def load_data(self):
        """Load the MovieLens dataset."""
        ratings_path = os.path.join(self.data_path, 'u.data')
        self.ratings_df = pd.read_csv(
            ratings_path, 
            sep='\t', 
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        
    
        users_path = os.path.join(self.data_path, 'u.user')
        self.users_df = pd.read_csv(
            users_path,
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
        )
        
        movies_path = os.path.join(self.data_path, 'u.item')
        self.movies_df = pd.read_csv(
            movies_path,
            sep='|',
            encoding='latin-1',
            names=['movie_id', 'title', 'release_date', 'video_release_date', 
                   'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 
                   'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                   'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                   'Thriller', 'War', 'Western']
        )
        
        print(f"Loaded {len(self.ratings_df)} ratings from {self.ratings_df['user_id'].nunique()} users on {self.ratings_df['movie_id'].nunique()} movies")
        
    def build_knowledge_graph(self):
        """Build the knowledge graph from the MovieLens data."""
        if self.ratings_df is None:
            self.load_data()
        
        for _, user in self.users_df.iterrows():
            self.G.add_node(
                f"user_{user['user_id']}", 
                type='user',
                age=user['age'],
                gender=user['gender'],
                occupation=user['occupation']
            )
        
        
        for _, movie in self.movies_df.iterrows():
            genre_features = movie[5:].values.astype(int)  # All genre columns
            
          
            self.G.add_node(
                f"movie_{movie['movie_id']}", 
                type='movie',
                title=movie['title'],
                release_date=movie['release_date']
            )
            
            
            self.movie_features[movie['movie_id']] = genre_features
    
        for _, rating in self.ratings_df.iterrows():
            user_id = rating['user_id']
            movie_id = rating['movie_id']
            rating_value = rating['rating']
            timestamp = rating['timestamp']
            
    
            rating_time = datetime.datetime.fromtimestamp(timestamp)

            self.G.add_edge(
                f"user_{user_id}", 
                f"movie_{movie_id}", 
                weight=rating_value,
                timestamp=timestamp,
                rating_time=rating_time
            )
        
        self._add_movie_similarity_edges()
        
        self._build_user_profiles()
        
        print(f"Knowledge graph built with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        
    def _add_movie_similarity_edges(self, threshold=0.7):
        """
        Add movie-movie edges based on genre similarity.
        
        Args:
            threshold: Minimum similarity score to create an edge
        """
        
        movie_ids = list(self.movie_features.keys())
        feature_matrix = np.array([self.movie_features[mid] for mid in movie_ids])
        
    
        self.similarity_matrix = cosine_similarity(feature_matrix)
        
    
        for i in range(len(movie_ids)):
            for j in range(i+1, len(movie_ids)):
                similarity = self.similarity_matrix[i, j]
                if similarity >= threshold:
                    self.G.add_edge(
                        f"movie_{movie_ids[i]}", 
                        f"movie_{movie_ids[j]}", 
                        weight=similarity,
                        relation_type='similar'
                    )
    
    def _build_user_profiles(self):
        """Build user profiles based on their ratings."""
        
        user_ratings = self.ratings_df.groupby('user_id')
        
        for user_id, ratings in user_ratings:
        
            avg_rating = ratings['rating'].mean()
            
            rated_movies = ratings[['movie_id', 'rating']].values
            
            genre_preferences = np.zeros(19)
            genre_counts = np.zeros(19)
            
            for movie_id, rating in rated_movies:
                if movie_id in self.movie_features:
            
                    movie_genres = self.movie_features[movie_id]
                    
                    normalized_rating = rating - avg_rating
                    
                    for i, has_genre in enumerate(movie_genres):
                        if has_genre:
                            genre_preferences[i] += normalized_rating
                            genre_counts[i] += 1
          
            genre_counts[genre_counts == 0] = 1
            
            genre_preferences = genre_preferences / genre_counts
            
            self.user_profiles[user_id] = {
                'avg_rating': avg_rating,
                'genre_preferences': genre_preferences,
                'rated_movies': set(ratings['movie_id'].values)
            }
    
    def get_personalized_recommendations(self, user_id, n=10):
        """
        Get personalized movie recommendations for a user.
        
        Args:
            user_id: The user ID to generate recommendations for
            n: Number of recommendations to return
            
        Returns:
            A list of recommended movie IDs
        """
        if user_id not in self.user_profiles:
            print(f"User {user_id} not found in profiles")
            return []
        
        user_profile = self.user_profiles[user_id]
        rated_movies = user_profile['rated_movies']
        genre_preferences = user_profile['genre_preferences']
        
        movie_scores = []
        
        for movie_id, features in self.movie_features.items():
            if movie_id not in rated_movies:
                score = np.dot(genre_preferences, features)
                
                movie_rating_count = len(self.ratings_df[self.ratings_df['movie_id'] == movie_id])
                popularity_factor = np.log1p(movie_rating_count) / 10
                
                final_score = score + popularity_factor
                
                movie_scores.append((movie_id, final_score))
        
        recommendations = sorted(movie_scores, key=lambda x: x[1], reverse=True)[:n]
        return [movie_id for movie_id, _ in recommendations]
    
    def get_graph_based_recommendations(self, user_id, n=10, depth=2):
        """
        Get recommendations using graph traversal.
        
        Args:
            user_id: The user ID to generate recommendations for
            n: Number of recommendations to return
            depth: How many hops to traverse in the graph
            
        Returns:
            A list of recommended movie IDs
        """
        if f"user_{user_id}" not in self.G:
            print(f"User {user_id} not found in graph")
            return []
        
        user_node = f"user_{user_id}"
        rated_movies = set()
        candidate_scores = defaultdict(float)
        
        for neighbor in self.G.neighbors(user_node):
            if neighbor.startswith("movie_"):
                movie_id = int(neighbor.split("_")[1])
                rated_movies.add(movie_id)
        
        paths = [(user_node, [])]
        visited = {user_node}
        
        for _ in range(depth):
            new_paths = []
            for node, path in paths:
                for neighbor in self.G.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_path = path + [(node, neighbor)]
                        new_paths.append((neighbor, new_path))
                        
                        if neighbor.startswith("movie_"):
                            movie_id = int(neighbor.split("_")[1])
                            if movie_id not in rated_movies:
                                path_weight = 1.0
                                for i in range(len(new_path)):
                                    n1, n2 = new_path[i]
                                    edge_weight = self.G.edges[n1, n2].get('weight', 1.0)
                                    path_weight *= edge_weight
                                
                                path_score = path_weight / len(new_path)
                                candidate_scores[movie_id] += path_score
            
            paths = new_paths
        
        recommendations = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [movie_id for movie_id, _ in recommendations]
    
    def visualize_subgraph(self, center_node, depth=1):
        """
        Visualize a subgraph around a specific node.
        
        Args:
            center_node: Center node (e.g., "user_12" or "movie_456")
            depth: How many hops to include
        """
        nodes = {center_node}
        for _ in range(depth):
            new_nodes = set()
            for node in nodes:
                new_nodes.update(self.G.neighbors(node))
            nodes.update(new_nodes)
        
        subgraph = self.G.subgraph(nodes)
        
        node_colors = []
        for node in subgraph.nodes():
            if node.startswith('user'):
                node_colors.append('skyblue')
            else:
                node_colors.append('lightgreen')
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph)
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.2)
        nx.draw_networkx_labels(subgraph, pos, font_size=8)
        plt.title(f"Subgraph around {center_node}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    kg = MovieLensKnowledgeGraph(data_path='./ml-100k')
    
    kg.load_data()
    kg.build_knowledge_graph()
    
    user_id = 12
    recommendations = kg.get_personalized_recommendations(user_id)
    print(f"Personalized recommendations for user {user_id}:")
    for i, movie_id in enumerate(recommendations):
        movie_title = kg.movies_df[kg.movies_df['movie_id'] == movie_id]['title'].values[0]
        print(f"{i+1}. {movie_title}")
    
    graph_recommendations = kg.get_graph_based_recommendations(user_id)
    print(f"\nGraph-based recommendations for user {user_id}:")
    for i, movie_id in enumerate(graph_recommendations):
        movie_title = kg.movies_df[kg.movies_df['movie_id'] == movie_id]['title'].values[0]
        print(f"{i+1}. {movie_title}")
    
    kg.visualize_subgraph(f"user_{user_id}")


