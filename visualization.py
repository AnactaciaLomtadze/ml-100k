import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
from datetime import datetime, timedelta

class VisualizationTool:
    def __init__(self, knowledge_graph, forgetting_mechanism=None):
        """
        Initialize the visualization tool.
        
        Args:
            knowledge_graph: The MovieLensKnowledgeGraph instance
            forgetting_mechanism: Optional ForgettingMechanism instance
        """
        self.kg = knowledge_graph
        self.fm = forgetting_mechanism
        
    def visualize_forgetting_impact(self, user_id, time_period=30, visualization_type='memory_strength'):
        """
        Visualize the impact of forgetting mechanism on a user's memory over time.
        
        Args:
            user_id: The user ID to visualize
            time_period: Number of days to simulate
            visualization_type: Type of visualization ('memory_strength', 'recommendations', 'graph')
        """
        if self.fm is None:
            print("Forgetting mechanism not provided. Cannot visualize forgetting impact.")
            return
        
        if visualization_type == 'memory_strength':
            self._visualize_memory_strength_decay(user_id, time_period)
        elif visualization_type == 'recommendations':
            self._visualize_recommendation_changes(user_id, time_period)
        elif visualization_type == 'graph':
            self._visualize_graph_changes(user_id)
        else:
            print(f"Unknown visualization type: {visualization_type}")
    
    def _visualize_memory_strength_decay(self, user_id, time_period):
        """Visualize how memory strength decays over time for a user's top movies."""
        # Get user's rated movies
        user_ratings = self.kg.ratings_df[self.kg.ratings_df['user_id'] == user_id]
        if user_ratings.empty:
            print(f"No ratings found for user {user_id}")
            return
        
        # Get top 5 movies by rating
        top_movies = user_ratings.sort_values('rating', ascending=False).head(5)
        movie_ids = top_movies['movie_id'].values
        movie_titles = [self.kg.movies_df[self.kg.movies_df['movie_id'] == mid]['title'].values[0][:20] 
                        for mid in movie_ids]
        
        # Simulate memory decay over time
        days = list(range(time_period + 1))
        memory_strengths = {mid: [] for mid in movie_ids}
        
        # Save original memory strengths
        original_strengths = {(user_id, mid): self.fm.memory_strength.get((user_id, mid), 0.5) 
                             for mid in movie_ids}
        
        # Get forgetting parameters
        params = self.fm.personalize_forgetting_parameters(user_id)
        
        # Simulate memory decay for each day
        for day in days:
            if day > 0:
                # Simulate advancing time
                current_time = datetime.now() - timedelta(days=(time_period - day))
                
                # Update last interaction time for simulation
                for mid in movie_ids:
                    self.fm.last_interaction_time[(user_id, mid)] = current_time.timestamp() - day * 86400
                
                # Apply hybrid decay
                self.fm.create_hybrid_decay_function(
                    user_id, 
                    time_weight=params['time_weight'],
                    usage_weight=params['usage_weight'],
                    novelty_weight=params['novelty_weight']
                )
            
            # Record memory strengths
            for mid in movie_ids:
                memory_strengths[mid].append(self.fm.memory_strength.get((user_id, mid), 0.5))
        
        # Restore original memory strengths
        for (u_id, mid), strength in original_strengths.items():
            self.fm.memory_strength[(u_id, mid)] = strength
        
        # Plot memory strength decay
        plt.figure(figsize=(12, 8))
        for i, mid in enumerate(movie_ids):
            plt.plot(days, memory_strengths[mid], marker='o', label=movie_titles[i])
        
        plt.title(f"Memory Strength Decay Over Time for User {user_id}")
        plt.xlabel("Days")
        plt.ylabel("Memory Strength")
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def _visualize_recommendation_changes(self, user_id, time_period):
        """Visualize how recommendations change over time with forgetting."""
        # Initialize forgetting-aware recommendation function
        params = self.fm.personalize_forgetting_parameters(user_id)
        forgetting_rec_fn = self.fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
            'personalized', params
        )
        
        # Get initial recommendations without forgetting
        initial_recs = self.kg.get_personalized_recommendations(user_id, n=10)
        initial_rec_titles = [self.kg.movies_df[self.kg.movies_df['movie_id'] == mid]['title'].values[0][:30] 
                              for mid in initial_recs]
        
        # Sample points over time
        sample_days = [0, time_period // 4, time_period // 2, 3 * time_period // 4, time_period]
        
        # Save original memory strengths
        user_movies = [(u_id, m_id) for (u_id, m_id) in self.fm.memory_strength.keys() if u_id == user_id]
        original_strengths = {key: self.fm.memory_strength[key] for key in user_movies}
        original_times = {key: self.fm.last_interaction_time.get(key, 0) for key in user_movies}
        
        # Track recommendation changes
        rec_changes = []
        
        for day in sample_days:
            # Simulate advancing time for all user-movie interactions
            current_time = datetime.now().timestamp()
            for u_m_key in user_movies:
                original_time = original_times[u_m_key]
                self.fm.last_interaction_time[u_m_key] = original_time - (time_period - day) * 86400
            
            # Get recommendations with forgetting at this point
            forgetting_recs = forgetting_rec_fn(user_id, n=10)
            forgetting_rec_titles = [self.kg.movies_df[self.kg.movies_df['movie_id'] == mid]['title'].values[0][:30] 
                                     for mid in forgetting_recs]
            
            # Record recommendation changes
            rec_changes.append({
                'day': day,
                'recommendations': forgetting_rec_titles
            })
        
        # Restore original memory strengths and times
        for key in user_movies:
            self.fm.memory_strength[key] = original_strengths[key]
            self.fm.last_interaction_time[key] = original_times[key]
        
        # Plot recommendation changes
        plt.figure(figsize=(14, 10))
        
        # Create a visualization showing how recommendations change
        plt.subplot(1, 2, 1)
        plt.title(f"Initial Recommendations for User {user_id}")
        plt.barh(range(len(initial_rec_titles)), [1]*len(initial_rec_titles))
        plt.yticks(range(len(initial_rec_titles)), initial_rec_titles)
        plt.xlabel('Recommendation Score')
        
        # Create a visualization showing final recommendations
        plt.subplot(1, 2, 2)
        plt.title(f"Recommendations After {time_period} Days")
        plt.barh(range(len(rec_changes[-1]['recommendations'])), 
                [1]*len(rec_changes[-1]['recommendations']))
        plt.yticks(range(len(rec_changes[-1]['recommendations'])), 
                  rec_changes[-1]['recommendations'])
        plt.xlabel('Recommendation Score')
        
        plt.tight_layout()
        plt.show()
        
        # Create a table showing the evolution of recommendations
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.axis('off')
        ax.axis('tight')
        
        # Create table data
        table_data = []
        for i in range(len(rec_changes)):
            row = [f"Day {rec_changes[i]['day']}"] + rec_changes[i]['recommendations'][:5]
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, 
                        colLabels=["Time"] + [f"Rec {i+1}" for i in range(5)],
                        loc='center', cellLoc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        plt.title(f"Evolution of Recommendations Over Time for User {user_id}")
        plt.tight_layout()
        plt.show()
    
    def _visualize_graph_changes(self, user_id):
        """Visualize how the knowledge graph changes with forgetting mechanism."""
        if self.fm is None:
            print("Forgetting mechanism not provided. Cannot visualize graph changes.")
            return
        
        # Create a copy of the current graph
        G_before = self.kg.G.copy()
        
        # Apply forgetting to user's interactions
        params = self.fm.personalize_forgetting_parameters(user_id)
        self.fm.create_hybrid_decay_function(
            user_id, 
            time_weight=params['time_weight'],
            usage_weight=params['usage_weight'],
            novelty_weight=params['novelty_weight']
        )
        
        # Create a modified graph with edge weights adjusted by memory strength
        G_after = self.kg.G.copy()
        
        # Adjust edge weights based on memory strength
        user_node = f"user_{user_id}"
        for movie_node in G_after.neighbors(user_node):
            if movie_node.startswith("movie_"):
                movie_id = int(movie_node.split("_")[1])
                memory_strength = self.fm.memory_strength.get((user_id, movie_id), 0.5)
                
                # Update edge weight based on memory strength
                edge_data = G_after.get_edge_data(user_node, movie_node)
                if edge_data and 'weight' in edge_data:
                    G_after[user_node][movie_node]['weight'] = edge_data['weight'] * memory_strength
        
        # Extract subgraph around user
        nodes_before = {user_node}
        nodes_after = {user_node}
        
        # Get 1-hop neighborhood
        for node in self.kg.G.neighbors(user_node):
            nodes_before.add(node)
            nodes_after.add(node)
        
        # Get 2-hop neighborhood (movie-movie connections)
        for movie_node in list(nodes_before):
            if movie_node.startswith("movie_"):
                for neighbor in self.kg.G.neighbors(movie_node):
                    if neighbor.startswith("movie_"):
                        nodes_before.add(neighbor)
                        nodes_after.add(neighbor)
        
        # Create subgraphs
        subgraph_before = G_before.subgraph(nodes_before)
        subgraph_after = G_after.subgraph(nodes_after)
        
        # Visualize both graphs
        plt.figure(figsize=(18, 8))
        
        # Plot original graph
        plt.subplot(1, 2, 1)
        pos_before = nx.spring_layout(subgraph_before, seed=42)
        
        # Draw nodes with different colors
        node_colors = []
        for node in subgraph_before.nodes():
            if node == user_node:
                node_colors.append('red')
            elif node.startswith('user'):
                node_colors.append('skyblue')
            else:
                node_colors.append('lightgreen')
        
        # Draw edges with width proportional to weight
        edge_widths = [subgraph_before[u][v].get('weight', 1.0) * 1.5 for u, v in subgraph_before.edges()]
        
        nx.draw_networkx_nodes(subgraph_before, pos_before, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(subgraph_before, pos_before, width=edge_widths, alpha=0.5)
        nx.draw_networkx_labels(subgraph_before, pos_before, font_size=8)
        plt.title(f"User {user_id} Subgraph Before Forgetting")
        plt.axis('off')
        
        # Plot graph after forgetting
        plt.subplot(1, 2, 2)
        # Use same layout for comparison
        pos_after = pos_before
        
        # Draw nodes with different colors (same as before)
        node_colors_after = []
        for node in subgraph_after.nodes():
            if node == user_node:
                node_colors_after.append('red')
            elif node.startswith('user'):
                node_colors_after.append('skyblue')
            else:
                node_colors_after.append('lightgreen')
        
        # Draw edges with width proportional to weight after forgetting
        edge_widths_after = [subgraph_after[u][v].get('weight', 1.0) * 1.5 for u, v in subgraph_after.edges()]
        
        nx.draw_networkx_nodes(subgraph_after, pos_after, node_color=node_colors_after, alpha=0.8)
        nx.draw_networkx_edges(subgraph_after, pos_after, width=edge_widths_after, alpha=0.5)
        nx.draw_networkx_labels(subgraph_after, pos_after, font_size=8)
        plt.title(f"User {user_id} Subgraph After Forgetting")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Visualize edge weight changes in a heatmap
        shared_edges = []
        weights_before = []
        weights_after = []
        
        # Get edges connected to the user node
        user_movies_before = [(user_node, m) for m in subgraph_before.neighbors(user_node) if m.startswith("movie_")]
        
        for edge in user_movies_before:
            u, v = edge
            movie_id = int(v.split("_")[1])
            movie_title = self.kg.movies_df[self.kg.movies_df['movie_id'] == movie_id]['title'].values[0][:20]
            
            w_before = subgraph_before[u][v].get('weight', 1.0)
            w_after = subgraph_after[u][v].get('weight', 1.0)
            
            shared_edges.append(movie_title)
            weights_before.append(w_before)
            weights_after.append(w_after)
        
        # Create a comparison bar chart of edge weights
        plt.figure(figsize=(14, 8))
        x = np.arange(len(shared_edges))
        width = 0.35
        
        plt.bar(x - width/2, weights_before, width, label='Before Forgetting')
        plt.bar(x + width/2, weights_after, width, label='After Forgetting')
        
        plt.xlabel('Movies')
        plt.ylabel('Edge Weight')
        plt.title(f'Edge Weight Changes Due to Forgetting for User {user_id}')
        plt.xticks(x, shared_edges, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def visualize_memory_distribution(self, user_id=None):
        """
        Visualize distribution of memory strengths across users or for a specific user.
        
        Args:
            user_id: Optional specific user to visualize
        """
        if self.fm is None:
            print("Forgetting mechanism not provided. Cannot visualize memory distribution.")
            return
        
        memory_strengths = []
        user_ids = []
        movie_ratings = []
        
        if user_id is not None:
            # Get data for specific user
            for (u_id, m_id), strength in self.fm.memory_strength.items():
                if u_id == user_id:
                    memory_strengths.append(strength)
                    movie_ratings.append(self.kg.ratings_df[
                        (self.kg.ratings_df['user_id'] == u_id) & 
                        (self.kg.ratings_df['movie_id'] == m_id)
                    ]['rating'].values[0] if not self.kg.ratings_df[
                        (self.kg.ratings_df['user_id'] == u_id) & 
                        (self.kg.ratings_df['movie_id'] == m_id)
                    ].empty else 0)
            
            # Scatter plot of memory strength vs. rating
            plt.figure(figsize=(10, 6))
            plt.scatter(movie_ratings, memory_strengths, alpha=0.6)
            plt.xlabel('Movie Rating')
            plt.ylabel('Memory Strength')
            plt.title(f'Memory Strength vs. Rating for User {user_id}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            # Get data for all users
            for (u_id, _), strength in self.fm.memory_strength.items():
                memory_strengths.append(strength)
                user_ids.append(u_id)
            
            # Create a boxplot of memory strengths by user
            plt.figure(figsize=(14, 8))
            df = pd.DataFrame({'user_id': user_ids, 'memory_strength': memory_strengths})
            
            # If too many users, sample a subset
            if df['user_id'].nunique() > 20:
                user_sample = np.random.choice(df['user_id'].unique(), 20, replace=False)
                df = df[df['user_id'].isin(user_sample)]
            
            sns.boxplot(x='user_id', y='memory_strength', data=df)
            plt.xlabel('User ID')
            plt.ylabel('Memory Strength')
            plt.title('Distribution of Memory Strengths Across Users')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def compare_forgetting_strategies(self, user_id, n=10):
        """
        Compare different forgetting strategies for a user.
        
        Args:
            user_id: The user ID to analyze
            n: Number of recommendations to consider
        """
        if self.fm is None:
            print("Forgetting mechanism not provided. Cannot compare strategies.")
            return
        
        # Store original memory strengths
        original_strengths = {}
        for key, value in self.fm.memory_strength.items():
            if key[0] == user_id:
                original_strengths[key] = value
        
        # Get baseline recommendations (no forgetting)
        baseline_recs = self.kg.get_personalized_recommendations(user_id, n=n)
        baseline_titles = [self.kg.movies_df[self.kg.movies_df['movie_id'] == mid]['title'].values[0][:30] 
                          for mid in baseline_recs]
        
        strategies = {
            'Time-based': {'decay_parameter': 0.1},
            'Usage-based': {'interaction_threshold': 3},
            'Hybrid (default)': {'time_weight': 0.4, 'usage_weight': 0.3, 'novelty_weight': 0.3},
            'Hybrid (time-heavy)': {'time_weight': 0.7, 'usage_weight': 0.2, 'novelty_weight': 0.1},
            'Hybrid (novelty-heavy)': {'time_weight': 0.2, 'usage_weight': 0.3, 'novelty_weight': 0.5},
            'Personalized': self.fm.personalize_forgetting_parameters(user_id)
        }
        
        all_recommendations = {}
        
        # Apply each strategy and get recommendations
        for strategy_name, params in strategies.items():
            # Apply strategy
            if strategy_name == 'Time-based':
                self.fm.implement_time_based_decay(user_id, params['decay_parameter'])
            elif strategy_name == 'Usage-based':
                self.fm.implement_usage_based_decay(user_id, params['interaction_threshold'])
            elif strategy_name.startswith('Hybrid'):
                self.fm.create_hybrid_decay_function(
                    user_id,
                    time_weight=params['time_weight'],
                    usage_weight=params['usage_weight'],
                    novelty_weight=params['novelty_weight']
                )
            elif strategy_name == 'Personalized':
                self.fm.create_hybrid_decay_function(
                    user_id,
                    time_weight=params['time_weight'],
                    usage_weight=params['usage_weight'],
                    novelty_weight=params['novelty_weight']
                )
            
            # Get recommendations after applying the strategy
            forgetting_rec_fn = self.fm.integrate_forgetting_mechanism_into_recommendation_pipeline(
                'personalized', params
            )
            recs = forgetting_rec_fn(user_id, n=n)
            titles = [self.kg.movies_df[self.kg.movies_df['movie_id'] == mid]['title'].values[0][:30] 
                     for mid in recs]
            
            all_recommendations[strategy_name] = titles
            
            # Reset memory strengths to original values
            for key, value in original_strengths.items():
                self.fm.memory_strength[key] = value
        
        # Create a visualization to compare strategies
        plt.figure(figsize=(15, 10))
        
        # Create a set of all unique recommendations
        all_unique_recs = set()
        for recs in all_recommendations.values():
            all_unique_recs.update(recs)
        all_unique_recs = list(all_unique_recs)
        
        # Create a matrix of recommendation ranks
        strategy_names = list(strategies.keys()) + ['Baseline']
        all_recommendations['Baseline'] = baseline_titles
        
        # Create a heatmap
        heatmap_data = np.zeros((len(all_unique_recs), len(strategy_names)))
        
        for i, strategy in enumerate(strategy_names):
            recs = all_recommendations[strategy]
            for j, title in enumerate(all_unique_recs):
                if title in recs:
                    rank = recs.index(title) + 1
                    heatmap_data[j, i] = n - rank + 1  # Higher score for higher rank
                else:
                    heatmap_data[j, i] = 0
        
        # Sort rows by total score
        row_sums = np.sum(heatmap_data, axis=1)
        sorted_indices = np.argsort(-row_sums)
        heatmap_data = heatmap_data[sorted_indices]
        all_unique_recs = [all_unique_recs[i] for i in sorted_indices]
        
        # Create heatmap
        plt.figure(figsize=(12, len(all_unique_recs) * 0.4))
        ax = sns.heatmap(heatmap_data, cmap='YlGnBu', 
                         xticklabels=strategy_names, 
                         yticklabels=all_unique_recs,
                         linewidths=.5)
        
        plt.title(f'Recommendation Comparison Across Forgetting Strategies for User {user_id}')
        plt.tight_layout()
        plt.show()

        