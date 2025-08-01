#!/usr/bin/env python3
"""
Visualization module for IMDB RPA Sentiment Analysis Tool

This module provides various visualization capabilities for analyzing and presenting
the results of the IMDB sentiment analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SentimentVisualizer:
    """Handles visualization of sentiment analysis results."""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style for matplotlib
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_sentiment_distribution_plot(self, df: pd.DataFrame, save_path: str = None) -> None:
        """Create a distribution plot of sentiment scores."""
        try:
            # Filter out failed movies
            successful_df = df[df['error_message'].isna()]
            
            if successful_df.empty:
                logger.warning("No successful results to visualize")
                return
            
            plt.figure(figsize=(12, 8))
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('IMDB Movie Sentiment Analysis Results', fontsize=16, fontweight='bold')
            
            # 1. Sentiment Score Distribution
            axes[0, 0].hist(successful_df['avg_sentiment'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Distribution of Sentiment Scores')
            axes[0, 0].set_xlabel('Average Sentiment Score')
            axes[0, 0].set_ylabel('Number of Movies')
            axes[0, 0].axvline(successful_df['avg_sentiment'].mean(), color='red', linestyle='--', 
                             label=f'Mean: {successful_df["avg_sentiment"].mean():.3f}')
            axes[0, 0].legend()
            
            # 2. Sentiment vs Rating Scatter Plot
            axes[0, 1].scatter(successful_df['rating'].astype(float), successful_df['avg_sentiment'], 
                             alpha=0.6, s=50)
            axes[0, 1].set_title('Sentiment Score vs IMDB Rating')
            axes[0, 1].set_xlabel('IMDB Rating')
            axes[0, 1].set_ylabel('Average Sentiment Score')
            
            # Add trend line
            z = np.polyfit(successful_df['rating'].astype(float), successful_df['avg_sentiment'], 1)
            p = np.poly1d(z)
            axes[0, 1].plot(successful_df['rating'].astype(float), p(successful_df['rating'].astype(float)), 
                           "r--", alpha=0.8)
            
            # 3. Positive vs Negative Reviews
            axes[1, 0].bar(['Positive Reviews', 'Negative Reviews'], 
                          [successful_df['positive_reviews'].sum(), successful_df['negative_reviews'].sum()],
                          color=['green', 'red'], alpha=0.7)
            axes[1, 0].set_title('Total Positive vs Negative Reviews')
            axes[1, 0].set_ylabel('Number of Reviews')
            
            # 4. Processing Time Distribution
            axes[1, 1].hist(successful_df['processing_time'], bins=15, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 1].set_title('Processing Time Distribution')
            axes[1, 1].set_xlabel('Processing Time (seconds)')
            axes[1, 1].set_ylabel('Number of Movies')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Sentiment distribution plot saved to: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error creating sentiment distribution plot: {e}")
    
    def create_top_movies_chart(self, df: pd.DataFrame, top_n: int = 10, save_path: str = None) -> None:
        """Create a bar chart of top movies by sentiment score."""
        try:
            # Filter out failed movies and sort by sentiment
            successful_df = df[df['error_message'].isna()].copy()
            
            if successful_df.empty:
                logger.warning("No successful results to visualize")
                return
            
            # Sort by sentiment score and get top N
            top_movies = successful_df.nlargest(top_n, 'avg_sentiment')
            
            plt.figure(figsize=(14, 8))
            
            # Create horizontal bar chart
            bars = plt.barh(range(len(top_movies)), top_movies['avg_sentiment'], 
                          color=plt.cm.viridis(np.linspace(0, 1, len(top_movies))))
            
            # Customize the chart
            plt.yticks(range(len(top_movies)), [f"{title} ({year})" for title, year in 
                                              zip(top_movies['title'], top_movies['year'])])
            plt.xlabel('Average Sentiment Score')
            plt.title(f'Top {top_n} Movies by Sentiment Score', fontsize=16, fontweight='bold')
            
            # Add value labels on bars
            for i, (bar, sentiment) in enumerate(zip(bars, top_movies['avg_sentiment'])):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{sentiment:.3f}', va='center', fontweight='bold')
            
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Top movies chart saved to: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error creating top movies chart: {e}")
    
    def create_interactive_dashboard(self, df: pd.DataFrame, save_path: str = None) -> None:
        """Create an interactive Plotly dashboard."""
        try:
            # Filter out failed movies
            successful_df = df[df['error_message'].isna()].copy()
            
            if successful_df.empty:
                logger.warning("No successful results to visualize")
                return
            
            # Convert rating to numeric
            successful_df['rating'] = pd.to_numeric(successful_df['rating'], errors='coerce')
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Sentiment Score Distribution', 'Sentiment vs Rating', 
                              'Reviews Analysis', 'Processing Performance'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Sentiment Score Distribution
            fig.add_trace(
                go.Histogram(x=successful_df['avg_sentiment'], nbinsx=20, name='Sentiment Distribution',
                           marker_color='skyblue'),
                row=1, col=1
            )
            
            # 2. Sentiment vs Rating Scatter
            fig.add_trace(
                go.Scatter(x=successful_df['rating'], y=successful_df['avg_sentiment'],
                          mode='markers', name='Movies',
                          marker=dict(size=8, color=successful_df['avg_sentiment'], 
                                    colorscale='Viridis', showscale=True),
                          text=successful_df['title'],
                          hovertemplate='<b>%{text}</b><br>Rating: %{x}<br>Sentiment: %{y:.3f}<extra></extra>'),
                row=1, col=2
            )
            
            # 3. Reviews Analysis
            fig.add_trace(
                go.Bar(x=['Positive', 'Negative'], 
                      y=[successful_df['positive_reviews'].sum(), successful_df['negative_reviews'].sum()],
                      name='Review Counts',
                      marker_color=['green', 'red']),
                row=2, col=1
            )
            
            # 4. Processing Time
            fig.add_trace(
                go.Box(y=successful_df['processing_time'], name='Processing Time',
                      marker_color='orange'),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title_text="IMDB Sentiment Analysis Dashboard",
                showlegend=False,
                height=800
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Sentiment Score", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            
            fig.update_xaxes(title_text="IMDB Rating", row=1, col=2)
            fig.update_yaxes(title_text="Sentiment Score", row=1, col=2)
            
            fig.update_xaxes(title_text="Review Type", row=2, col=1)
            fig.update_yaxes(title_text="Count", row=2, col=1)
            
            fig.update_xaxes(title_text="", row=2, col=2)
            fig.update_yaxes(title_text="Processing Time (seconds)", row=2, col=2)
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Interactive dashboard saved to: {save_path}")
            else:
                fig.show()
                
        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {e}")
    
    def create_word_cloud_data(self, df: pd.DataFrame) -> Dict[str, int]:
        """Prepare data for word cloud visualization."""
        try:
            # This would require the actual review text data
            # For now, we'll create a simple frequency based on movie titles
            word_freq = {}
            
            for title in df['title']:
                words = title.lower().split()
                for word in words:
                    if len(word) > 2:  # Filter out short words
                        word_freq[word] = word_freq.get(word, 0) + 1
            
            return word_freq
            
        except Exception as e:
            logger.error(f"Error creating word cloud data: {e}")
            return {}
    
    def create_performance_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate and return performance metrics."""
        try:
            successful_df = df[df['error_message'].isna()]
            
            if successful_df.empty:
                return {}
            
            metrics = {
                'total_movies': len(df),
                'successful_movies': len(successful_df),
                'success_rate': len(successful_df) / len(df) * 100,
                'avg_sentiment': successful_df['avg_sentiment'].mean(),
                'sentiment_std': successful_df['avg_sentiment'].std(),
                'avg_processing_time': successful_df['processing_time'].mean(),
                'total_reviews': successful_df['num_reviews'].sum(),
                'avg_reviews_per_movie': successful_df['num_reviews'].mean(),
                'positive_review_ratio': successful_df['positive_reviews'].sum() / 
                                       (successful_df['positive_reviews'].sum() + successful_df['negative_reviews'].sum()) * 100
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def generate_all_visualizations(self, csv_path: str) -> None:
        """Generate all visualizations from a CSV file."""
        try:
            # Load data
            df = pd.read_csv(csv_path)
            
            # Generate timestamp for file names
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            # Create all visualizations
            self.create_sentiment_distribution_plot(
                df, 
                self.output_dir / f"sentiment_distribution_{timestamp}.png"
            )
            
            self.create_top_movies_chart(
                df, 
                save_path=self.output_dir / f"top_movies_{timestamp}.png"
            )
            
            self.create_interactive_dashboard(
                df, 
                save_path=self.output_dir / f"dashboard_{timestamp}.html"
            )
            
            # Generate performance metrics
            metrics = self.create_performance_metrics(df)
            
            # Save metrics to JSON
            metrics_path = self.output_dir / f"performance_metrics_{timestamp}.json"
            import json
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"All visualizations generated successfully in {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")

def main():
    """Main function for standalone visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate visualizations for IMDB sentiment analysis")
    parser.add_argument("csv_file", help="Path to the CSV file with results")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Create visualizer and generate all visualizations
    visualizer = SentimentVisualizer(args.output_dir)
    visualizer.generate_all_visualizations(args.csv_file)

if __name__ == "__main__":
    main() 