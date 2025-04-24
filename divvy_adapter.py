"""
Adapter script to convert Divvy Trips data to a format compatible with the sentiment analysis pipeline.
This is only needed if you're actually using Divvy Trips data instead of Amazon reviews data.
"""

import os
import logging
import argparse
import pandas as pd
from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("divvy_adapter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_nltk():
    """Download necessary NLTK data"""
    try:
        logger.info("Setting up NLTK resources...")
        nltk.download('vader_lexicon')
        logger.info("NLTK setup complete")
    except Exception as e:
        logger.error(f"Error setting up NLTK: {e}")
        raise

def load_divvy_data(file_path):
    """Load Divvy Trips data"""
    try:
        logger.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Total records: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def generate_text_from_trip_data(df):
    """
    Generate review-like text data from trip information
    
    This function creates artificial text descriptions based on trip attributes
    to demonstrate sentiment analysis. In a real application, you would have
    actual textual reviews/comments to analyze.
    """
    try:
        logger.info("Generating text descriptions from trip data...")
        
        # Create text descriptions based on trip duration and distance
        df['trip_duration'] = pd.to_datetime(df['ended_at']) - pd.to_datetime(df['started_at'])
        df['trip_duration_minutes'] = df['trip_duration'].dt.total_seconds() / 60
        
        # Define sentiment categories based on trip attributes
        conditions = [
            (df['trip_duration_minutes'] < 5),
            (df['trip_duration_minutes'] >= 5) & (df['trip_duration_minutes'] < 20),
            (df['trip_duration_minutes'] >= 20)
        ]
        
        descriptions = [
            "Short and quick bike ride. Convenient for getting around quickly.",
            "Medium length bike ride. Good exercise and practical transportation.",
            "Long bike ride. Great workout but tiring journey."
        ]
        
        df['reviewText'] = pd.np.select(conditions, descriptions, default="Normal bike ride.")
        
        # Add some variation based on membership type
        df.loc[df['member_casual'] == 'member', 'reviewText'] += " As a member, the service is reliable."
        df.loc[df['member_casual'] == 'casual', 'reviewText'] += " As a casual rider, the process was straightforward."
        
        # Generate a synthetic rating based on trip attributes
        # Short trips get high ratings, long trips get lower ratings (simulating user satisfaction)
        df['overall'] = 5 - (df['trip_duration_minutes'].clip(0, 60) / 15).astype(int)
        df['overall'] = df['overall'].clip(1, 5)
        
        logger.info("Text generation complete")
        return df
    except Exception as e:
        logger.error(f"Error generating text data: {e}")
        raise

def add_sentiment_annotation(df):
    """
    Add sentiment annotation to the dataset for evaluation purposes
    """
    try:
        logger.info("Adding sentiment annotation...")
        
        # Initialize the VADER sentiment analyzer
        sia = SentimentIntensityAnalyzer()
        
        # Apply sentiment analysis to generate ground truth
        df['sentiment_score'] = df['reviewText'].apply(lambda text: sia.polarity_scores(text)['compound'])
        
        # Convert scores to binary sentiment
        df['true_sentiment'] = df['sentiment_score'].apply(lambda score: "POSITIVE" if score >= 0 else "NEGATIVE")
        
        # Also create a sentiment based on rating (for comparison with Task 2)
        df['rating_sentiment'] = df['overall'].apply(lambda rating: "POSITIVE" if rating >= 3 else "NEGATIVE")
        
        logger.info("Sentiment annotation complete")
        return df
    except Exception as e:
        logger.error(f"Error adding sentiment annotation: {e}")
        raise

def save_adapted_data(df, output_path):
    """Save the adapted data to CSV"""
    try:
        logger.info(f"Saving adapted data to {output_path}...")
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved successfully")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def main():
    """Main function to adapt Divvy Trips data for sentiment analysis"""
    parser = argparse.ArgumentParser(description='Adapt Divvy Trips data for sentiment analysis')
    parser.add_argument('--input', type=str, required=True, help='Path to the Divvy Trips CSV file')
    parser.add_argument('--output', type=str, default='adapted_data.csv', help='Output file path')
    parser.add_argument('--sample', type=int, default=0, help='Sample size (0 for all data)')
    
    args = parser.parse_args()
    
    try:
        # Setup NLTK
        setup_nltk()
        
        # Load data
        df = load_divvy_data(args.input)
        
        # Take a sample if specified
        if args.sample > 0:
            logger.info(f"Sampling {args.sample} records...")
            df = df.sample(min(args.sample, len(df)), random_state=42)
        
        # Generate text and add sentiment
        df = generate_text_from_trip_data(df)
        df = add_sentiment_annotation(df)
        
        # Save the adapted data
        save_adapted_data(df, args.output)
        
        logger.info("Data adaptation completed successfully")
        
    except Exception as e:
        logger.error(f"Data adaptation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())