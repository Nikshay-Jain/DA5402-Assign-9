import os
import time
import logging
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import StringType
from transformers import pipeline
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def initialize_spark():
    """Initialize and return a Spark session"""
    try:
        logger.info("Initializing Spark session...")
        spark = SparkSession.builder \
            .appName("AmazonReviewsSentimentAnalysis") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
        logger.info("Spark session initialized successfully")
        return spark
    except Exception as e:
        logger.error(f"Failed to initialize Spark session: {e}")
        raise

def load_data(spark, file_path):
    """Load the Amazon reviews dataset"""
    try:
        logger.info(f"Loading data from {file_path}...")
        if file_path.endswith('.csv'):
            df = spark.read.csv(file_path, header=True, inferSchema=True)
        elif file_path.endswith('.json'):
            df = spark.read.json(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide CSV or JSON file.")
        
        logger.info(f"Data loaded successfully. Total records: {df.count()}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def task1_sentiment_analysis(spark, df):
    """
    Task 1: Apply sentiment analysis on reviews using transformers pipeline
    """
    logger.info("Starting Task 1: Sentiment Analysis using Transformers")
    
    try:
        # Initialize the sentiment analysis pipeline
        logger.info("Initializing sentiment analysis pipeline...")
        sentiment_analyzer = pipeline("sentiment-analysis")
        
        # Define UDF for sentiment analysis
        @udf(StringType())
        def analyze_sentiment(text):
            try:
                if text is None or len(str(text).strip()) == 0:
                    return "NEUTRAL"
                result = sentiment_analyzer(str(text))
                return "POSITIVE" if result[0]['label'] == 'POSITIVE' else "NEGATIVE"
            except Exception as e:
                logger.warning(f"Error analyzing sentiment for text: {e}")
                return "NEUTRAL"
        
        # Assuming 'reviewText' is the column containing review content
        # Check if the column exists
        if 'reviewText' not in df.columns:
            potential_columns = [col for col in df.columns if 'review' in col.lower() or 'text' in col.lower()]
            if potential_columns:
                review_column = potential_columns[0]
                logger.warning(f"Column 'reviewText' not found. Using '{review_column}' instead.")
            else:
                logger.error("No suitable review text column found in the dataset")
                raise ValueError("No review text column found in the dataset")
        else:
            review_column = 'reviewText'
        
        # Apply sentiment analysis to the review text
        logger.info(f"Applying sentiment analysis to the '{review_column}' column...")
        result_df = df.withColumn("predicted_sentiment", analyze_sentiment(col(review_column)))
        
        logger.info("Task 1 completed successfully")
        return result_df
        
    except Exception as e:
        logger.error(f"Task 1 failed: {e}")
        raise

def task2_evaluation(spark, df):
    """
    Task 2: Evaluate the sentiment analysis model
    """
    logger.info("Starting Task 2: Model Evaluation")
    
    try:
        # Determine which column contains the star rating
        rating_column = None
        for col_name in df.columns:
            if 'rating' in col_name.lower() or 'star' in col_name.lower() or 'overall' in col_name.lower():
                rating_column = col_name
                break
        
        if rating_column is None:
            logger.error("No suitable rating column found in the dataset")
            raise ValueError("No rating column found in the dataset")
        
        logger.info(f"Using '{rating_column}' as the rating column")
        
        # Convert ratings to true sentiment labels (POSITIVE if rating >= 3.0, else NEGATIVE)
        logger.info("Converting ratings to true sentiment labels...")
        df_with_true = df.withColumn(
            "true_sentiment", 
            when(col(rating_column) >= 3.0, "POSITIVE").otherwise("NEGATIVE")
        )
        
        # Count the occurrences in each category for confusion matrix
        logger.info("Computing confusion matrix and metrics...")
        confusion_data = df_with_true.select("predicted_sentiment", "true_sentiment").toPandas()
        
        # Filter out any NEUTRAL predictions for evaluation purposes
        confusion_data = confusion_data[confusion_data['predicted_sentiment'] != "NEUTRAL"]
        
        # Create confusion matrix
        cm = confusion_matrix(
            confusion_data['true_sentiment'], 
            confusion_data['predicted_sentiment'],
            labels=["POSITIVE", "NEGATIVE"]
        )
        
        # Calculate precision and recall
        precision = precision_score(
            confusion_data['true_sentiment'], 
            confusion_data['predicted_sentiment'],
            pos_label="POSITIVE"
        )
        
        recall = recall_score(
            confusion_data['true_sentiment'], 
            confusion_data['predicted_sentiment'],
            pos_label="POSITIVE"
        )
        
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        
        # Create a DataFrame for confusion matrix
        conf_matrix_df = pd.DataFrame(
            cm,
            index=['True POSITIVE', 'True NEGATIVE'],
            columns=['Predicted POSITIVE', 'Predicted NEGATIVE']
        )
        
        # Return results
        return {
            "confusion_matrix": conf_matrix_df,
            "precision": precision,
            "recall": recall,
            "evaluated_df": df_with_true
        }
        
    except Exception as e:
        logger.error(f"Task 2 failed: {e}")
        raise

def save_results(results, output_dir="output"):
    """Save the results to output directory"""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save confusion matrix
        results["confusion_matrix"].to_csv(f"{output_dir}/confusion_matrix.csv")
        
        # Save metrics
        with open(f"{output_dir}/metrics.txt", "w") as f:
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
        
        # Save a sample of the evaluated dataframe
        results["evaluated_df"].limit(1000).toPandas().to_csv(f"{output_dir}/sample_results.csv", index=False)
        
        logger.info(f"Results saved to {output_dir} directory")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

def main():
    """Main function to run the sentiment analysis pipeline"""
    parser = argparse.ArgumentParser(description='Amazon Reviews Sentiment Analysis')
    parser.add_argument('--input', type=str, required=True, help='Path to the input file')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        # Initialize Spark
        spark = initialize_spark()
        df = load_data(spark, args.input)
        df_with_sentiment = task1_sentiment_analysis(spark, df)
        evaluation_results = task2_evaluation(spark, df_with_sentiment)
        save_results(evaluation_results, args.output)
        
        logger.info(f"Pipeline completed successfully in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1
    finally:
        # Stop Spark session
        if 'spark' in locals():
            spark.stop()
            logger.info("Spark session stopped")
    
    return 0

if __name__ == "__main__":
    exit(main())