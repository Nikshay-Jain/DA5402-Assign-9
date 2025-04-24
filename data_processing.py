import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, explode, split, regexp_replace

logger = logging.getLogger(__name__)

def clean_text(df, text_column):
    """
    Clean and preprocess text data
    
    Args:
        df: Spark DataFrame
        text_column: Name of the column containing text to clean
        
    Returns:
        DataFrame with cleaned text
    """
    try:
        logger.info(f"Cleaning text in '{text_column}' column...")
        
        # Handle missing values
        df = df.filter(col(text_column).isNotNull())
        
        # Convert to lowercase, remove special characters, and extra whitespaces
        cleaned_df = df.withColumn(
            f"cleaned_{text_column}",
            regexp_replace(
                regexp_replace(
                    lower(col(text_column)),
                    "[^a-zA-Z0-9\\s]", " "
                ),
                "\\s+", " "
            ).trim()
        )
        
        logger.info("Text cleaning completed")
        return cleaned_df
    
    except Exception as e:
        logger.error(f"Error during text cleaning: {e}")
        raise

def sample_data(df, fraction=0.1, seed=42):
    """
    Create a random sample of the data for testing purposes
    
    Args:
        df: Spark DataFrame
        fraction: Fraction of data to sample
        seed: Random seed for reproducibility
        
    Returns:
        Sampled DataFrame
    """
    try:
        logger.info(f"Sampling {fraction*100}% of data...")
        sampled_df = df.sample(fraction=fraction, seed=seed)
        logger.info(f"Sampled {sampled_df.count()} records from original {df.count()} records")
        return sampled_df
    
    except Exception as e:
        logger.error(f"Error during data sampling: {e}")
        raise

def handle_imbalanced_data(df, sentiment_column, majority_class, minority_class, balance_ratio=0.5):
    """
    Balance the dataset by downsampling the majority class
    
    Args:
        df: Spark DataFrame
        sentiment_column: Column containing sentiment labels
        majority_class: Label of the majority class
        minority_class: Label of the minority class
        balance_ratio: Target ratio of minority to majority class
        
    Returns:
        Balanced DataFrame
    """
    try:
        logger.info("Handling imbalanced data...")
        
        # Count each class
        majority_count = df.filter(col(sentiment_column) == majority_class).count()
        minority_count = df.filter(col(sentiment_column) == minority_class).count()
        
        logger.info(f"Original distribution - {majority_class}: {majority_count}, {minority_class}: {minority_count}")
        
        # Calculate the fraction to sample from majority class
        if majority_count > minority_count:
            fraction = (minority_count * (1/balance_ratio)) / majority_count
            
            # Sample majority class
            majority_df = df.filter(col(sentiment_column) == majority_class).sample(fraction=fraction, seed=42)
            minority_df = df.filter(col(sentiment_column) == minority_class)
            
            # Combine the dataframes
            balanced_df = majority_df.union(minority_df)
            
            new_majority_count = balanced_df.filter(col(sentiment_column) == majority_class).count()
            new_minority_count = balanced_df.filter(col(sentiment_column) == minority_class).count()
            
            logger.info(f"New distribution - {majority_class}: {new_majority_count}, {minority_class}: {new_minority_count}")
            
            return balanced_df
        else:
            logger.info("Data is already balanced or minority class is larger than majority class")
            return df
        
    except Exception as e:
        logger.error(f"Error during data balancing: {e}")
        raise

def extract_features(df, text_column, n_features=1000):
    """
    Extract features from text using TF-IDF
    
    Args:
        df: Spark DataFrame
        text_column: Column containing text
        n_features: Number of features to extract
        
    Returns:
        DataFrame with features
    """
    try:
        from pyspark.ml.feature import HashingTF, IDF, Tokenizer
        
        logger.info(f"Extracting features from '{text_column}' column...")
        
        # Tokenize text
        tokenizer = Tokenizer(inputCol=text_column, outputCol="words")
        words_df = tokenizer.transform(df)
        
        # Calculate term frequency
        hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=n_features)
        featurized_df = hashingTF.transform(words_df)
        
        # Calculate IDF
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        idf_model = idf.fit(featurized_df)
        featured_df = idf_model.transform(featurized_df)
        
        logger.info("Feature extraction completed")
        return featured_df
    
    except Exception as e:
        logger.error(f"Error during feature extraction: {e}")
        raise
