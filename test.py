"""
Test script to verify the sentiment analysis pipeline
"""

import os
import unittest
import tempfile
import pandas as pd
import logging
from pyspark.sql import SparkSession
from transformers import pipeline

# Import our modules
import sys
sys.path.append('.')  # Add current directory to path
from main import initialize_spark, load_data, task1_sentiment_analysis, task2_evaluation
from data_processing import clean_text, sample_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSentimentAnalysisPipeline(unittest.TestCase):
    """Test cases for the sentiment analysis pipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Create a small test dataset
        cls.test_data = pd.DataFrame({
            'reviewText': [
                "This product is amazing! I love it so much.",
                "Terrible experience. Would not recommend.",
                "It's okay, nothing special but does the job.",
                "Absolutely fantastic product, exceeded my expectations!",
                "Complete waste of money. Avoid at all costs."
            ],
            'overall': [5.0, 1.0, 3.0, 5.0, 1.0]
        })
        
        # Create a temporary file for the test data
        fd, cls.test_file = tempfile.mkstemp(suffix='.csv')
        os.close(fd)
        cls.test_data.to_csv(cls.test_file, index=False)
        
        # Initialize Spark session
        cls.spark = SparkSession.builder \
            .appName("TestSentimentAnalysis") \
            .master("local[*]") \
            .getOrCreate()
        
        # Load test data as Spark DataFrame
        cls.df = cls.spark.read.csv(cls.test_file, header=True, inferSchema=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        # Stop Spark session
        cls.spark.stop()
        
        # Remove temporary file
        if os.path.exists(cls.test_file):
            os.remove(cls.test_file)
    
    def test_initialize_spark(self):
        """Test Spark initialization"""
        spark = initialize_spark()
        self.assertIsNotNone(spark)
        self.assertTrue(isinstance(spark, SparkSession))
    
    def test_load_data(self):
        """Test data loading"""
        df = load_data(self.spark, self.test_file)
        self.assertEqual(df.count(), 5)
        self.assertTrue('reviewText' in df.columns)
        self.assertTrue('overall' in df.columns)
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis"""
        result_df = task1_sentiment_analysis(self.spark, self.df)
        self.assertTrue('predicted_sentiment' in result_df.columns)
        
        # Convert to Pandas for easier testing
        result_pd = result_df.toPandas()
        
        # Verify positive sentiment for positive reviews
        positive_reviews = result_pd[result_pd['reviewText'].str.contains('amazing|fantastic')]
        self.assertTrue(all(positive_reviews['predicted_sentiment'] == 'POSITIVE'))
        
        # Verify negative sentiment for negative reviews
        negative_reviews = result_pd[result_pd['reviewText'].str.contains('Terrible|waste')]
        self.assertTrue(all(negative_reviews['predicted_sentiment'] == 'NEGATIVE'))
    
    def test_evaluation(self):
        """Test model evaluation"""
        # First run sentiment analysis
        df_with_sentiment = task1_sentiment_analysis(self.spark, self.df)
        
        # Then evaluate
        eval_results = task2_evaluation(self.spark, df_with_sentiment)
        
        # Check that we have all the expected output
        self.assertIn('confusion_matrix', eval_results)
        self.assertIn('precision', eval_results)
        self.assertIn('recall', eval_results)
        self.assertIn('evaluated_df', eval_results)
        
        # Precision and recall should be between 0 and 1
        self.assertTrue(0 <= eval_results['precision'] <= 1)
        self.assertTrue(0 <= eval_results['recall'] <= 1)
        
        # Check confusion matrix dimensions
        self.assertEqual(eval_results['confusion_matrix'].shape, (2, 2))

    def test_data_processing(self):
        """Test data processing utilities"""
        # Test clean_text
        cleaned_df = clean_text(self.df, 'reviewText')
        self.assertTrue('cleaned_reviewText' in cleaned_df.columns)
        
        # Test sample_data
        sampled_df = sample_data(self.df, fraction=0.5)
        # The test dataset is small, so it might sample all or none
        self.assertTrue(0 <= sampled_df.count() <= self.df.count())

def run_tests():
    """Run the test suite"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSentimentAnalysisPipeline)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == "__main__":
    run_tests()
