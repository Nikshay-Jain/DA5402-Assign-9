import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_true, y_pred, labels=None, output_path=None):
    """
    Plot confusion matrix and save to file
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        output_path: Path to save the plot
    """
    try:
        logger.info("Plotting confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        
        plt.figure(figsize=(10, 8))
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title('Sentiment Analysis Confusion Matrix')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Confusion matrix saved to {output_path}")
        
        return disp
    
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")
        raise

def plot_sentiment_distribution(df, output_path=None):
    """
    Plot the distribution of sentiment labels
    
    Args:
        df: Pandas DataFrame with sentiment columns
        output_path: Path to save the plot
    """
    try:
        logger.info("Plotting sentiment distribution...")
        
        plt.figure(figsize=(12, 6))
        
        # Create subplots for true and predicted sentiment
        plt.subplot(1, 2, 1)
        true_counts = df['true_sentiment'].value_counts()
        sns.barplot(x=true_counts.index, y=true_counts.values)
        plt.title('True Sentiment Distribution')
        plt.ylabel('Count')
        plt.xlabel('Sentiment')
        
        plt.subplot(1, 2, 2)
        pred_counts = df['predicted_sentiment'].value_counts()
        sns.barplot(x=pred_counts.index, y=pred_counts.values)
        plt.title('Predicted Sentiment Distribution')
        plt.ylabel('Count')
        plt.xlabel('Sentiment')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Sentiment distribution plot saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error plotting sentiment distribution: {e}")
        raise

def plot_rating_vs_sentiment(df, rating_column='overall', output_path=None):
    """
    Plot the relationship between rating and sentiment
    
    Args:
        df: Pandas DataFrame with rating and sentiment columns
        rating_column: Name of the rating column
        output_path: Path to save the plot
    """
    try:
        logger.info("Plotting rating vs sentiment...")
        
        plt.figure(figsize=(12, 6))
        
        # Count sentiment by rating
        sentiment_by_rating = pd.crosstab(df[rating_column], df['predicted_sentiment'])
        
        # Plot stacked bars
        sentiment_by_rating.plot(kind='bar', stacked=True)
        plt.title('Predicted Sentiment by Rating')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Sentiment')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Rating vs sentiment plot saved to {output_path}")
            
    except Exception as e:
        logger.error(f"Error plotting rating vs sentiment: {e}")
        raise

def plot_metrics_comparison(results_dict, output_path=None):
    """
    Plot comparison of metrics for different models/approaches
    
    Args:
        results_dict: Dictionary with model names as keys and metric dictionaries as values
        output_path: Path to save the plot
    """
    try:
        logger.info("Plotting metrics comparison...")
        
        metrics = ['precision', 'recall', 'f1_score', 'accuracy']
        model_names = list(results_dict.keys())
        
        # Create DataFrame for plotting
        data = []
        for model, metrics_dict in results_dict.items():
            for metric, value in metrics_dict.items():
                if metric in metrics:
                    data.append({'Model': model, 'Metric': metric, 'Value': value})
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 8))
        chart = sns.barplot(x='Metric', y='Value', hue='Model', data=df)
        
        plt.title('Performance Metrics Comparison')
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Model/Approach')
        
        # Add value labels on the bars
        for container in chart.containers:
            chart.bar_label(container, fmt='%.2f')
            
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Metrics comparison plot saved to {output_path}")
            
    except Exception as e:
        logger.error(f"Error plotting metrics comparison: {e}")
        raise
