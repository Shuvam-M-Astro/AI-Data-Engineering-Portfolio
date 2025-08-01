import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import dvc.api
import os
import logging
from typing import Dict, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPipeline:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data using DVC"""
        try:
            with dvc.api.open(data_path, mode='rb') as f:
                data = pd.read_csv(f)
            logger.info(f"Data loaded successfully from {data_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame) -> tuple:
        """Preprocess data for training"""
        # Separate features and target
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   params: Dict[str, Any]) -> RandomForestClassifier:
        """Train model with MLflow tracking"""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            return model
    
    def evaluate_model(self, model: RandomForestClassifier,
                      X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model and log metrics"""
        with mlflow.start_run():
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{label}_{metric_name}", value)
            
            return {
                "accuracy": accuracy,
                "classification_report": report
            }
    
    def save_metrics(self, metrics: Dict[str, Any], output_path: str):
        """Save evaluation metrics to file"""
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {output_path}")

def main():
    # Initialize pipeline
    pipeline = MLPipeline("mlops_experiment")
    
    # Load data
    data = pipeline.load_data("data/processed/data.csv")
    
    # Preprocess data
    X_train, X_test, y_train, y_test = pipeline.preprocess_data(data)
    
    # Define model parameters
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
    
    # Train model
    logger.info("Training model...")
    model = pipeline.train_model(X_train, y_train, params)
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics = pipeline.evaluate_model(model, X_test, y_test)
    
    # Save metrics
    pipeline.save_metrics(metrics, "metrics/evaluation_metrics.json")
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main() 