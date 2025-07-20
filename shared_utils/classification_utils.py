"""
Shared utilities for classification tasks across different projects.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class ClassificationUtils:
    """Utility class for common classification tasks."""
    
    @staticmethod
    def load_and_preprocess_data(
        file_path: str, 
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
        """
        Load and preprocess data for classification.
        
        Args:
            file_path: Path to the CSV file
            target_column: Name of the target column
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test, scaler, label_encoder
        """
        # Load data
        data = pd.read_csv(file_path)
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target if categorical
        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        else:
            label_encoder = None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder
    
    @staticmethod
    def evaluate_model(
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        class_names: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a classification model and return metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            class_names: Names of classes for reporting
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
    
    @staticmethod
    def plot_confusion_matrix(
        conf_matrix: np.ndarray,
        class_names: Optional[list] = None,
        title: str = "Confusion Matrix",
        figsize: Tuple[int, int] = (8, 6)
    ):
        """
        Plot confusion matrix with seaborn.
        
        Args:
            conf_matrix: Confusion matrix array
            class_names: Names of classes
            title: Plot title
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_training_history(
        history: Dict[str, list],
        figsize: Tuple[int, int] = (12, 4)
    ):
        """
        Plot training history for neural networks.
        
        Args:
            history: Training history dictionary
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot accuracy
        ax1.plot(history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history:
            ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def save_model_results(
        model, 
        results: Dict[str, Any],
        model_name: str,
        dataset_name: str,
        save_path: str = "results/"
    ):
        """
        Save model and results to disk.
        
        Args:
            model: Trained model
            results: Evaluation results
            model_name: Name of the model
            dataset_name: Name of the dataset
            save_path: Directory to save results
        """
        import os
        import joblib
        from datetime import datetime
        
        # Create results directory
        os.makedirs(save_path, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_filename = f"{save_path}{dataset_name}_{model_name}_{timestamp}.joblib"
        joblib.dump(model, model_filename)
        
        # Save results
        results_filename = f"{save_path}{dataset_name}_{model_name}_results_{timestamp}.json"
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value
        
        with open(results_filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Model saved to: {model_filename}")
        print(f"Results saved to: {results_filename}") 