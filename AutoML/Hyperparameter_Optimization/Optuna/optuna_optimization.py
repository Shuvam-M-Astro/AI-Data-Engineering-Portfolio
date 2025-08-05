"""
AutoML Hyperparameter Optimization with Optuna
=============================================

This module implements a comprehensive AutoML hyperparameter optimization system using Optuna that:
- Automatically optimizes hyperparameters for multiple ML algorithms
- Supports various optimization strategies (TPE, Random, Grid Search)
- Implements multi-objective optimization (accuracy, speed, interpretability)
- Provides visualization and analysis of optimization results
- Supports early stopping and pruning for efficient optimization

Author: AI Data Engineering Portfolio
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler, RandomSampler, GridSampler
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.datasets import make_classification, make_regression

class AutoMLOptimizer:
    """
    Comprehensive AutoML Hyperparameter Optimization System using Optuna
    """
    
    def __init__(self, n_trials=100, timeout=300, n_jobs=-1):
        """
        Initialize the AutoML optimizer
        
        Parameters:
        -----------
        n_trials : int
            Number of optimization trials
        timeout : int
            Timeout in seconds for optimization
        n_jobs : int
            Number of parallel jobs
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.study = None
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
        
    def generate_synthetic_data(self, n_samples=10000, n_features=20, n_classes=2, task='classification'):
        """
        Generate synthetic data for optimization
        
        Parameters:
        -----------
        n_samples : int
            Number of samples
        n_features : int
            Number of features
        n_classes : int
            Number of classes (for classification)
        task : str
            'classification' or 'regression'
            
        Returns:
        --------
        tuple
            (X, y) synthetic data
        """
        if task == 'classification':
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=10,
                n_redundant=5,
                n_clusters_per_class=2,
                n_classes=n_classes,
                random_state=42
            )
        else:
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=10,
                random_state=42
            )
        
        return X, y
    
    def create_model(self, trial, model_type='random_forest'):
        """
        Create a model with hyperparameters from trial
        
        Parameters:
        -----------
        trial : optuna.Trial
            Optuna trial object
        model_type : str
            Type of model to create
            
        Returns:
        --------
        sklearn estimator
            Configured model
        """
        if model_type == 'random_forest':
            n_estimators = trial.suggest_int('rf_n_estimators', 50, 300)
            max_depth = trial.suggest_int('rf_max_depth', 3, 20)
            min_samples_split = trial.suggest_int('rf_min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('rf_min_samples_leaf', 1, 10)
            max_features = trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None])
            
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42,
                n_jobs=1
            )
        
        elif model_type == 'gradient_boosting':
            n_estimators = trial.suggest_int('gb_n_estimators', 50, 300)
            learning_rate = trial.suggest_float('gb_learning_rate', 0.01, 0.3, log=True)
            max_depth = trial.suggest_int('gb_max_depth', 3, 10)
            subsample = trial.suggest_float('gb_subsample', 0.6, 1.0)
            
            return GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                random_state=42
            )
        
        elif model_type == 'logistic_regression':
            C = trial.suggest_float('lr_C', 1e-4, 1e2, log=True)
            penalty = trial.suggest_categorical('lr_penalty', ['l1', 'l2', 'elasticnet'])
            solver = trial.suggest_categorical('lr_solver', ['liblinear', 'saga'])
            
            return LogisticRegression(
                C=C,
                penalty=penalty,
                solver=solver,
                random_state=42,
                max_iter=1000
            )
        
        elif model_type == 'svm':
            C = trial.suggest_float('svm_C', 1e-4, 1e2, log=True)
            kernel = trial.suggest_categorical('svm_kernel', ['rbf', 'linear', 'poly'])
            gamma = trial.suggest_categorical('svm_gamma', ['scale', 'auto'])
            
            return SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                random_state=42,
                probability=True
            )
        
        elif model_type == 'neural_network':
            hidden_layer_sizes = trial.suggest_categorical('nn_hidden_layer_sizes', 
                                                        [(50,), (100,), (50, 50), (100, 50), (100, 100)])
            alpha = trial.suggest_float('nn_alpha', 1e-5, 1e-1, log=True)
            learning_rate_init = trial.suggest_float('nn_learning_rate_init', 1e-4, 1e-1, log=True)
            
            return MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                random_state=42,
                max_iter=500
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def objective_function(self, trial, X, y, model_type='random_forest', cv_folds=5):
        """
        Objective function for optimization
        
        Parameters:
        -----------
        trial : optuna.Trial
            Optuna trial object
        X : array-like
            Feature matrix
        y : array-like
            Target variable
        model_type : str
            Type of model to optimize
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        float
            Cross-validation score
        """
        # Create model with suggested hyperparameters
        model = self.create_model(trial, model_type)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X, y, 
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=1
        )
        
        # Return mean CV score
        return cv_scores.mean()
    
    def multi_objective_function(self, trial, X, y, model_type='random_forest', cv_folds=5):
        """
        Multi-objective function for optimization (accuracy, speed, interpretability)
        
        Parameters:
        -----------
        trial : optuna.Trial
            Optuna trial object
        X : array-like
            Feature matrix
        y : array-like
            Target variable
        model_type : str
            Type of model to optimize
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        tuple
            (accuracy, speed_score, interpretability_score)
        """
        # Create model with suggested hyperparameters
        model = self.create_model(trial, model_type)
        
        # Measure training time
        start_time = time.time()
        cv_scores = cross_val_score(
            model, X, y, 
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=1
        )
        training_time = time.time() - start_time
        
        # Calculate accuracy
        accuracy = cv_scores.mean()
        
        # Calculate speed score (inverse of training time)
        speed_score = 1.0 / (1.0 + training_time)
        
        # Calculate interpretability score based on model type
        if model_type == 'logistic_regression':
            interpretability_score = 1.0
        elif model_type == 'random_forest':
            interpretability_score = 0.8
        elif model_type == 'gradient_boosting':
            interpretability_score = 0.7
        elif model_type == 'svm':
            interpretability_score = 0.6
        elif model_type == 'neural_network':
            interpretability_score = 0.3
        else:
            interpretability_score = 0.5
        
        return accuracy, speed_score, interpretability_score
    
    def optimize_hyperparameters(self, X, y, model_type='random_forest', 
                               optimization_type='single_objective', sampler_type='tpe'):
        """
        Optimize hyperparameters for a given model type
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target variable
        model_type : str
            Type of model to optimize
        optimization_type : str
            'single_objective' or 'multi_objective'
        sampler_type : str
            Type of sampler ('tpe', 'random', 'grid')
            
        Returns:
        --------
        optuna.Study
            Optimization study object
        """
        # Create study
        study_name = f"{model_type}_{optimization_type}"
        
        if sampler_type == 'tpe':
            sampler = TPESampler(seed=42)
        elif sampler_type == 'random':
            sampler = RandomSampler(seed=42)
        elif sampler_type == 'grid':
            sampler = GridSampler(seed=42)
        else:
            sampler = TPESampler(seed=42)
        
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        if optimization_type == 'single_objective':
            self.study = optuna.create_study(
                direction='maximize',
                sampler=sampler,
                pruner=pruner,
                study_name=study_name
            )
            
            # Run optimization
            self.study.optimize(
                lambda trial: self.objective_function(trial, X, y, model_type),
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=self.n_jobs
            )
            
        else:  # multi_objective
            self.study = optuna.create_study(
                directions=['maximize', 'maximize', 'maximize'],
                sampler=sampler,
                pruner=pruner,
                study_name=study_name
            )
            
            # Run optimization
            self.study.optimize(
                lambda trial: self.multi_objective_function(trial, X, y, model_type),
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=self.n_jobs
            )
        
        # Store best results
        if optimization_type == 'single_objective':
            self.best_params = self.study.best_params
            self.best_score = self.study.best_value
        else:
            # For multi-objective, get the best trial based on accuracy
            best_trial = max(self.study.trials, key=lambda t: t.values[0])
            self.best_params = best_trial.params
            self.best_score = best_trial.values[0]
        
        return self.study
    
    def optimize_multiple_models(self, X, y, model_types=None, optimization_type='single_objective'):
        """
        Optimize multiple model types and compare results
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target variable
        model_types : list
            List of model types to optimize
        optimization_type : str
            Type of optimization
            
        Returns:
        --------
        dict
            Results for each model type
        """
        if model_types is None:
            model_types = ['random_forest', 'gradient_boosting', 'logistic_regression', 'svm', 'neural_network']
        
        results = {}
        
        for model_type in model_types:
            print(f"\nOptimizing {model_type.replace('_', ' ').title()}...")
            
            try:
                study = self.optimize_hyperparameters(
                    X, y, model_type, optimization_type
                )
                
                results[model_type] = {
                    'study': study,
                    'best_params': self.best_params,
                    'best_score': self.best_score,
                    'n_trials': len(study.trials)
                }
                
                print(f"Best score: {self.best_score:.4f}")
                print(f"Best parameters: {self.best_params}")
                
            except Exception as e:
                print(f"Error optimizing {model_type}: {e}")
                results[model_type] = None
        
        return results
    
    def visualize_optimization_results(self, study, model_type):
        """
        Create visualizations for optimization results
        
        Parameters:
        -----------
        study : optuna.Study
            Optimization study
        model_type : str
            Type of model that was optimized
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Optimization Results for {model_type.replace("_", " ").title()}', fontsize=16)
        
        # Optimization history
        axes[0, 0].plot([trial.value for trial in study.trials if trial.value is not None])
        axes[0, 0].set_xlabel('Trial')
        axes[0, 0].set_ylabel('Objective Value')
        axes[0, 0].set_title('Optimization History')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Parameter importance
        try:
            importances = optuna.importance.get_param_importances(study)
            param_names = list(importances.keys())
            importance_values = list(importances.values())
            
            axes[0, 1].barh(param_names, importance_values)
            axes[0, 1].set_xlabel('Importance')
            axes[0, 1].set_title('Parameter Importance')
        except:
            axes[0, 1].text(0.5, 0.5, 'Parameter importance not available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # Parameter relationships (for first two parameters)
        if len(study.trials) > 0 and hasattr(study.trials[0], 'params'):
            params = list(study.trials[0].params.keys())
            if len(params) >= 2:
                param1, param2 = params[0], params[1]
                values1 = [trial.params.get(param1, 0) for trial in study.trials if trial.value is not None]
                values2 = [trial.params.get(param2, 0) for trial in study.trials if trial.value is not None]
                scores = [trial.value for trial in study.trials if trial.value is not None]
                
                scatter = axes[1, 0].scatter(values1, values2, c=scores, cmap='viridis')
                axes[1, 0].set_xlabel(param1)
                axes[1, 0].set_ylabel(param2)
                axes[1, 0].set_title(f'{param1} vs {param2}')
                plt.colorbar(scatter, ax=axes[1, 0])
        
        # Score distribution
        scores = [trial.value for trial in study.trials if trial.value is not None]
        if scores:
            axes[1, 1].hist(scores, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.4f}')
            axes[1, 1].axvline(np.max(scores), color='green', linestyle='--', label=f'Best: {np.max(scores):.4f}')
            axes[1, 1].set_xlabel('Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Score Distribution')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, results):
        """
        Compare results from multiple model optimizations
        
        Parameters:
        -----------
        results : dict
            Results from optimize_multiple_models
            
        Returns:
        --------
        pd.DataFrame
            Comparison table
        """
        comparison_data = []
        
        for model_type, result in results.items():
            if result is not None:
                comparison_data.append({
                    'Model': model_type.replace('_', ' ').title(),
                    'Best Score': result['best_score'],
                    'Number of Trials': result['n_trials'],
                    'Best Parameters': str(result['best_params'])
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Best Score', ascending=False)
        
        return comparison_df
    
    def save_optimization_results(self, results, filename='optimization_results.pkl'):
        """
        Save optimization results to file
        
        Parameters:
        -----------
        results : dict
            Optimization results
        filename : str
            Output filename
        """
        joblib.dump(results, filename)
        print(f"Results saved to {filename}")
    
    def load_optimization_results(self, filename='optimization_results.pkl'):
        """
        Load optimization results from file
        
        Parameters:
        -----------
        filename : str
            Input filename
            
        Returns:
        --------
        dict
            Loaded results
        """
        results = joblib.load(filename)
        print(f"Results loaded from {filename}")
        return results


def main():
    """
    Main function to demonstrate AutoML hyperparameter optimization
    """
    print("AutoML Hyperparameter Optimization with Optuna")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = AutoMLOptimizer(n_trials=50, timeout=120)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    X, y = optimizer.generate_synthetic_data(n_samples=5000, n_features=20)
    print(f"Generated {len(X)} samples with {X.shape[1]} features")
    
    # Optimize multiple models
    print("\nOptimizing multiple models...")
    results = optimizer.optimize_multiple_models(X, y)
    
    # Compare results
    print("\nModel Comparison:")
    comparison_df = optimizer.compare_models(results)
    print(comparison_df)
    
    # Visualize results for best model
    best_model = comparison_df.iloc[0]['Model'].lower().replace(' ', '_')
    if best_model in results and results[best_model] is not None:
        print(f"\nVisualizing results for {best_model}...")
        optimizer.visualize_optimization_results(results[best_model]['study'], best_model)
    
    # Save results
    optimizer.save_optimization_results(results)
    
    print("\nAutoML optimization demonstration completed!")


if __name__ == "__main__":
    main() 