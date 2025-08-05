"""
Predictive Maintenance System for Manufacturing Equipment
======================================================

This module implements a comprehensive predictive maintenance system that:
- Analyzes sensor data from manufacturing equipment
- Detects early warning signs of equipment failure
- Predicts remaining useful life (RUL) of components
- Provides maintenance recommendations
- Implements anomaly detection for equipment monitoring

Author: AI Data Engineering Portfolio
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class PredictiveMaintenance:
    """
    Predictive Maintenance System for Manufacturing Equipment
    """
    
    def __init__(self):
        self.rul_model = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.feature_importance = None
        
    def generate_synthetic_data(self, n_samples=10000, n_equipment=50):
        """
        Generate synthetic sensor data for manufacturing equipment
        
        Parameters:
        -----------
        n_samples : int
            Number of data points to generate
        n_equipment : int
            Number of equipment units to simulate
            
        Returns:
        --------
        pd.DataFrame
            Synthetic sensor data with equipment health indicators
        """
        np.random.seed(42)
        
        # Equipment IDs
        equipment_ids = np.random.choice(range(1, n_equipment + 1), n_samples)
        
        # Time features
        time_in_service = np.random.uniform(0, 1000, n_samples)
        operating_hours = np.random.uniform(0, 24, n_samples)
        
        # Sensor readings
        temperature = np.random.normal(60, 15, n_samples)
        vibration = np.random.normal(0.5, 0.2, n_samples)
        pressure = np.random.normal(100, 20, n_samples)
        speed = np.random.normal(1500, 200, n_samples)
        current = np.random.normal(10, 2, n_samples)
        voltage = np.random.normal(220, 10, n_samples)
        
        # Equipment age effect (older equipment shows more wear)
        age_factor = time_in_service / 1000
        temperature += age_factor * 20
        vibration += age_factor * 0.3
        pressure += age_factor * 15
        current += age_factor * 3
        
        # Add some anomalies
        anomaly_mask = np.random.random(n_samples) < 0.05
        temperature[anomaly_mask] += np.random.normal(30, 10, sum(anomaly_mask))
        vibration[anomaly_mask] += np.random.normal(0.5, 0.3, sum(anomaly_mask))
        
        # Calculate remaining useful life (RUL)
        # RUL decreases with age and increases with anomalies
        base_rul = 1000 - time_in_service
        anomaly_penalty = np.where(anomaly_mask, -200, 0)
        rul = np.maximum(0, base_rul + anomaly_penalty)
        
        # Create DataFrame
        data = pd.DataFrame({
            'equipment_id': equipment_ids,
            'time_in_service': time_in_service,
            'operating_hours': operating_hours,
            'temperature': temperature,
            'vibration': vibration,
            'pressure': pressure,
            'speed': speed,
            'current': current,
            'voltage': voltage,
            'rul': rul,
            'is_anomaly': anomaly_mask
        })
        
        return data
    
    def preprocess_data(self, data):
        """
        Preprocess sensor data for modeling
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw sensor data
            
        Returns:
        --------
        tuple
            (X, y) preprocessed features and target
        """
        # Select features for modeling
        feature_cols = ['time_in_service', 'operating_hours', 'temperature', 
                       'vibration', 'pressure', 'speed', 'current', 'voltage']
        
        X = data[feature_cols].copy()
        y = data['rul'].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Remove outliers using IQR method
        for col in feature_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X = X[(X[col] >= lower_bound) & (X[col] <= upper_bound)]
            y = y[X.index]
        
        return X, y
    
    def train_rul_model(self, X, y):
        """
        Train a model to predict remaining useful life
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable (RUL)
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.rul_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.rul_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.rul_model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("RUL Model Performance:")
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.3f}")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.rul_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return X_test_scaled, y_test, y_pred
    
    def train_anomaly_detector(self, X):
        """
        Train an anomaly detection model
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Train Isolation Forest for anomaly detection
        self.anomaly_detector = IsolationForest(
            contamination=0.05,
            random_state=42,
            n_estimators=100
        )
        
        self.anomaly_detector.fit(X_pca)
        
        # Predict anomalies
        anomalies = self.anomaly_detector.predict(X_pca)
        anomaly_scores = self.anomaly_detector.decision_function(X_pca)
        
        print(f"Anomaly Detection Results:")
        print(f"Number of anomalies detected: {sum(anomalies == -1)}")
        print(f"Anomaly rate: {sum(anomalies == -1) / len(anomalies):.3f}")
        
        return anomalies, anomaly_scores
    
    def predict_maintenance_needs(self, equipment_data):
        """
        Predict maintenance needs for equipment
        
        Parameters:
        -----------
        equipment_data : pd.DataFrame
            Current sensor data for equipment
            
        Returns:
        --------
        dict
            Maintenance predictions and recommendations
        """
        if self.rul_model is None:
            raise ValueError("RUL model must be trained first")
        
        # Preprocess data
        feature_cols = ['time_in_service', 'operating_hours', 'temperature', 
                       'vibration', 'pressure', 'speed', 'current', 'voltage']
        X = equipment_data[feature_cols].fillna(equipment_data[feature_cols].mean())
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict RUL
        predicted_rul = self.rul_model.predict(X_scaled)
        
        # Detect anomalies
        X_pca = self.pca.transform(X_scaled)
        anomalies = self.anomaly_detector.predict(X_pca)
        anomaly_scores = self.anomaly_detector.decision_function(X_pca)
        
        # Generate recommendations
        recommendations = []
        for i, (rul, anomaly, score) in enumerate(zip(predicted_rul, anomalies, anomaly_scores)):
            if anomaly == -1:
                recommendations.append("IMMEDIATE MAINTENANCE REQUIRED - Anomaly detected")
            elif rul < 50:
                recommendations.append("URGENT - Equipment likely to fail soon")
            elif rul < 200:
                recommendations.append("SCHEDULE MAINTENANCE - Plan maintenance within 200 hours")
            elif rul < 500:
                recommendations.append("MONITOR - Equipment showing signs of wear")
            else:
                recommendations.append("NORMAL - Equipment operating normally")
        
        return {
            'predicted_rul': predicted_rul,
            'anomalies': anomalies,
            'anomaly_scores': anomaly_scores,
            'recommendations': recommendations
        }
    
    def visualize_results(self, data, rul_predictions=None):
        """
        Create visualizations for predictive maintenance analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Sensor data
        rul_predictions : array-like, optional
            Predicted RUL values
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Predictive Maintenance Analysis', fontsize=16)
        
        # Temperature vs Time
        axes[0, 0].scatter(data['time_in_service'], data['temperature'], 
                           alpha=0.6, c=data['rul'], cmap='viridis')
        axes[0, 0].set_xlabel('Time in Service')
        axes[0, 0].set_ylabel('Temperature')
        axes[0, 0].set_title('Temperature vs Time (colored by RUL)')
        
        # Vibration vs Time
        axes[0, 1].scatter(data['time_in_service'], data['vibration'], 
                           alpha=0.6, c=data['rul'], cmap='viridis')
        axes[0, 1].set_xlabel('Time in Service')
        axes[0, 1].set_ylabel('Vibration')
        axes[0, 1].set_title('Vibration vs Time (colored by RUL)')
        
        # Pressure vs Current
        axes[0, 2].scatter(data['pressure'], data['current'], 
                           alpha=0.6, c=data['rul'], cmap='viridis')
        axes[0, 2].set_xlabel('Pressure')
        axes[0, 2].set_ylabel('Current')
        axes[0, 2].set_title('Pressure vs Current (colored by RUL)')
        
        # RUL Distribution
        axes[1, 0].hist(data['rul'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Remaining Useful Life')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('RUL Distribution')
        
        # Feature Importance
        if self.feature_importance is not None:
            axes[1, 1].barh(self.feature_importance['feature'], 
                           self.feature_importance['importance'])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Feature Importance')
        
        # Anomaly Detection
        if 'is_anomaly' in data.columns:
            normal_data = data[~data['is_anomaly']]
            anomaly_data = data[data['is_anomaly']]
            
            axes[1, 2].scatter(normal_data['temperature'], normal_data['vibration'], 
                              alpha=0.6, label='Normal', color='blue')
            axes[1, 2].scatter(anomaly_data['temperature'], anomaly_data['vibration'], 
                              alpha=0.8, label='Anomaly', color='red', s=100)
            axes[1, 2].set_xlabel('Temperature')
            axes[1, 2].set_ylabel('Vibration')
            axes[1, 2].set_title('Anomaly Detection')
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.show()
    
    def generate_maintenance_report(self, equipment_id, sensor_data):
        """
        Generate a comprehensive maintenance report for specific equipment
        
        Parameters:
        -----------
        equipment_id : int
            Equipment identifier
        sensor_data : pd.DataFrame
            Current sensor readings
            
        Returns:
        --------
        str
            Formatted maintenance report
        """
        predictions = self.predict_maintenance_needs(sensor_data)
        
        report = f"""
        EQUIPMENT MAINTENANCE REPORT
        ============================
        Equipment ID: {equipment_id}
        Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        CURRENT STATUS:
        - Predicted RUL: {predictions['predicted_rul'][0]:.1f} hours
        - Anomaly Detected: {'YES' if predictions['anomalies'][0] == -1 else 'NO'}
        - Anomaly Score: {predictions['anomaly_scores'][0]:.3f}
        
        RECOMMENDATION:
        {predictions['recommendations'][0]}
        
        SENSOR READINGS:
        - Temperature: {sensor_data['temperature'].iloc[0]:.1f}°C
        - Vibration: {sensor_data['vibration'].iloc[0]:.3f} mm/s
        - Pressure: {sensor_data['pressure'].iloc[0]:.1f} PSI
        - Speed: {sensor_data['speed'].iloc[0]:.0f} RPM
        - Current: {sensor_data['current'].iloc[0]:.1f} A
        - Voltage: {sensor_data['voltage'].iloc[0]:.1f} V
        
        NEXT ACTIONS:
        """
        
        if predictions['anomalies'][0] == -1:
            report += "- IMMEDIATE: Shutdown equipment and perform emergency maintenance\n"
            report += "- INSPECT: Check for mechanical failures, overheating, or electrical issues\n"
        elif predictions['predicted_rul'][0] < 50:
            report += "- SCHEDULE: Plan maintenance within 48 hours\n"
            report += "- MONITOR: Increase monitoring frequency to hourly checks\n"
        elif predictions['predicted_rul'][0] < 200:
            report += "- PLAN: Schedule maintenance within 1-2 weeks\n"
            report += "- PREPARE: Order necessary replacement parts\n"
        else:
            report += "- CONTINUE: Normal operation, continue routine monitoring\n"
            report += "- DOCUMENT: Record current condition for future reference\n"
        
        return report


def main():
    """
    Main function to demonstrate predictive maintenance system
    """
    print("Manufacturing Predictive Maintenance System")
    print("=" * 50)
    
    # Initialize system
    pm_system = PredictiveMaintenance()
    
    # Generate synthetic data
    print("Generating synthetic sensor data...")
    data = pm_system.generate_synthetic_data(n_samples=10000, n_equipment=50)
    print(f"Generated {len(data)} data points for {data['equipment_id'].nunique()} equipment units")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X, y = pm_system.preprocess_data(data)
    print(f"Preprocessed {len(X)} samples with {len(X.columns)} features")
    
    # Train RUL model
    print("\nTraining RUL prediction model...")
    X_test, y_test, y_pred = pm_system.train_rul_model(X, y)
    
    # Train anomaly detector
    print("\nTraining anomaly detection model...")
    anomalies, scores = pm_system.train_anomaly_detector(X)
    
    # Visualize results
    print("\nGenerating visualizations...")
    pm_system.visualize_results(data)
    
    # Demonstrate maintenance prediction
    print("\nDemonstrating maintenance prediction...")
    sample_equipment = data[data['equipment_id'] == 1].iloc[:1]
    report = pm_system.generate_maintenance_report(1, sample_equipment)
    print(report)
    
    print("\nPredictive maintenance system demonstration completed!")


if __name__ == "__main__":
    main() 