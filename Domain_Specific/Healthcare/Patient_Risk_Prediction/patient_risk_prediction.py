"""
Healthcare Patient Risk Prediction System
========================================

This module implements a comprehensive healthcare patient risk prediction system that:
- Analyzes patient data for risk factors
- Predicts various health outcomes (readmission, mortality, complications)
- Uses multiple ML algorithms for robust prediction
- Provides explainable AI for medical decisions
- Implements risk stratification and early warning systems

Author: AI Data Engineering Portfolio
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

class PatientRiskPrediction:
    """
    Comprehensive Healthcare Patient Risk Prediction System
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.risk_threshold = 0.5
        
    def generate_synthetic_patient_data(self, n_samples=10000):
        """
        Generate synthetic patient data with health risk patterns
        
        Parameters:
        -----------
        n_samples : int
            Number of patients to generate
            
        Returns:
        --------
        pd.DataFrame
            Synthetic patient data with health outcomes
        """
        np.random.seed(42)
        
        # Patient IDs
        patient_ids = range(1, n_samples + 1)
        
        # Demographics
        ages = np.random.normal(65, 15, n_samples)
        ages = np.clip(ages, 18, 95)
        
        genders = np.random.choice(['Male', 'Female'], n_samples)
        
        # Vital signs
        systolic_bp = np.random.normal(130, 20, n_samples)
        diastolic_bp = np.random.normal(80, 10, n_samples)
        heart_rate = np.random.normal(75, 15, n_samples)
        temperature = np.random.normal(98.6, 1, n_samples)
        respiratory_rate = np.random.normal(16, 3, n_samples)
        oxygen_saturation = np.random.normal(97, 2, n_samples)
        
        # Lab values
        glucose = np.random.normal(100, 30, n_samples)
        creatinine = np.random.normal(1.0, 0.3, n_samples)
        hemoglobin = np.random.normal(14, 2, n_samples)
        white_blood_cells = np.random.normal(7, 2, n_samples)
        platelets = np.random.normal(250, 50, n_samples)
        
        # Medical history
        diabetes = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        hypertension = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        heart_disease = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        kidney_disease = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        cancer = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        
        # Current conditions
        infection = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        pneumonia = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        sepsis = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        
        # Treatment factors
        length_of_stay = np.random.exponential(5, n_samples)
        icu_stay = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        ventilator_use = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        
        # Create risk patterns
        risk_probability = np.zeros(n_samples)
        
        # Age risk
        age_risk = np.where(ages > 75, 0.3, np.where(ages > 65, 0.2, 0.1))
        risk_probability += age_risk
        
        # Vital signs risk
        bp_risk = np.where((systolic_bp > 160) | (diastolic_bp > 100), 0.2, 0)
        hr_risk = np.where((heart_rate > 100) | (heart_rate < 50), 0.15, 0)
        temp_risk = np.where(temperature > 100.4, 0.25, 0)
        o2_risk = np.where(oxygen_saturation < 95, 0.3, 0)
        
        risk_probability += bp_risk + hr_risk + temp_risk + o2_risk
        
        # Lab values risk
        glucose_risk = np.where(glucose > 200, 0.2, np.where(glucose < 70, 0.15, 0))
        creatinine_risk = np.where(creatinine > 1.5, 0.25, 0)
        wbc_risk = np.where(white_blood_cells > 12, 0.2, 0)
        
        risk_probability += glucose_risk + creatinine_risk + wbc_risk
        
        # Medical history risk
        risk_probability += diabetes * 0.2
        risk_probability += hypertension * 0.15
        risk_probability += heart_disease * 0.3
        risk_probability += kidney_disease * 0.25
        risk_probability += cancer * 0.4
        
        # Current conditions risk
        risk_probability += infection * 0.2
        risk_probability += pneumonia * 0.3
        risk_probability += sepsis * 0.6
        
        # Treatment risk
        los_risk = np.where(length_of_stay > 10, 0.2, 0)
        risk_probability += los_risk
        risk_probability += icu_stay * 0.3
        risk_probability += ventilator_use * 0.4
        
        # Generate outcomes
        readmission_30_days = np.random.random(n_samples) < (risk_probability * 0.8)
        mortality_30_days = np.random.random(n_samples) < (risk_probability * 0.3)
        complications = np.random.random(n_samples) < (risk_probability * 0.6)
        
        # Create additional features
        bmi = np.random.normal(27, 5, n_samples)
        bmi = np.clip(bmi, 18, 45)
        
        # Create DataFrame
        data = pd.DataFrame({
            'patient_id': patient_ids,
            'age': ages,
            'gender': genders,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'heart_rate': heart_rate,
            'temperature': temperature,
            'respiratory_rate': respiratory_rate,
            'oxygen_saturation': oxygen_saturation,
            'glucose': glucose,
            'creatinine': creatinine,
            'hemoglobin': hemoglobin,
            'white_blood_cells': white_blood_cells,
            'platelets': platelets,
            'bmi': bmi,
            'diabetes': diabetes,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'kidney_disease': kidney_disease,
            'cancer': cancer,
            'infection': infection,
            'pneumonia': pneumonia,
            'sepsis': sepsis,
            'length_of_stay': length_of_stay,
            'icu_stay': icu_stay,
            'ventilator_use': ventilator_use,
            'readmission_30_days': readmission_30_days.astype(int),
            'mortality_30_days': mortality_30_days.astype(int),
            'complications': complications.astype(int)
        })
        
        return data
    
    def preprocess_data(self, data, target_column='readmission_30_days'):
        """
        Preprocess patient data for risk prediction
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw patient data
        target_column : str
            Target variable for prediction
            
        Returns:
        --------
        tuple
            (X, y) preprocessed features and target
        """
        # Select features for modeling
        categorical_features = ['gender']
        numerical_features = ['age', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
                           'temperature', 'respiratory_rate', 'oxygen_saturation',
                           'glucose', 'creatinine', 'hemoglobin', 'white_blood_cells',
                           'platelets', 'bmi', 'length_of_stay',
                           'diabetes', 'hypertension', 'heart_disease', 'kidney_disease',
                           'cancer', 'infection', 'pneumonia', 'sepsis', 'icu_stay', 'ventilator_use']
        
        # Encode categorical features
        X_encoded = data[numerical_features].copy()
        
        for feature in categorical_features:
            le = LabelEncoder()
            X_encoded[feature] = le.fit_transform(data[feature])
            self.label_encoders[feature] = le
        
        # Target variable
        y = data[target_column].copy()
        
        # Handle missing values
        X_encoded = X_encoded.fillna(X_encoded.mean())
        
        return X_encoded, y
    
    def train_risk_models(self, X, y, model_type='readmission'):
        """
        Train multiple risk prediction models
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        model_type : str
            Type of risk prediction model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000
            ),
            'SVM': SVC(
                probability=True, random_state=42, kernel='rbf'
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name} for {model_type}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
            
            # Train on full training set
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = (y_pred == y_test).mean()
            precision = (y_pred & y_test).sum() / max(y_pred.sum(), 1)
            recall = (y_pred & y_test).sum() / max(y_test.sum(), 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)
            
            # ROC AUC
            try:
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
            except:
                auc = 0.5
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba if 'y_pred_proba' in locals() else None
            }
            
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1-Score: {f1:.3f}")
            print(f"AUC: {auc:.3f}")
            print(f"CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        self.models[model_type] = results
        
        # Store feature importance from Random Forest
        if 'Random Forest' in results:
            rf_model = results['Random Forest']['model']
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return X_test_scaled, y_test
    
    def predict_patient_risk(self, patient_data, model_type='readmission'):
        """
        Predict risk for new patient data
        
        Parameters:
        -----------
        patient_data : pd.DataFrame
            New patient data to analyze
        model_type : str
            Type of risk prediction
            
        Returns:
        --------
        dict
            Risk prediction results and scores
        """
        if model_type not in self.models:
            raise ValueError(f"Model for {model_type} must be trained first")
        
        # Preprocess new data
        categorical_features = ['gender']
        numerical_features = ['age', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
                           'temperature', 'respiratory_rate', 'oxygen_saturation',
                           'glucose', 'creatinine', 'hemoglobin', 'white_blood_cells',
                           'platelets', 'bmi', 'length_of_stay',
                           'diabetes', 'hypertension', 'heart_disease', 'kidney_disease',
                           'cancer', 'infection', 'pneumonia', 'sepsis', 'icu_stay', 'ventilator_use']
        
        # Encode categorical features
        X_new = patient_data[numerical_features].copy()
        
        for feature in categorical_features:
            if feature in self.label_encoders:
                le = self.label_encoders[feature]
                X_new[feature] = patient_data[feature].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
        
        # Handle missing values
        X_new = X_new.fillna(X_new.mean())
        
        # Scale features
        X_new_scaled = self.scaler.transform(X_new)
        
        # Get predictions from all models
        risk_scores = {}
        risk_predictions = {}
        
        for name, result in self.models[model_type].items():
            model = result['model']
            
            try:
                predictions = model.predict(X_new_scaled)
                scores = model.predict_proba(X_new_scaled)[:, 1]
            except:
                predictions = np.zeros(len(X_new_scaled))
                scores = np.zeros(len(X_new_scaled))
            
            risk_scores[name] = scores
            risk_predictions[name] = predictions
        
        # Ensemble prediction
        ensemble_scores = np.mean([scores for scores in risk_scores.values()], axis=0)
        ensemble_predictions = (ensemble_scores > self.risk_threshold).astype(int)
        
        return {
            'individual_scores': risk_scores,
            'individual_predictions': risk_predictions,
            'ensemble_scores': ensemble_scores,
            'ensemble_predictions': ensemble_predictions
        }
    
    def analyze_risk_factors(self, data):
        """
        Analyze risk factors in patient data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Patient data with outcomes
        """
        print("\nRisk Factor Analysis")
        print("=" * 50)
        
        # Overall risk rates
        readmission_rate = data['readmission_30_days'].mean()
        mortality_rate = data['mortality_30_days'].mean()
        complication_rate = data['complications'].mean()
        
        print(f"Overall 30-day readmission rate: {readmission_rate:.3f} ({readmission_rate*100:.1f}%)")
        print(f"Overall 30-day mortality rate: {mortality_rate:.3f} ({mortality_rate*100:.1f}%)")
        print(f"Overall complication rate: {complication_rate:.3f} ({complication_rate*100:.1f}%)")
        
        # Risk by age groups
        print("\nRisk by Age Groups:")
        age_bins = [0, 50, 65, 75, 100]
        age_labels = ['18-50', '51-65', '66-75', '75+']
        data['age_group'] = pd.cut(data['age'], bins=age_bins, labels=age_labels)
        
        risk_by_age = data.groupby('age_group')[['readmission_30_days', 'mortality_30_days', 'complications']].mean()
        print(risk_by_age)
        
        # Risk by vital signs
        print("\nRisk by Vital Signs:")
        print("High Blood Pressure (>140/90):")
        high_bp_mask = (data['systolic_bp'] > 140) | (data['diastolic_bp'] > 90)
        print(f"Readmission rate: {data[high_bp_mask]['readmission_30_days'].mean():.3f}")
        print(f"Mortality rate: {data[high_bp_mask]['mortality_30_days'].mean():.3f}")
        
        print("\nHigh Heart Rate (>100):")
        high_hr_mask = data['heart_rate'] > 100
        print(f"Readmission rate: {data[high_hr_mask]['readmission_30_days'].mean():.3f}")
        print(f"Mortality rate: {data[high_hr_mask]['mortality_30_days'].mean():.3f}")
        
        # Risk by comorbidities
        print("\nRisk by Comorbidities:")
        comorbidities = ['diabetes', 'hypertension', 'heart_disease', 'kidney_disease', 'cancer']
        for comorbidity in comorbidities:
            comorbidity_mask = data[comorbidity] == 1
            if comorbidity_mask.sum() > 0:
                print(f"{comorbidity.title()}:")
                print(f"  Readmission rate: {data[comorbidity_mask]['readmission_30_days'].mean():.3f}")
                print(f"  Mortality rate: {data[comorbidity_mask]['mortality_30_days'].mean():.3f}")
    
    def visualize_risk_analysis(self, data, predictions=None):
        """
        Create visualizations for risk analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Patient data
        predictions : dict, optional
            Model predictions
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Healthcare Risk Prediction Analysis', fontsize=16)
        
        # Age vs Risk
        axes[0, 0].scatter(data[data['readmission_30_days'] == 0]['age'], 
                           data[data['readmission_30_days'] == 0]['systolic_bp'], 
                           alpha=0.6, label='No Readmission', s=20)
        axes[0, 0].scatter(data[data['readmission_30_days'] == 1]['age'], 
                           data[data['readmission_30_days'] == 1]['systolic_bp'], 
                           alpha=0.8, label='Readmission', s=50, color='red')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Systolic BP')
        axes[0, 0].set_title('Age vs Blood Pressure')
        axes[0, 0].legend()
        
        # Risk by age groups
        risk_by_age = data.groupby('age_group')['readmission_30_days'].mean()
        axes[0, 1].bar(risk_by_age.index, risk_by_age.values)
        axes[0, 1].set_xlabel('Age Group')
        axes[0, 1].set_ylabel('Readmission Rate')
        axes[0, 1].set_title('Readmission Rate by Age Group')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Vital signs distribution
        axes[0, 2].hist(data[data['readmission_30_days'] == 0]['heart_rate'], 
                        bins=30, alpha=0.7, label='No Readmission', density=True)
        axes[0, 2].hist(data[data['readmission_30_days'] == 1]['heart_rate'], 
                        bins=30, alpha=0.7, label='Readmission', density=True)
        axes[0, 2].set_xlabel('Heart Rate')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Heart Rate Distribution')
        axes[0, 2].legend()
        
        # Comorbidity impact
        comorbidities = ['diabetes', 'hypertension', 'heart_disease', 'kidney_disease', 'cancer']
        comorbidity_risks = []
        for comorbidity in comorbidities:
            comorbidity_mask = data[comorbidity] == 1
            if comorbidity_mask.sum() > 0:
                risk = data[comorbidity_mask]['readmission_30_days'].mean()
                comorbidity_risks.append(risk)
            else:
                comorbidity_risks.append(0)
        
        axes[1, 0].barh(comorbidities, comorbidity_risks)
        axes[1, 0].set_xlabel('Readmission Rate')
        axes[1, 0].set_title('Risk by Comorbidity')
        
        # Feature importance
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            axes[1, 1].barh(top_features['feature'], top_features['importance'])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Top 10 Risk Factors')
        
        # Model comparison
        if predictions and 'ensemble_scores' in predictions:
            # ROC curve for ensemble
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(data['readmission_30_days'], predictions['ensemble_scores'])
            axes[1, 2].plot(fpr, tpr, label='Ensemble Model')
            axes[1, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[1, 2].set_xlabel('False Positive Rate')
            axes[1, 2].set_ylabel('True Positive Rate')
            axes[1, 2].set_title('ROC Curve')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_risk_report(self, patient_id, patient_data, risk_results, model_type='readmission'):
        """
        Generate a detailed risk analysis report
        
        Parameters:
        -----------
        patient_id : int
            Patient identifier
        patient_data : pd.DataFrame
            Patient details
        risk_results : dict
            Risk prediction results
        model_type : str
            Type of risk prediction
            
        Returns:
        --------
        str
            Formatted risk report
        """
        ensemble_score = risk_results['ensemble_scores'][0]
        ensemble_prediction = risk_results['ensemble_predictions'][0]
        
        # Get individual model scores
        model_scores = {name: scores[0] for name, scores in risk_results['individual_scores'].items()}
        
        # Risk level classification
        if ensemble_score > 0.7:
            risk_level = "HIGH"
            risk_description = "Patient shows significant risk factors"
        elif ensemble_score > 0.4:
            risk_level = "MODERATE"
            risk_description = "Patient shows some risk factors"
        else:
            risk_level = "LOW"
            risk_description = "Patient shows minimal risk factors"
        
        report = f"""
        PATIENT RISK ASSESSMENT REPORT
        ==============================
        Patient ID: {patient_id}
        Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        Risk Type: {model_type.replace('_', ' ').title()}
        
        PATIENT DEMOGRAPHICS:
        - Age: {patient_data['age'].iloc[0]:.0f} years
        - Gender: {patient_data['gender'].iloc[0]}
        - BMI: {patient_data['bmi'].iloc[0]:.1f}
        
        VITAL SIGNS:
        - Blood Pressure: {patient_data['systolic_bp'].iloc[0]:.0f}/{patient_data['diastolic_bp'].iloc[0]:.0f} mmHg
        - Heart Rate: {patient_data['heart_rate'].iloc[0]:.0f} bpm
        - Temperature: {patient_data['temperature'].iloc[0]:.1f}°F
        - Oxygen Saturation: {patient_data['oxygen_saturation'].iloc[0]:.1f}%
        
        LAB VALUES:
        - Glucose: {patient_data['glucose'].iloc[0]:.0f} mg/dL
        - Creatinine: {patient_data['creatinine'].iloc[0]:.2f} mg/dL
        - Hemoglobin: {patient_data['hemoglobin'].iloc[0]:.1f} g/dL
        - White Blood Cells: {patient_data['white_blood_cells'].iloc[0]:.1f} K/μL
        
        COMORBIDITIES:
        - Diabetes: {'Yes' if patient_data['diabetes'].iloc[0] else 'No'}
        - Hypertension: {'Yes' if patient_data['hypertension'].iloc[0] else 'No'}
        - Heart Disease: {'Yes' if patient_data['heart_disease'].iloc[0] else 'No'}
        - Kidney Disease: {'Yes' if patient_data['kidney_disease'].iloc[0] else 'No'}
        - Cancer: {'Yes' if patient_data['cancer'].iloc[0] else 'No'}
        
        RISK ASSESSMENT:
        - Ensemble Risk Score: {ensemble_score:.3f}
        - Risk Level: {risk_level}
        - Risk Description: {risk_description}
        - Prediction: {'HIGH RISK' if ensemble_prediction == 1 else 'LOW RISK'}
        
        INDIVIDUAL MODEL SCORES:
        """
        
        for model_name, score in model_scores.items():
            report += f"- {model_name}: {score:.3f}\n"
        
        report += f"""
        RECOMMENDATIONS:
        """
        
        if ensemble_prediction == 1:
            if ensemble_score > 0.7:
                report += "- IMMEDIATE: Schedule follow-up within 1 week\n"
                report += "- MONITOR: Increase monitoring frequency\n"
                report += "- INTERVENTION: Consider preventive measures\n"
                report += "- EDUCATION: Provide patient education on risk factors\n"
            elif ensemble_score > 0.4:
                report += "- SCHEDULE: Schedule follow-up within 2 weeks\n"
                report += "- MONITOR: Regular monitoring recommended\n"
                report += "- PREVENTION: Focus on preventive care\n"
            else:
                report += "- MONITOR: Continue standard monitoring\n"
                report += "- PREVENTION: Emphasize preventive measures\n"
        else:
            if ensemble_score < 0.2:
                report += "- CONTINUE: Standard care protocol\n"
                report += "- ROUTINE: Regular check-ups as scheduled\n"
            else:
                report += "- MONITOR: Continue monitoring with attention\n"
                report += "- PREVENTION: Focus on preventive measures\n"
        
        return report


def main():
    """
    Main function to demonstrate patient risk prediction system
    """
    print("Healthcare Patient Risk Prediction System")
    print("=" * 50)
    
    # Initialize system
    risk_system = PatientRiskPrediction()
    
    # Generate synthetic data
    print("Generating synthetic patient data...")
    data = risk_system.generate_synthetic_patient_data(n_samples=10000)
    print(f"Generated {len(data)} patient records")
    
    # Analyze risk factors
    risk_system.analyze_risk_factors(data)
    
    # Train models for different outcomes
    outcomes = ['readmission_30_days', 'mortality_30_days', 'complications']
    
    for outcome in outcomes:
        print(f"\n{'='*20} Training {outcome.replace('_', ' ').title()} Model {'='*20}")
        
        # Preprocess data
        X, y = risk_system.preprocess_data(data, outcome)
        print(f"Preprocessed {len(X)} samples with {len(X.columns)} features")
        
        # Train models
        X_test, y_test = risk_system.train_risk_models(X, y, outcome)
    
    # Test risk prediction
    print("\nTesting risk prediction on new patients...")
    sample_patients = data.iloc[:5]
    
    for outcome in outcomes:
        risk_results = risk_system.predict_patient_risk(sample_patients, outcome)
        print(f"\n{outcome.replace('_', ' ').title()} Risk Scores:")
        for model_name, scores in risk_results['individual_scores'].items():
            print(f"  {model_name}: {scores[0]:.3f}")
        print(f"  Ensemble: {risk_results['ensemble_scores'][0]:.3f}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    risk_system.visualize_risk_analysis(data, risk_results)
    
    # Generate sample report
    print("\nGenerating sample risk report...")
    sample_patient = data.iloc[:1]
    report = risk_system.generate_risk_report(1, sample_patient, risk_results)
    print(report)
    
    print("\nPatient risk prediction system demonstration completed!")


if __name__ == "__main__":
    main() 