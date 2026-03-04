import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from Traditional_Cryptographic_System import TraditionalCryptographicValidator

class CombinedAICryptoValidator:
    """
    Combined AI-assisted and cryptographic validation system.
    Integrates AI-driven anomaly detection with traditional cryptographic checks.
    """
    
    def __init__(self, model_path: str = "./results", crypto_weight: float = 0.6, ai_weight: float = 0.4):
        self.crypto_validator = TraditionalCryptographicValidator()
        self.crypto_weight = crypto_weight
        self.ai_weight = ai_weight
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.validation_results = []
        self.performance_metrics = {}
        
        # Load the fine-tuned AI model
        self.load_ai_model()
    
    def load_ai_model(self):
        """Load the fine-tuned AI model for update validation."""
        try:
            print(f"Loading AI model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()
            print("AI model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load AI model from {self.model_path}: {e}")
            print("Using fallback model...")
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    
    def get_ai_prediction(self, text: str) -> Tuple[int, float]:
        """Get AI model prediction and confidence score."""
        if self.model is None or self.tokenizer is None:
            return 0, 0.5  # Fallback prediction
        
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(logits, dim=-1).item()
                confidence = probabilities[0][prediction].item()
            
            return prediction, confidence
        except Exception as e:
            print(f"Error in AI prediction: {e}")
            return 0, 0.5  # Fallback
    
    def extract_provenance_features(self, log_text: str) -> Dict:
        """Extract provenance and metadata features for enhanced analysis."""
        features = self.crypto_validator.extract_crypto_features(log_text)
        
        # Additional provenance features
        # Analyze update dissemination patterns
        features['is_p2p'] = 1 if features['channel'] == 'P2P' else 0
        features['is_ota'] = 1 if features['channel'] == 'OTA' else 0
        
        # Version analysis
        features['version_anomaly'] = 1 if features['version'] > 30 or features['version'] < 1 else 0
        
        # Path length analysis (longer paths might indicate routing attacks)
        features['path_anomaly'] = 1 if features['path_length'] > 5 else 0
        
        # Time-based analysis
        features['time_anomaly'] = 1 if features['time'] > 500 or features['time'] < 10 else 0
        
        # Role-based analysis
        features['adversary_role'] = 1 if features['role'] == 'adversary' else 0
        
        # Creator legitimacy
        features['fake_creator'] = 1 if 'Fake' in features['creator'] else 0
        
        return features
    
    def combined_validation(self, text: str, features: Dict) -> Tuple[int, float, Dict]:
        """
        Combined validation using both cryptographic and AI methods.
        Returns prediction, confidence, and detailed scores.
        """
        # Get cryptographic validation
        crypto_prediction = self.crypto_validator.cryptographic_validation(features)
        crypto_confidence = 1.0 if crypto_prediction == 1 else 0.8  # High confidence for crypto
        
        # Get AI prediction
        ai_prediction, ai_confidence = self.get_ai_prediction(text)
        
        # Enhanced decision logic
        decision_details = {
            'crypto_prediction': crypto_prediction,
            'crypto_confidence': crypto_confidence,
            'ai_prediction': ai_prediction,
            'ai_confidence': ai_confidence,
            'crypto_weight': self.crypto_weight,
            'ai_weight': self.ai_weight
        }
        
        # Weighted combination approach
        weighted_score = (crypto_prediction * self.crypto_weight) + (ai_prediction * self.ai_weight)
        combined_prediction = 1 if weighted_score > 0.5 else 0
        
        # Confidence calculation
        confidence_score = (crypto_confidence * self.crypto_weight) + (ai_confidence * self.ai_weight)
        
        # Enhanced logic: If crypto says malicious, trust it more
        if crypto_prediction == 1:
            combined_prediction = 1
            confidence_score = max(confidence_score, 0.85)
        
        # If AI detects anomaly with high confidence and crypto is uncertain
        elif ai_prediction == 1 and ai_confidence > 0.8 and crypto_prediction == 0:
            # Check for sophisticated attacks that might bypass crypto
            if (features.get('fake_creator', 0) == 1 or 
                features.get('path_anomaly', 0) == 1 or
                features.get('version_anomaly', 0) == 1):
                combined_prediction = 1
                confidence_score = ai_confidence
        
        decision_details.update({
            'weighted_score': weighted_score,
            'final_prediction': combined_prediction,
            'final_confidence': confidence_score
        })
        
        return combined_prediction, confidence_score, decision_details
    
    def validate_dataset(self, df: pd.DataFrame) -> Tuple[List[int], List[Dict]]:
        """Validate entire dataset using combined AI+crypto approach."""
        predictions = []
        detailed_results = []
        
        print(f"Validating {len(df)} samples with combined AI+Crypto system...")
        
        for idx, row in df.iterrows():
            if idx % 50 == 0:
                print(f"Processing sample {idx+1}/{len(df)}")
            
            # Extract features
            features = self.extract_provenance_features(row['input_prompt'])
            
            # Get combined prediction
            prediction, confidence, details = self.combined_validation(row['input_prompt'], features)
            predictions.append(prediction)
            
            # Store detailed results
            result = {
                'index': idx,
                'features': features,
                'prediction': prediction,
                'confidence': confidence,
                'decision_details': details,
                'actual_label': row['output_label'],
                'correct': prediction == row['output_label']
            }
            detailed_results.append(result)
        
        self.validation_results = detailed_results
        return predictions, detailed_results
    
    def calculate_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict:
        """Calculate comprehensive performance metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Detection rates
        malicious_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        benign_acceptance_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'malicious_detection_rate': malicious_detection_rate,
            'benign_acceptance_rate': benign_acceptance_rate,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'confusion_matrix': cm
        }
        
        self.performance_metrics = metrics
        return metrics
    
    def analyze_ai_crypto_agreement(self) -> Dict:
        """Analyze agreement between AI and cryptographic components."""
        agreements = []
        disagreements = []
        
        for result in self.validation_results:
            details = result['decision_details']
            crypto_pred = details['crypto_prediction']
            ai_pred = details['ai_prediction']
            
            if crypto_pred == ai_pred:
                agreements.append(result)
            else:
                disagreements.append(result)
        
        analysis = {
            'total_agreements': len(agreements),
            'total_disagreements': len(disagreements),
            'agreement_rate': len(agreements) / len(self.validation_results) if self.validation_results else 0,
            'disagreement_patterns': {}
        }
        
        # Analyze disagreement patterns
        for disagree in disagreements:
            details = disagree['decision_details']
            pattern = f"Crypto:{details['crypto_prediction']}_AI:{details['ai_prediction']}"
            if pattern not in analysis['disagreement_patterns']:
                analysis['disagreement_patterns'][pattern] = {
                    'count': 0,
                    'correct_final': 0,
                    'examples': []
                }
            
            analysis['disagreement_patterns'][pattern]['count'] += 1
            if disagree['correct']:
                analysis['disagreement_patterns'][pattern]['correct_final'] += 1
            
            if len(analysis['disagreement_patterns'][pattern]['examples']) < 3:
                analysis['disagreement_patterns'][pattern]['examples'].append({
                    'index': disagree['index'],
                    'actual_label': disagree['actual_label'],
                    'final_prediction': disagree['prediction']
                })
        
        return analysis

def main():
    """Main function to run combined AI+crypto validation."""
    print("=== Combined AI + Cryptographic Validation System ===\n")
    
    # Initialize combined validator
    validator = CombinedAICryptoValidator(crypto_weight=0.6, ai_weight=0.4)
    
    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv("./3. training_set.csv")
    val_df = pd.read_csv("./4. validation_set.csv")
    test_df = pd.read_csv("./5. test_set.csv")
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples\n")
    
    # Validate test set
    print("Running combined AI+Crypto validation on test set...")
    test_predictions, test_results = validator.validate_dataset(test_df)
    
    # Calculate metrics
    test_metrics = validator.calculate_metrics(test_df['output_label'].tolist(), test_predictions)
    
    # Print results
    print("\n=== Combined System Test Performance ===")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"Specificity: {test_metrics['specificity']:.4f}")
    print(f"Malicious Detection Rate: {test_metrics['malicious_detection_rate']:.4f}")
    print(f"Benign Acceptance Rate: {test_metrics['benign_acceptance_rate']:.4f}")
    print(f"False Positive Rate: {test_metrics['false_positive_rate']:.4f}")
    print(f"False Negative Rate: {test_metrics['false_negative_rate']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"True Negatives: {test_metrics['true_negatives']}")
    print(f"False Positives: {test_metrics['false_positives']}")
    print(f"False Negatives: {test_metrics['false_negatives']}")
    print(f"True Positives: {test_metrics['true_positives']}")
    
    # Analyze AI-Crypto agreement
    agreement_analysis = validator.analyze_ai_crypto_agreement()
    print(f"\n=== AI-Crypto Agreement Analysis ===")
    print(f"Total Agreements: {agreement_analysis['total_agreements']}")
    print(f"Total Disagreements: {agreement_analysis['total_disagreements']}")
    print(f"Agreement Rate: {agreement_analysis['agreement_rate']:.4f}")
    print(f"Disagreement Patterns: {agreement_analysis['disagreement_patterns']}")
    
    # Save results
    results_df = pd.DataFrame([
        {
            'method': 'Combined_AI_Crypto',
            'dataset': 'test',
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1_score': test_metrics['f1_score'],
            'specificity': test_metrics['specificity'],
            'false_positive_rate': test_metrics['false_positive_rate'],
            'false_negative_rate': test_metrics['false_negative_rate'],
            'malicious_detection_rate': test_metrics['malicious_detection_rate'],
            'benign_acceptance_rate': test_metrics['benign_acceptance_rate'],
            'crypto_weight': validator.crypto_weight,
            'ai_weight': validator.ai_weight
        }
    ])
    
    results_df.to_csv('./combined_ai_crypto_results.csv', index=False)
    print(f"\nResults saved to './combined_ai_crypto_results.csv'")
    
    return validator, test_metrics, agreement_analysis

if __name__ == "__main__":
    validator, metrics, analysis = main()
