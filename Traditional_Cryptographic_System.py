import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import re
import os
import base64
import hashlib
import json
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa, padding as asy_padding, ed25519, ed448
from cryptography.exceptions import InvalidSignature

class TraditionalCryptographicValidator:
    """
    Traditional cryptographic-only validation system for autonomous vehicle updates.
    Uses cryptographic checks: authenticity, rollforward protection, and chain integrity.
    """
    
    def __init__(self):
        self.validation_results = []
        self.performance_metrics = {}
    
    def extract_crypto_features(self, log_text: str) -> Dict:
        """Extract cryptographic features from simulation log text."""
        features = {}
        
        # Extract authenticity
        auth_match = re.search(r'Authenticity was (\d)', log_text)
        features['authenticity'] = int(auth_match.group(1)) if auth_match else 0
        
        # Extract rollforward
        roll_match = re.search(r'rollforward was (\d)', log_text)
        features['rollforward'] = int(roll_match.group(1)) if roll_match else 0
        
        # Extract chain integrity
        chain_match = re.search(r'chain integrity was (\d)', log_text)
        features['chain_integrity'] = int(chain_match.group(1)) if chain_match else 0
        
        # Extract baseline system acceptance
        baseline_match = re.search(r'baseline system acceptance was (\d)', log_text)
        features['baseline_acceptance'] = int(baseline_match.group(1)) if baseline_match else 0
        
        # Extract additional metadata for analysis
        time_match = re.search(r'time (\d+)', log_text)
        features['time'] = int(time_match.group(1)) if time_match else 0
        
        node_match = re.search(r'node (N\d+)', log_text)
        features['node'] = node_match.group(1) if node_match else 'Unknown'
        
        role_match = re.search(r"'(\w+)' role", log_text)
        features['role'] = role_match.group(1) if role_match else 'unknown'
        
        channel_match = re.search(r'via the (\w+) channel', log_text)
        features['channel'] = channel_match.group(1) if channel_match else 'unknown'
        
        path_match = re.search(r'path length of (\d+)', log_text)
        features['path_length'] = int(path_match.group(1)) if path_match else 0
        
        creator_match = re.search(r'created by (\w+)', log_text)
        features['creator'] = creator_match.group(1) if creator_match else 'unknown'
        
        version_match = re.search(r'version (\d+)', log_text)
        features['version'] = int(version_match.group(1)) if version_match else 0
        
        return features
    
    def _load_public_key(self, path):
        with open(path, 'rb') as f:
            data = f.read()
        return serialization.load_pem_public_key(data)

    def _verify_signature(self, payload_bytes, signature_b64, public_key_path):
        sig = base64.b64decode(signature_b64)
        key = self._load_public_key(public_key_path)
        try:
            if isinstance(key, rsa.RSAPublicKey):
                key.verify(sig, payload_bytes, asy_padding.PKCS1v15(), hashes.SHA256())
            elif isinstance(key, ec.EllipticCurvePublicKey):
                key.verify(sig, payload_bytes, ec.ECDSA(hashes.SHA256()))
            elif isinstance(key, ed25519.Ed25519PublicKey):
                key.verify(sig, payload_bytes)
            elif isinstance(key, ed448.Ed448PublicKey):
                key.verify(sig, payload_bytes)
            else:
                return False
            return True
        except InvalidSignature:
            return False
        except Exception:
            return False

    def _verify_integrity(self, payload_path, expected_sha256):
        h = hashlib.sha256()
        with open(payload_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest().lower() == str(expected_sha256).lower()

    def _read_version(self, state_path):
        try:
            with open(state_path, 'r') as f:
                obj = json.load(f)
            return int(obj.get('version', -1))
        except Exception:
            return -1

    def _write_version(self, state_path, version):
        try:
            with open(state_path, 'w') as f:
                json.dump({'version': int(version)}, f)
        except Exception:
            pass
    
    def cryptographic_validation(self, features: Dict) -> int:
        """
        Traditional cryptographic validation logic.
        Returns 0 for benign (accept), 1 for malicious (reject).
        """
        if features['authenticity'] == 0:
            return 1
        if features['rollforward'] == 0:
            return 1
        if features['chain_integrity'] == 0:
            return 1
        if features['baseline_acceptance'] == 0:
            return 1

        strict = os.getenv("CRYPTO_STRICT", "0") == "1"
        enforce_version = os.getenv("CRYPTO_VERSION_ENFORCE", "0") == "1"
        state_path = os.getenv("CRYPTO_VERSION_STATE", "./version_state.json")
        payload_path = features.get('payload_path')
        signature_b64 = features.get('signature_b64')
        expected_sha256 = features.get('expected_sha256')
        public_key_path = features.get('public_key_path') or os.getenv("CRYPTO_PUBLIC_KEY_PATH")

        if payload_path and expected_sha256:
            if not self._verify_integrity(payload_path, expected_sha256):
                return 1
        elif strict:
            return 1

        if payload_path and signature_b64 and public_key_path:
            try:
                with open(payload_path, 'rb') as f:
                    payload_bytes = f.read()
            except Exception:
                return 1
            if not self._verify_signature(payload_bytes, signature_b64, public_key_path):
                return 1
        elif strict:
            return 1

        if features['creator'] not in ['OEM']:
            return 1
        if features['path_length'] > 6:
            return 1

        if enforce_version and 'version' in features and features['version']:
            incoming = int(features['version'])
            current = self._read_version(state_path)
            if incoming <= current:
                return 1
            self._write_version(state_path, incoming)

        return 0
    
    def validate_dataset(self, df: pd.DataFrame) -> Tuple[List[int], Dict]:
        """Validate entire dataset using traditional cryptographic methods."""
        predictions = []
        detailed_results = []
        
        structured_cols = [c for c in ['payload_path', 'signature_b64', 'expected_sha256', 'public_key_path', 'version'] if c in df.columns]
        
        for idx, row in df.iterrows():
            features = self.extract_crypto_features(row['input_prompt'])
            for c in structured_cols:
                val = row[c]
                if pd.isna(val):
                    continue
                features[c] = val
            prediction = self.cryptographic_validation(features)
            predictions.append(prediction)
            
            # Store detailed results for analysis
            result = {
                'index': idx,
                'features': features,
                'prediction': prediction,
                'actual_label': row['output_label'],
                'correct': prediction == row['output_label']
            }
            detailed_results.append(result)
        
        self.validation_results = detailed_results
        return predictions, detailed_results
    
    def calculate_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict:
        """Calculate performance metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'confusion_matrix': cm
        }
        
        self.performance_metrics = metrics
        return metrics
    
    def analyze_failure_patterns(self) -> Dict:
        """Analyze patterns in validation failures."""
        failures = [r for r in self.validation_results if not r['correct']]
        
        analysis = {
            'total_failures': len(failures),
            'false_positives': len([f for f in failures if f['prediction'] == 1 and f['actual_label'] == 0]),
            'false_negatives': len([f for f in failures if f['prediction'] == 0 and f['actual_label'] == 1]),
            'failure_by_channel': {},
            'failure_by_role': {},
            'failure_by_creator': {}
        }
        
        # Analyze failure patterns
        for failure in failures:
            features = failure['features']
            
            # By channel
            channel = features.get('channel', 'unknown')
            analysis['failure_by_channel'][channel] = analysis['failure_by_channel'].get(channel, 0) + 1
            
            # By role
            role = features.get('role', 'unknown')
            analysis['failure_by_role'][role] = analysis['failure_by_role'].get(role, 0) + 1
            
            # By creator
            creator = features.get('creator', 'unknown')
            analysis['failure_by_creator'][creator] = analysis['failure_by_creator'].get(creator, 0) + 1
        
        return analysis

def main():
    """Main function to run traditional cryptographic validation."""
    print("=== Traditional Cryptographic Validation System ===\n")
    
    # Initialize validator
    validator = TraditionalCryptographicValidator()
    
    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv("./data/training_set.csv")
    val_df = pd.read_csv("./data/validation_set.csv")
    test_df = pd.read_csv("./data/test_set.csv")
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples\n")
    
    # Validate test set (main evaluation)
    print("Running cryptographic validation on test set...")
    test_predictions, test_results = validator.validate_dataset(test_df)
    
    # Calculate metrics
    test_metrics = validator.calculate_metrics(test_df['output_label'].tolist(), test_predictions)
    
    # Print results
    print("\n=== Test Set Performance ===")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"Specificity: {test_metrics['specificity']:.4f}")
    print(f"False Positive Rate: {test_metrics['false_positive_rate']:.4f}")
    print(f"False Negative Rate: {test_metrics['false_negative_rate']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"True Negatives: {test_metrics['true_negatives']}")
    print(f"False Positives: {test_metrics['false_positives']}")
    print(f"False Negatives: {test_metrics['false_negatives']}")
    print(f"True Positives: {test_metrics['true_positives']}")
    
    # Analyze failure patterns
    failure_analysis = validator.analyze_failure_patterns()
    print(f"\n=== Failure Analysis ===")
    print(f"Total Failures: {failure_analysis['total_failures']}")
    print(f"False Positives: {failure_analysis['false_positives']}")
    print(f"False Negatives: {failure_analysis['false_negatives']}")
    print(f"Failures by Channel: {failure_analysis['failure_by_channel']}")
    print(f"Failures by Role: {failure_analysis['failure_by_role']}")
    print(f"Failures by Creator: {failure_analysis['failure_by_creator']}")
    
    # Save results
    results_df = pd.DataFrame([
        {
            'method': 'Traditional_Cryptographic',
            'dataset': 'test',
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1_score': test_metrics['f1_score'],
            'specificity': test_metrics['specificity'],
            'false_positive_rate': test_metrics['false_positive_rate'],
            'false_negative_rate': test_metrics['false_negative_rate']
        }
    ])
    
    results_df.to_csv('./traditional_crypto_results.csv', index=False)
    print(f"\nResults saved to './traditional_crypto_results.csv'")
    
    return validator, test_metrics, failure_analysis

if __name__ == "__main__":
    validator, metrics, analysis = main()
