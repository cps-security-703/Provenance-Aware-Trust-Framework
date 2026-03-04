#!/usr/bin/env python3
"""
Model Retraining Pipeline to Fix Data Leakage Issues

This script creates a proper training pipeline that:
1. Cleans the training data by removing obvious indicators
2. Creates realistic features for training
3. Retrains the model on legitimate patterns only
4. Validates the retrained model doesn't overfit
"""

import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import os
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class DataCleaner:
    """Clean training data to remove obvious indicators that cause data leakage."""
    
    def __init__(self):
        self.replacement_patterns = {
            # Replace obvious attack indicators with neutral terms
            'adversary': ['participant', 'node', 'entity', 'actor'],
            'FakeOEM': ['ServiceProvider', 'UpdateSource', 'SystemEntity', 'NetworkNode'],
            'Fake': ['Alt', 'Secondary', 'Backup', 'External']
        }
        
        # Legitimate creators that should be preserved
        self.legitimate_creators = ['OEM', 'WeatherSvc', 'TrafficMgmt', 'PoliceDept']
        
    def clean_text(self, text: str, label: int) -> str:
        """Clean text by removing obvious indicators while preserving legitimate patterns."""
        cleaned_text = text
        
        # For malicious samples, replace obvious indicators
        if label == 1:  # Malicious
            # Replace 'adversary' role with neutral terms
            if "'adversary' role" in cleaned_text:
                replacement = np.random.choice(self.replacement_patterns['adversary'])
                cleaned_text = cleaned_text.replace("'adversary' role", f"'{replacement}' role")
            
            # Replace 'FakeOEM' with neutral service providers
            if 'FakeOEM' in cleaned_text:
                replacement = np.random.choice(self.replacement_patterns['FakeOEM'])
                cleaned_text = cleaned_text.replace('FakeOEM', replacement)
            
            # Replace other 'Fake' prefixes
            cleaned_text = re.sub(r'Fake(\w+)', 
                                lambda m: np.random.choice(self.replacement_patterns['Fake']) + m.group(1), 
                                cleaned_text)
        
        return cleaned_text
    
    def add_realistic_noise(self, text: str) -> str:
        """Add realistic variations to make the data more diverse."""
        # Randomly vary some numerical values slightly
        def vary_number(match):
            num = int(match.group(1))
            # Add small random variation (±10%)
            variation = int(num * 0.1 * (np.random.random() - 0.5))
            return f"time {max(1, num + variation)}"
        
        # Vary timing values
        text = re.sub(r'time (\d+)', vary_number, text)
        
        # Randomly vary version numbers slightly for diversity
        def vary_version(match):
            version = int(match.group(1))
            # Small variation for realism
            variation = np.random.randint(-2, 3)
            return f"version {max(1, version + variation)}"
        
        if np.random.random() < 0.3:  # 30% chance to vary
            text = re.sub(r'version (\d+)', vary_version, text)
        
        return text
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean entire dataset."""
        print("Cleaning dataset to remove data leakage...")
        
        cleaned_df = df.copy()
        
        # Clean each text sample
        for idx, row in df.iterrows():
            original_text = row['input_prompt']
            label = row['output_label']
            
            # Clean obvious indicators
            cleaned_text = self.clean_text(original_text, label)
            
            # Add realistic noise for diversity
            cleaned_text = self.add_realistic_noise(cleaned_text)
            
            cleaned_df.at[idx, 'input_prompt'] = cleaned_text
            
            if idx % 500 == 0:
                print(f"Cleaned {idx}/{len(df)} samples...")
        
        print(f"Dataset cleaning completed. Cleaned {len(df)} samples.")
        return cleaned_df

class UpdateDataset(Dataset):
    """PyTorch Dataset for update validation data."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ModelRetrainer:
    """Retrain the model on cleaned data."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.trainer = None
        
    def prepare_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[UpdateDataset, UpdateDataset]:
        """Prepare training and validation datasets."""
        print("Preparing datasets for training...")
        
        # Create datasets
        train_dataset = UpdateDataset(
            train_df['input_prompt'].tolist(),
            train_df['output_label'].tolist(),
            self.tokenizer
        )
        
        val_dataset = UpdateDataset(
            val_df['input_prompt'].tolist(),
            val_df['output_label'].tolist(),
            self.tokenizer
        )
        
        print(f"Training dataset: {len(train_dataset)} samples")
        print(f"Validation dataset: {len(val_dataset)} samples")
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train_model(self, train_dataset: UpdateDataset, val_dataset: UpdateDataset, 
                   output_dir: str = "./retrained_model"):
        """Train the model with proper regularization."""
        print("Initializing model for retraining...")
        
        # Initialize model with proper DistilBERT config
        from transformers import DistilBertConfig
        
        config = DistilBertConfig.from_pretrained(self.model_name)
        config.num_labels = 2
        config.dropout = 0.3  # Increased dropout for regularization
        config.attention_dropout = 0.3
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            config=config
        )
        
        # Training arguments with regularization
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,  # Reduced epochs to prevent overfitting
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=100,
            weight_decay=0.01,  # L2 regularization
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            eval_steps=100,
            save_steps=200,
            eval_strategy="steps",  # Fixed parameter name
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=3,
            seed=42,
            learning_rate=2e-5,  # Conservative learning rate
            dataloader_num_workers=0,  # Disable multiprocessing for stability
            remove_unused_columns=False,
            report_to=[]  # Disable wandb/tensorboard
        )
        
        # Initialize trainer with early stopping
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        print("Starting model training...")
        print("Training with regularization to prevent overfitting:")
        print(f"- Dropout: 0.3")
        print(f"- Weight decay: 0.01") 
        print(f"- Early stopping: 3 patience")
        print(f"- Reduced epochs: 5")
        
        # Train the model
        self.trainer.train()
        
        # Save the final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model training completed. Saved to {output_dir}")
        
        return self.trainer.state.log_history

def validate_retrained_model(model_path: str, test_df: pd.DataFrame) -> Dict:
    """Validate the retrained model to ensure it doesn't overfit."""
    print(f"Validating retrained model from {model_path}...")
    
    # Load retrained model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for _, row in test_df.iterrows():
            text = row['input_prompt']
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Get prediction
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][prediction].item()
            
            predictions.append(prediction)
            confidences.append(confidence)
    
    # Calculate metrics
    labels = test_df['output_label'].tolist()
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    # Check for overfitting indicators
    high_confidence_count = sum(1 for c in confidences if c > 0.95)
    overfitting_ratio = high_confidence_count / len(confidences)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_confidence': np.mean(confidences),
        'high_confidence_ratio': overfitting_ratio,
        'overfitting_risk': 'HIGH' if overfitting_ratio > 0.8 else 'MEDIUM' if overfitting_ratio > 0.5 else 'LOW'
    }
    
    return results

def main():
    """Main retraining pipeline."""
    print("=== MODEL RETRAINING PIPELINE ===")
    print("Fixing data leakage and overfitting issues\n")
    
    # Step 1: Load original datasets
    print("Step 1: Loading original datasets...")
    train_df = pd.read_csv("./data/training_set.csv")
    val_df = pd.read_csv("./data/validation_set.csv") 
    test_df = pd.read_csv("./data/test_set.csv")
    
    print(f"Original datasets:")
    print(f"- Training: {len(train_df)} samples")
    print(f"- Validation: {len(val_df)} samples")
    print(f"- Test: {len(test_df)} samples")
    
    # Step 2: Clean training data
    print("\nStep 2: Cleaning training data...")
    cleaner = DataCleaner()
    
    # Clean training and validation data (but not test data - keep it as ground truth)
    clean_train_df = cleaner.clean_dataset(train_df)
    clean_val_df = cleaner.clean_dataset(val_df)
    
    # Save cleaned datasets
    clean_train_df.to_csv("./data/cleaned_training_set.csv", index=False)
    clean_val_df.to_csv("./data/cleaned_validation_set.csv", index=False)
    print("Cleaned datasets saved.")
    
    # Step 3: Retrain model
    print("\nStep 3: Retraining model on cleaned data...")
    retrainer = ModelRetrainer()
    
    # Prepare datasets
    train_dataset, val_dataset = retrainer.prepare_data(clean_train_df, clean_val_df)
    
    # Train model
    output_dir = "./retrained_model_no_leakage"
    training_history = retrainer.train_model(train_dataset, val_dataset, output_dir)
    
    # Step 4: Validate retrained model
    print("\nStep 4: Validating retrained model...")
    validation_results = validate_retrained_model(output_dir, test_df)
    
    # Step 5: Compare with original model
    print("\nStep 5: Comparison Results")
    print("=" * 50)
    print("RETRAINED MODEL PERFORMANCE:")
    print(f"Accuracy: {validation_results['accuracy']:.4f}")
    print(f"Precision: {validation_results['precision']:.4f}")
    print(f"Recall: {validation_results['recall']:.4f}")
    print(f"F1-Score: {validation_results['f1_score']:.4f}")
    print(f"Average Confidence: {validation_results['avg_confidence']:.4f}")
    print(f"High Confidence Ratio: {validation_results['high_confidence_ratio']:.4f}")
    print(f"Overfitting Risk: {validation_results['overfitting_risk']}")
    
    # Step 6: Save results and recommendations
    results_summary = {
        'retrained_model_path': output_dir,
        'performance': validation_results,
        'training_samples': len(clean_train_df),
        'validation_samples': len(clean_val_df),
        'test_samples': len(test_df),
        'data_cleaning_applied': True,
        'regularization_applied': True
    }
    
    with open('./retraining_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nRetraining completed!")
    print(f"- Retrained model saved to: {output_dir}")
    print(f"- Results saved to: ./retraining_results.json")
    print(f"- Cleaned datasets saved to: ./data/cleaned_*_set.csv")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    
    if validation_results['overfitting_risk'] == 'LOW':
        print("✅ SUCCESS: Retrained model shows low overfitting risk")
        print(f"✅ Realistic performance: {validation_results['accuracy']:.1%} accuracy")
        print("✅ Ready for deployment in Enhanced AI+Crypto system")
    elif validation_results['overfitting_risk'] == 'MEDIUM':
        print("⚠️  MODERATE: Some overfitting detected, but improved")
        print("⚠️  Consider additional regularization or data augmentation")
    else:
        print("🚨 HIGH RISK: Model still shows overfitting")
        print("🚨 Need more aggressive regularization or more diverse training data")
    
    print(f"\nTo use the retrained model, update the model_path in your systems to:")
    print(f"model_path = '{output_dir}'")
    
    return results_summary

if __name__ == "__main__":
    results = main()
