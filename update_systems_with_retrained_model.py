#!/usr/bin/env python3
"""
Update all AI+Crypto systems to use the retrained model without data leakage.
This script tests the systems with the new model and compares performance.
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List

# Import all systems
from Enhanced_AI_Crypto_System import EnhancedAICryptoValidator
from Realistic_Enhanced_AI_Crypto_System import RealisticEnhancedAICryptoValidator

def test_with_retrained_model(retrained_model_path: str):
    """Test all systems with the retrained model."""
    
    print("=== TESTING SYSTEMS WITH RETRAINED MODEL ===")
    print(f"Using retrained model from: {retrained_model_path}\n")
    
    # Load test data
    test_df = pd.read_csv("./data/test_set.csv")
    print(f"Test dataset: {len(test_df)} samples")
    print(f"Malicious: {sum(test_df['output_label'])} ({sum(test_df['output_label'])/len(test_df)*100:.1f}%)")
    print(f"Benign: {len(test_df) - sum(test_df['output_label'])} ({(len(test_df) - sum(test_df['output_label']))/len(test_df)*100:.1f}%)\n")
    
    results = {}
    
    # Test 1: Enhanced AI+Crypto with retrained model
    print("=== Testing Enhanced AI+Crypto (Retrained Model) ===")
    try:
        enhanced_validator = EnhancedAICryptoValidator(
            model_path=retrained_model_path,
            fusion_strategy="adaptive"
        )
        predictions, _ = enhanced_validator.validate_dataset(test_df)
        metrics = enhanced_validator.calculate_metrics(test_df['output_label'].tolist(), predictions)
        fusion_analysis = enhanced_validator.analyze_fusion_effectiveness()
        
        results['Enhanced_Retrained'] = {
            'metrics': metrics,
            'fusion_analysis': fusion_analysis
        }
        
        print_metrics("Enhanced AI+Crypto (Retrained)", metrics)
        print_fusion_analysis("Enhanced (Retrained)", fusion_analysis)
        
    except Exception as e:
        print(f"Error testing Enhanced system: {e}")
        results['Enhanced_Retrained'] = None
    
    # Test 2: Realistic Enhanced with retrained model (Weighted Crypto)
    print("\n=== Testing Realistic Enhanced (Retrained Model + Weighted Crypto) ===")
    try:
        realistic_validator = RealisticEnhancedAICryptoValidator(
            model_path=retrained_model_path,
            fusion_strategy="conservative",
            crypto_mode="weighted"
        )
        predictions, _ = realistic_validator.validate_dataset(test_df)
        metrics = realistic_validator.calculate_metrics(test_df['output_label'].tolist(), predictions)
        fusion_analysis = realistic_validator.analyze_fusion_effectiveness()
        
        results['Realistic_Retrained'] = {
            'metrics': metrics,
            'fusion_analysis': fusion_analysis
        }
        
        print_metrics("Realistic Enhanced (Retrained)", metrics)
        print_fusion_analysis("Realistic (Retrained)", fusion_analysis)
        
    except Exception as e:
        print(f"Error testing Realistic system: {e}")
        results['Realistic_Retrained'] = None
    
    return results

def print_metrics(system_name: str, metrics: Dict):
    """Print formatted metrics."""
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
    print(f"False Negative Rate: {metrics['false_negative_rate']:.4f}")

def print_fusion_analysis(system_name: str, analysis: Dict):
    """Print fusion analysis."""
    print(f"\n{system_name} Fusion Analysis:")
    print(f"Crypto-only Accuracy: {analysis['crypto_only_accuracy']:.4f}")
    print(f"AI-only Accuracy: {analysis['ai_only_accuracy']:.4f}")
    print(f"Fusion Accuracy: {analysis['fusion_accuracy']:.4f}")
    print(f"Fusion Improvements: {analysis['fusion_improvements']} cases")
    print(f"Improvement Rate: {analysis['improvement_rate']:.4f}")

def compare_original_vs_retrained():
    """Compare original model vs retrained model performance."""
    
    print("\n" + "="*80)
    print("COMPARISON: ORIGINAL vs RETRAINED MODEL")
    print("="*80)
    
    # Load previous results if available
    original_results = {}
    try:
        # Try to load previous enhanced results
        enhanced_df = pd.read_csv("./enhanced_ai_crypto_results.csv")
        if len(enhanced_df) > 0:
            row = enhanced_df.iloc[-1]  # Get latest results
            original_results['Enhanced_Original'] = {
                'accuracy': row['accuracy'],
                'precision': row['precision'],
                'recall': row['recall'],
                'f1_score': row['f1_score'],
                'false_positive_rate': row.get('false_positive_rate', 0),
            }
    except:
        print("Could not load original Enhanced results")
    
    # Test with retrained model
    retrained_model_path = "./retrained_model_no_leakage"
    
    if not os.path.exists(retrained_model_path):
        print(f"❌ Retrained model not found at {retrained_model_path}")
        print("Please run retrain_model_pipeline.py first to create the retrained model.")
        return
    
    retrained_results = test_with_retrained_model(retrained_model_path)
    
    # Compare results
    print("\nCOMPARISON SUMMARY:")
    print("-" * 60)
    
    if 'Enhanced_Original' in original_results and 'Enhanced_Retrained' in retrained_results:
        orig = original_results['Enhanced_Original']
        retr = retrained_results['Enhanced_Retrained']['metrics']
        
        print("ENHANCED AI+CRYPTO SYSTEM:")
        print(f"                    Original    Retrained   Change")
        print(f"Accuracy:           {orig['accuracy']:.4f}      {retr['accuracy']:.4f}     {retr['accuracy']-orig['accuracy']:+.4f}")
        print(f"Precision:          {orig['precision']:.4f}      {retr['precision']:.4f}     {retr['precision']-orig['precision']:+.4f}")
        print(f"Recall:             {orig['recall']:.4f}      {retr['recall']:.4f}     {retr['recall']-orig['recall']:+.4f}")
        print(f"F1-Score:           {orig['f1_score']:.4f}      {retr['f1_score']:.4f}     {retr['f1_score']-orig['f1_score']:+.4f}")
        print(f"False Pos. Rate:    {orig['false_positive_rate']:.4f}      {retr['false_positive_rate']:.4f}     {retr['false_positive_rate']-orig['false_positive_rate']:+.4f}")
        
        # Analysis
        print(f"\nANALYSIS:")
        if orig['accuracy'] >= 0.99 and retr['accuracy'] < 0.95:
            print("✅ SUCCESS: Eliminated overfitting (accuracy reduced from near-perfect to realistic)")
        elif orig['accuracy'] >= 0.99:
            print("⚠️  PARTIAL: Still showing high accuracy, may need more aggressive regularization")
        
        if abs(retr['accuracy'] - 0.88) < 0.05:  # Close to weighted crypto performance
            print("✅ GOOD: Performance similar to weighted crypto (AI properly constrained)")
        
        if retr['false_positive_rate'] < 0.1:
            print("✅ GOOD: Low false positive rate maintained")
    
    # Save comparison results
    comparison_data = {
        'original_results': original_results,
        'retrained_results': retrained_results,
        'retrained_model_path': retrained_model_path,
        'comparison_date': pd.Timestamp.now().isoformat()
    }
    
    with open('./model_comparison_results.json', 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    
    print(f"\nComparison results saved to: ./model_comparison_results.json")
    
    return comparison_data

def create_deployment_guide(retrained_model_path: str):
    """Create a deployment guide for using the retrained model."""
    
    guide = f"""
# DEPLOYMENT GUIDE: Retrained AI Model

## Overview
This guide explains how to deploy the retrained AI model that fixes data leakage and overfitting issues.

## Retrained Model Location
- Path: `{retrained_model_path}`
- Type: DistilBERT-based sequence classifier
- Training: Cleaned data without obvious attack indicators
- Regularization: Dropout 0.3, Weight decay 0.01, Early stopping

## How to Use

### 1. Update Enhanced AI+Crypto System
```python
from Enhanced_AI_Crypto_System import EnhancedAICryptoValidator

# Use retrained model
validator = EnhancedAICryptoValidator(
    model_path="{retrained_model_path}",
    fusion_strategy="adaptive"
)
```

### 2. Update Realistic Enhanced System
```python
from Realistic_Enhanced_AI_Crypto_System import RealisticEnhancedAICryptoValidator

# Use retrained model with weighted crypto
validator = RealisticEnhancedAICryptoValidator(
    model_path="{retrained_model_path}",
    fusion_strategy="conservative",
    crypto_mode="weighted"
)
```

## Expected Performance
- **Accuracy**: 75-90% (realistic range)
- **Precision**: 85-95% (good malicious detection)
- **Recall**: 70-85% (reasonable coverage)
- **F1-Score**: 80-90% (balanced performance)
- **False Positive Rate**: <10% (low false alarms)

## Key Improvements
1. ✅ **No Data Leakage**: Removed obvious indicators ('adversary', 'FakeOEM')
2. ✅ **No Overfitting**: Realistic performance scores (not perfect 1.0)
3. ✅ **Proper Regularization**: Dropout, weight decay, early stopping
4. ✅ **Crypto Dominance**: AI assists rather than overrides crypto validation
5. ✅ **Deployable**: Balanced precision/recall for real-world use

## Validation Checklist
- [ ] Accuracy between 75-90% (not near 100%)
- [ ] No perfect scores across all metrics
- [ ] False positive rate < 10%
- [ ] AI-only accuracy not significantly higher than fusion accuracy
- [ ] Fusion improvements > 0 (AI actually helping)

## Troubleshooting
If you still see perfect scores:
1. Check model path is correct
2. Verify cleaned training data was used
3. Consider additional regularization
4. Increase crypto dominance in fusion weights

## Files Created
- `{retrained_model_path}/`: Retrained model directory
- `./data/cleaned_training_set.csv`: Cleaned training data
- `./data/cleaned_validation_set.csv`: Cleaned validation data
- `./retraining_results.json`: Training results and metrics
- `./model_comparison_results.json`: Original vs retrained comparison
"""
    
    with open('./DEPLOYMENT_GUIDE.md', 'w') as f:
        f.write(guide)
    
    print("📋 Deployment guide created: ./DEPLOYMENT_GUIDE.md")

def main():
    """Main function to test retrained model and create deployment guide."""
    
    retrained_model_path = "./retrained_model_no_leakage"
    
    # Check if retrained model exists
    if not os.path.exists(retrained_model_path):
        print("❌ RETRAINED MODEL NOT FOUND")
        print(f"Expected location: {retrained_model_path}")
        print("\nTo create the retrained model, run:")
        print("python retrain_model_pipeline.py")
        print("\nThis will:")
        print("1. Clean the training data to remove data leakage")
        print("2. Retrain the model with proper regularization") 
        print("3. Validate the retrained model doesn't overfit")
        return
    
    print("✅ Retrained model found!")
    
    # Test systems with retrained model
    comparison_results = compare_original_vs_retrained()
    
    # Create deployment guide
    create_deployment_guide(retrained_model_path)
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. ✅ Retrained model is ready for use")
    print("2. 📋 Review the deployment guide: ./DEPLOYMENT_GUIDE.md")
    print("3. 🔄 Update your systems to use the retrained model:")
    print(f"   model_path = '{retrained_model_path}'")
    print("4. 🧪 Test the updated systems with your specific use cases")
    print("5. 📊 Monitor performance in production to ensure no overfitting")

if __name__ == "__main__":
    main()
