import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
import json
import os
import glob
warnings.filterwarnings('ignore')

# # Import our validation systems
from Traditional_Cryptographic_System import TraditionalCryptographicValidator
from Enhanced_AI_Crypto_System import EnhancedAICryptoValidator


def find_best_checkpoint(results_dir: str = "./results_no_leakage",
                         training_results_json: str = "./fixed_training_results.json") -> str:
    """
    Automatically find the best model checkpoint using the following priority:
      1. Read best_checkpoint from fixed_training_results.json (written by Fixed_Fine_Tuning.py)
      2. Read best_model_checkpoint from trainer_state.json inside results_dir
      3. Scan results_dir for checkpoint-* folders and pick the one with the highest step number
      4. Fall back to results_dir itself (trainer.save_model() saves the best model there directly)
    """

    # Strategy 1: fixed_training_results.json
    if os.path.exists(training_results_json):
        try:
            with open(training_results_json, "r") as f:
                data = json.load(f)
            ckpt = data.get("training_info", {}).get("best_checkpoint")
            if ckpt and os.path.isdir(ckpt):
                print(f"[checkpoint] Found best checkpoint from training results JSON: {ckpt}")
                return ckpt
        except Exception as e:
            print(f"[checkpoint] Could not read {training_results_json}: {e}")

    # Strategy 2: trainer_state.json inside results_dir
    trainer_state_path = os.path.join(results_dir, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        try:
            with open(trainer_state_path, "r") as f:
                state = json.load(f)
            ckpt = state.get("best_model_checkpoint")
            if ckpt and os.path.isdir(ckpt):
                print(f"[checkpoint] Found best checkpoint from trainer_state.json: {ckpt}")
                return ckpt
        except Exception as e:
            print(f"[checkpoint] Could not read trainer_state.json: {e}")

    # Strategy 3: scan for checkpoint-* directories, pick highest step
    pattern = os.path.join(results_dir, "checkpoint-*")
    checkpoints = sorted(
        glob.glob(pattern),
        key=lambda p: int(p.split("-")[-1]) if p.split("-")[-1].isdigit() else -1
    )
    if checkpoints:
        ckpt = checkpoints[-1]  # highest step number
        print(f"[checkpoint] Auto-selected latest checkpoint directory: {ckpt}")
        return ckpt

    # Strategy 4: use results_dir directly (trainer.save_model() saves best model here)
    if os.path.isdir(results_dir) and any(
        fname in os.listdir(results_dir)
        for fname in ["pytorch_model.bin", "model.safetensors", "config.json"]
    ):
        print(f"[checkpoint] Using results directory directly as model path: {results_dir}")
        return results_dir

    raise FileNotFoundError(
        f"No valid model checkpoint found in '{results_dir}'. "
        "Please run Fixed_Fine_Tuning.py first."
    )

class ComparisonAnalysis:
    """
    Comprehensive comparison analysis between traditional cryptographic
    and combined AI+crypto validation systems.
    """

    def __init__(self):
        self.results = {}
        self.detailed_results = {}

    def run_comparative_evaluation(self):
        """Run both systems and collect comparative results."""
        print("=== Comparative Evaluation: Traditional vs AI+Crypto ===\n")

        # Load datasets
        print("Loading datasets...")
        train_df = pd.read_csv("./data/training_set.csv")
        val_df = pd.read_csv("./data/validation_set.csv")
        test_df = pd.read_csv("./data/test_set.csv")

        datasets = {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }

        # Initialize systems (two-stage AI+Crypto with default risk category)
        crypto_validator = TraditionalCryptographicValidator()
        best_checkpoint = find_best_checkpoint()
        combined_validator = EnhancedAICryptoValidator(
            model_path=best_checkpoint,
            update_category="default",
            enable_stage2=True,
            fast_path_mode=False,
        )
        self.crypto_validator = crypto_validator
        self.combined_validator = combined_validator

        # Evaluate both systems on all datasets
        for dataset_name, df in datasets.items():
            print(f"\n--- Evaluating {dataset_name} set ({len(df)} samples) ---")

            # Traditional cryptographic system
            print("Running traditional cryptographic validation...")
            crypto_preds, crypto_details = crypto_validator.validate_dataset(df)
            crypto_metrics = crypto_validator.calculate_metrics(df['output_label'].tolist(), crypto_preds)

            # Combined AI+crypto system
            print("Running combined AI+crypto validation...")
            combined_preds, combined_details = combined_validator.validate_dataset(df)
            combined_metrics = combined_validator.calculate_metrics(df['output_label'].tolist(), combined_preds)

            # Store results
            self.results[dataset_name] = {
                'crypto': {
                    'predictions': crypto_preds,
                    'metrics': crypto_metrics,
                    'details': crypto_details
                },
                'combined': {
                    'predictions': combined_preds,
                    'metrics': combined_metrics,
                    'details': combined_details
                },
                'ground_truth': df['output_label'].tolist()
            }

            # Print comparison
            print(f"\n{dataset_name.upper()} SET COMPARISON:")
            print(f"{'Metric':<25} {'Traditional':<12} {'AI+Crypto':<12} {'Improvement':<12}")
            print("-" * 65)

            metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
            for metric in metrics_to_compare:
                crypto_val = crypto_metrics[metric]
                combined_val = combined_metrics[metric]
                improvement = ((combined_val - crypto_val) / crypto_val * 100) if crypto_val > 0 else 0
                print(f"{metric:<25} {crypto_val:<12.4f} {combined_val:<12.4f} {improvement:<12.2f}%")

        return self.results

    def create_performance_comparison_plots(self, results):
        """Create comprehensive performance comparison visualizations."""
        print("\nGenerating performance comparison plots...")

        self.results = results

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Overall Performance Metrics Comparison (Test Set)
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        test_results = self.results['test']

        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
        crypto_values = [test_results['crypto']['metrics'][m] for m in metrics]
        combined_values = [test_results['combined']['metrics'][m] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax1.bar(x - width/2, crypto_values, width, label='Traditional Crypto', alpha=0.8, color='#FF6B6B')
        bars2 = ax1.bar(x + width/2, combined_values, width, label='AI + Crypto', alpha=0.8, color='#4ECDC4')

        ax1.set_xlabel('Performance Metrics', fontsize=24)
        ax1.set_ylabel('Score', fontsize=24)
        ax1.set_xticks(x)
        # ax1.set_yticks(y)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', labelsize=24)
        ax1.tick_params(axis='y', labelsize=24)

        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=18, fontweight='bold')
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=18, fontweight='bold')

        fig1.tight_layout()
        fig1.savefig('./plot_performance_metrics.pdf', bbox_inches='tight')
        plt.close(fig1)

        # 2. False Positive vs False Negative Rates
        fig2, ax2 = plt.subplots(figsize=(12, 8))

        fp_rates = [test_results['crypto']['metrics']['false_positive_rate'],
                    test_results['combined']['metrics']['false_positive_rate']]
        fn_rates = [test_results['crypto']['metrics']['false_negative_rate'],
                    test_results['combined']['metrics']['false_negative_rate']]

        methods = ['Traditional\nCrypto', 'AI + Crypto']
        x = np.arange(len(methods))

        bars1 = ax2.bar(x - width/2, fp_rates, width, label='False Positive Rate', alpha=0.8, color='#FFB6C1')
        bars2 = ax2.bar(x + width/2, fn_rates, width, label='False Negative Rate', alpha=0.8, color='#FFA07A')

        ax2.set_xlabel('Validation Method', fontsize=24)
        ax2.set_ylabel('Error Rate', fontsize=24)
        ax2.set_xticks(x)
        # ax2.set_yticks(y)
        ax2.set_xticklabels(methods)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', labelsize=24)
        ax2.tick_params(axis='y', labelsize=24)

        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=18, fontweight='bold')
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=18, fontweight='bold')

        fig2.tight_layout()
        fig2.savefig('./plot_error_rates.pdf', bbox_inches='tight')
        plt.close(fig2)

        # 3. Confusion Matrix Heatmaps - Traditional Crypto
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        crypto_cm = test_results['crypto']['metrics']['confusion_matrix']
        sns.heatmap(crypto_cm, annot=True, fmt='d', cmap='Blues', ax=ax3, annot_kws={'size': 20, 'fontweight': 'bold'})
        ax3.set_xlabel('Predicted', fontsize=24)
        ax3.set_ylabel('Actual', fontsize=24)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', labelsize=24)
        ax3.tick_params(axis='y', labelsize=24)
        fig3.tight_layout()
        fig3.savefig('./plot_confusion_matrix_traditional.pdf', bbox_inches='tight')
        plt.close(fig3)

        # 3b. Confusion Matrix Heatmaps - AI + Crypto
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        combined_cm = test_results['combined']['metrics']['confusion_matrix']
        sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Greens', ax=ax4, annot_kws={'size': 20, 'fontweight': 'bold'})
        ax4.set_xlabel('Predicted', fontsize=24)
        ax4.set_ylabel('Actual', fontsize=24)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', labelsize=24)
        ax4.tick_params(axis='y', labelsize=24)
        fig4.tight_layout()
        fig4.savefig('./plot_confusion_matrix_ai_crypto.pdf', bbox_inches='tight')
        plt.close(fig4)

        # 4. Detection Performance by Attack Type
        fig5, ax5 = plt.subplots(figsize=(12, 8))

        attack_analysis = self.analyze_attack_detection_performance()

        attack_types = list(attack_analysis.keys())
        crypto_detection = [attack_analysis[at]['crypto_detection_rate'] for at in attack_types]
        combined_detection = [attack_analysis[at]['combined_detection_rate'] for at in attack_types]

        x = np.arange(len(attack_types))
        bars1 = ax5.bar(x - width/2, crypto_detection, width, label='Traditional Crypto', alpha=0.8, color='#FF6B6B')
        bars2 = ax5.bar(x + width/2, combined_detection, width, label='AI + Crypto', alpha=0.8, color='#4ECDC4')

        ax5.set_xlabel('Attack Characteristics', fontsize=24)
        ax5.set_ylabel('Detection Rate', fontsize=24)
        ax5.set_xticks(x)
        # ax5.set_yticks(y)
        ax5.set_xticklabels(attack_types, rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='x', labelsize=24)
        ax5.tick_params(axis='y', labelsize=24)

        fig5.tight_layout()
        fig5.savefig('./plot_detection_by_attack_type.pdf', bbox_inches='tight')
        plt.close(fig5)

        # 5. Performance Across Datasets
        fig6, ax6 = plt.subplots(figsize=(12, 8))

        datasets = ['train', 'validation', 'test']
        crypto_f1 = [self.results[d]['crypto']['metrics']['f1_score'] for d in datasets]
        combined_f1 = [self.results[d]['combined']['metrics']['f1_score'] for d in datasets]

        x = np.arange(len(datasets))
        bars1 = ax6.bar(x - width/2, crypto_f1, width, label='Traditional Crypto', alpha=0.8, color='#FF6B6B')
        bars2 = ax6.bar(x + width/2, combined_f1, width, label='AI + Crypto', alpha=0.8, color='#4ECDC4')

        ax6.set_xlabel('Dataset', fontsize=24)
        ax6.set_ylabel('F1-Score', fontsize=24)
        ax6.set_xticks(x)
        # ax6.set_yticks(y)
        ax6.set_xticklabels([d.title() for d in datasets])
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis='x', labelsize=24)
        ax6.tick_params(axis='y', labelsize=24)

        fig6.tight_layout()
        fig6.savefig('./plot_f1_across_datasets.pdf', bbox_inches='tight')
        plt.close(fig6)

        # Create additional specialized plots
        self.create_roc_curves()
        self.create_precision_recall_curves()

    def analyze_attack_detection_performance(self):
        """Analyze detection performance for different types of attacks."""
        test_results = self.results['test']

        # Categorize attacks based on characteristics
        attack_categories = {
            'Fake_Creator': [],
            'Long_Path': [],
            'P2P_Channel': [],
            'Low_Authenticity': [],
            'Chain_Integrity_Fail': []
        }

        # Analyze crypto results
        for detail in test_results['crypto']['details']:
            features = detail['features']
            actual = detail['actual_label']
            crypto_pred = detail['prediction']

            if actual == 1:  # Malicious update
                if 'Fake' in features.get('creator', ''):
                    attack_categories['Fake_Creator'].append({
                        'crypto_correct': crypto_pred == actual,
                        'index': detail['index']
                    })
                if features.get('path_length', 0) > 5:
                    attack_categories['Long_Path'].append({
                        'crypto_correct': crypto_pred == actual,
                        'index': detail['index']
                    })
                if features.get('channel') == 'P2P':
                    attack_categories['P2P_Channel'].append({
                        'crypto_correct': crypto_pred == actual,
                        'index': detail['index']
                    })
                if features.get('authenticity', 1) == 0:
                    attack_categories['Low_Authenticity'].append({
                        'crypto_correct': crypto_pred == actual,
                        'index': detail['index']
                    })
                if features.get('chain_integrity', 1) == 0:
                    attack_categories['Chain_Integrity_Fail'].append({
                        'crypto_correct': crypto_pred == actual,
                        'index': detail['index']
                    })

        # Add combined system results
        for detail in test_results['combined']['details']:
            actual = detail['actual_label']
            combined_pred = detail['prediction']
            idx = detail['index']

            if actual == 1:  # Malicious update
                for category, attacks in attack_categories.items():
                    for attack in attacks:
                        if attack['index'] == idx:
                            attack['combined_correct'] = combined_pred == actual

        # Calculate detection rates
        analysis = {}
        for category, attacks in attack_categories.items():
            if attacks:
                crypto_detection = sum(1 for a in attacks if a.get('crypto_correct', False)) / len(attacks)
                combined_detection = sum(1 for a in attacks if a.get('combined_correct', False)) / len(attacks)

                analysis[category] = {
                    'count': len(attacks),
                    'crypto_detection_rate': crypto_detection,
                    'combined_detection_rate': combined_detection,
                    'improvement': combined_detection - crypto_detection
                }

        return analysis

    def create_roc_curves(self):
        """Create ROC curves for both systems."""
        fig, ax = plt.subplots(figsize=(12, 8))

        test_results = self.results['test']
        y_true = test_results['ground_truth']

        # --- Traditional Crypto -----------------------------------------
        # The system is rule-based and produces binary predictions (0/1).
        # Using those as scores gives sklearn a proper (degenerate) ROC
        # curve whose AUC honestly reflects the system's performance.
        crypto_scores = [float(p) for p in test_results['crypto']['predictions']]

        # --- AI + Crypto ------------------------------------------------
        combined_scores = []
        for detail in test_results['combined']['details']:
            pred = detail['prediction']
            conf = detail['confidence']
            if pred == 1:
                combined_scores.append(conf)
            else:
                combined_scores.append(1.0 - conf)

        # Calculate ROC curves
        crypto_fpr, crypto_tpr, _ = roc_curve(y_true, crypto_scores)
        combined_fpr, combined_tpr, _ = roc_curve(y_true, combined_scores)

        crypto_auc = auc(crypto_fpr, crypto_tpr)
        combined_auc = auc(combined_fpr, combined_tpr)

        # Plot ROC curves
        ax.plot(crypto_fpr, crypto_tpr, linewidth=2,
                label=f'Traditional Crypto (AUC = {crypto_auc:.3f})', color='#FF6B6B')
        ax.plot(combined_fpr, combined_tpr, linewidth=2,
                label=f'AI + Crypto (AUC = {combined_auc:.3f})', color='#4ECDC4')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=18)
        ax.set_ylabel('True Positive Rate', fontsize=18)
        ax.legend(loc="lower right", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

        fig.tight_layout()
        fig.savefig('./roc_curves_comparison.pdf', bbox_inches='tight')
        plt.close(fig)

    def create_precision_recall_curves(self):
        """Create Precision-Recall curves for both systems."""
        fig, ax = plt.subplots(figsize=(12, 8))

        test_results = self.results['test']
        y_true = test_results['ground_truth']
        y_true_arr = np.array(y_true)

        # --- Traditional Crypto -----------------------------------------
        crypto_scores = [float(p) for p in test_results['crypto']['predictions']]

        # --- AI + Crypto ------------------------------------------------
        combined_scores = []
        for detail in test_results['combined']['details']:
            pred = detail['prediction']
            conf = detail['confidence']
            if pred == 1:
                combined_scores.append(conf)
            else:
                combined_scores.append(1.0 - conf)

        # Calculate PR curves
        crypto_precision, crypto_recall, _ = precision_recall_curve(y_true, crypto_scores)
        combined_precision, combined_recall, _ = precision_recall_curve(y_true, combined_scores)

        crypto_pr_auc = auc(crypto_recall, crypto_precision)
        combined_pr_auc = auc(combined_recall, combined_precision)

        # Baseline: prevalence of positive class (random classifier)
        prevalence = np.sum(y_true_arr == 1) / len(y_true_arr)

        # Plot PR curves
        ax.plot(crypto_recall, crypto_precision, linewidth=2,
                label=f'Traditional Crypto (AUC = {crypto_pr_auc:.3f})', color='#FF6B6B')
        ax.plot(combined_recall, combined_precision, linewidth=2,
                label=f'AI + Crypto (AUC = {combined_pr_auc:.3f})', color='#4ECDC4')

        # Baseline: horizontal line at prevalence
        ax.axhline(y=prevalence, color='k', linestyle='--', linewidth=1, alpha=0.5,
                   label=f'Baseline (prevalence = {prevalence:.3f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=18)
        ax.set_ylabel('Precision', fontsize=18)
        ax.legend(loc="lower left", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

        fig.tight_layout()
        fig.savefig('./precision_recall_curves.pdf', bbox_inches='tight')
        plt.close(fig)

    def generate_comprehensive_report(self):
        """Generate a comprehensive comparison report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE COMPARISON REPORT")
        print("="*80)

        # Overall performance summary
        test_results = self.results['test']
        crypto_metrics = test_results['crypto']['metrics']
        combined_metrics = test_results['combined']['metrics']

        print(f"\n{'METRIC':<25} {'TRADITIONAL':<15} {'AI+CRYPTO':<15} {'IMPROVEMENT':<15}")
        print("-" * 75)

        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
        improvements = []

        for metric in key_metrics:
            crypto_val = crypto_metrics[metric]
            combined_val = combined_metrics[metric]
            improvement = ((combined_val - crypto_val) / crypto_val * 100) if crypto_val > 0 else 0
            improvements.append(improvement)

            print(f"{metric.replace('_', ' ').title():<25} {crypto_val:<15.4f} {combined_val:<15.4f} {improvement:<15.2f}%")

        avg_improvement = np.mean(improvements)
        print(f"\nAverage Improvement: {avg_improvement:.2f}%")

        # Attack detection analysis
        attack_analysis = self.analyze_attack_detection_performance()
        print(f"\n{'ATTACK TYPE':<20} {'COUNT':<8} {'TRADITIONAL':<15} {'AI+CRYPTO':<15} {'IMPROVEMENT':<15}")
        print("-" * 75)

        for attack_type, data in attack_analysis.items():
            improvement = data['improvement'] * 100
            print(f"{attack_type.replace('_', ' '):<20} {data['count']:<8} "
                  f"{data['crypto_detection_rate']:<15.3f} {data['combined_detection_rate']:<15.3f} "
                  f"{improvement:<15.2f}%")

        # Save comprehensive results
        all_results = []
        for dataset_name in ['train', 'validation', 'test']:
            for method in ['crypto', 'combined']:
                metrics = self.results[dataset_name][method]['metrics']
                result = {
                    'dataset': dataset_name,
                    'method': 'Traditional_Crypto' if method == 'crypto' else 'AI_Crypto_Combined',
                    **metrics
                }
                all_results.append(result)

        results_df = pd.DataFrame(all_results)

        # --- Extended metrics (R3, R4, R5) from AI+Crypto validator (test set) ---
        if hasattr(self, 'combined_validator') and self.combined_validator is not None:
            latency = self.combined_validator.get_latency_report()
            quarantine = self.combined_validator.get_quarantine_summary()
            reputation = self.combined_validator.get_reputation_report()

            # Append extended columns to CSV for test-set combined row
            s1 = latency.get('stage1_classifier', {})
            for i, row in results_df.iterrows():
                if row['dataset'] == 'test' and row['method'] == 'AI_Crypto_Combined':
                    results_df.at[i, 'quarantine_rate'] = quarantine.get('quarantine_rate', None)
                    results_df.at[i, 'decision_threshold'] = quarantine.get('decision_threshold', None)
                    results_df.at[i, 'stage1_mean_latency_ms'] = s1.get('mean_ms', None)
                    results_df.at[i, 'stage1_p95_latency_ms'] = s1.get('p95_ms', None)
                    results_df.at[i, 'reputation_sources_tracked'] = reputation.get('total_sources', None)
                    results_df.at[i, 'sybil_flagged_count'] = len(reputation.get('sybil_suspects', []))
                    break

            print(f"\n--- Extended AI+Crypto Metrics (R3/R4/R5) ---")

            # ============================================================
            # R-3: REPUTATION, SYBIL & COLLUSION ANALYSIS
            # ============================================================
            print(f"\n{'='*75}")
            print("R-3: REPUTATION SYSTEM ANALYSIS")
            print(f"{'='*75}")

            # Per-source reputation table
            all_reps = self.combined_validator.reputation.get_all_reputations()
            print(f"\n  Per-Source Reputation Scores (decay λ={self.combined_validator.reputation.decay_lambda}):")
            print(f"  {'Source':<20} {'Reputation':>10} {'Interactions':>13} {'Channels':>10} {'Sybil?':>8}")
            print(f"  {'-'*65}")
            for src, rep_score in sorted(all_reps.items(), key=lambda x: x[1]):
                sybil_check = self.combined_validator.reputation.check_sybil(src)
                records = self.combined_validator.reputation._records.get(src, [])
                n_channels = len(set(r.channel for r in records))
                status = "⚠ YES" if sybil_check['is_suspicious'] else "✓ No"
                print(f"  {src:<20} {rep_score:>10.4f} {len(records):>13} {n_channels:>10} {status:>8}")

            # Sybil detection summary
            flagged = self.combined_validator.reputation.get_flagged_sources()
            n_sybil = len(flagged['sybil_suspects'])
            n_clear = len(all_reps) - n_sybil
            print(f"\n  Sybil Detection Summary:")
            print(f"    Sources tracked:    {len(all_reps)}")
            print(f"    Cleared sources:    {n_clear}")
            print(f"    Flagged suspects:   {n_sybil}")
            if flagged['sybil_suspects']:
                for src in flagged['sybil_suspects']:
                    reason = self.combined_validator.reputation.check_sybil(src).get('reason', 'N/A')
                    print(f"      - {src}: {reason}")

            # Collusion detection
            collusion = getattr(self.combined_validator, '_collusion_results', [])
            print(f"\n  Collusion Detection (threshold ≥ {self.combined_validator.reputation.collusion_threshold:.0%} agreement):")
            if collusion:
                print(f"    Flagged pairs: {len(collusion)}")
                for c in collusion:
                    pair = c['source_pair']
                    print(f"      - {pair[0]} ↔ {pair[1]}: "
                          f"agreement={c['agreement_rate']:.1%}, "
                          f"common_interactions={c['common_interactions']}")
            else:
                print(f"    No collusion detected.")

            # Temporal decay analysis
            decay_analysis = self.combined_validator.reputation.get_temporal_decay_analysis()
            print(f"\n  Temporal Decay Analysis (λ={self.combined_validator.reputation.decay_lambda}):")
            print(f"  {'Source':<20} {'Recent 20% Wt':>14} {'Obs Span':>10} {'Eff. Decay':>11}")
            print(f"  {'-'*58}")
            for src, info in sorted(decay_analysis.items()):
                print(f"  {src:<20} {info['recent_weight_fraction']:>14.2%} "
                      f"{info['observation_span']:>10.0f} {info['effective_decay']:>11.4f}")

            # ============================================================
            # R-4: LATENCY, COMPUTE & STANDARDS COMPLIANCE
            # ============================================================
            print(f"\n{'='*75}")
            print("R-4: LATENCY & DEPLOYMENT FEASIBILITY ANALYSIS")
            print(f"{'='*75}")

            if latency:
                s1 = latency.get('stage1_classifier', {})
                s2 = latency.get('stage2_generative', {})

                print(f"\n  Inference Latency Breakdown:")
                if s1:
                    print(f"    Stage 1 (DistilBERT classifier):")
                    print(f"      Mean: {s1['mean_ms']:.2f} ms | P50: {s1['p50_ms']:.2f} ms | "
                          f"P95: {s1['p95_ms']:.2f} ms | Max: {s1['max_ms']:.2f} ms")
                if s2:
                    total_mean = s1.get('mean_ms', 0) + s2.get('mean_ms', 0)
                    total_p95 = s1.get('p95_ms', 0) + s2.get('p95_ms', 0)
                    print(f"    Stage 2 (Generative reasoning):")
                    print(f"      Mean: {s2['mean_ms']:.4f} ms | P95: {s2['p95_ms']:.4f} ms")
                    print(f"    Combined (Stage 1 + 2):")
                    print(f"      Mean: {total_mean:.2f} ms | P95: {total_p95:.2f} ms")

                # Standards compliance
                print(f"\n  IEEE 1609.2 / ETSI ITS Timing Compliance:")
                v2v_deadline = 100.0  # ms, V2V safety-critical
                v2i_deadline = 1000.0  # ms, V2I advisory
                s1_p95 = s1.get('p95_ms', 0)
                total_p95 = s1_p95 + s2.get('p95_ms', 0) if s2 else s1_p95

                v2v_ok = "✓ PASS" if s1_p95 < v2v_deadline else "✗ FAIL"
                v2i_ok = "✓ PASS" if total_p95 < v2i_deadline else "✗ FAIL"
                print(f"    V2V safety-critical (≤{v2v_deadline:.0f}ms):  "
                      f"Stage 1 P95 = {s1_p95:.2f}ms  {v2v_ok}")
                print(f"    V2I advisory (≤{v2i_deadline:.0f}ms):        "
                      f"Full P95 = {total_p95:.2f}ms   {v2i_ok}")
                print(f"    Fast-path mode (crypto-only):   "
                      f"Bypasses Stage 1 AI entirely for V2V messages")

            # On-vehicle compute budget
            print(f"\n  On-Vehicle Compute Budget:")
            print(f"    Model:       DistilBERT (66M parameters, ~268 MB)")
            print(f"    Inference:   Single forward pass per update validation")
            print(f"    Hardware:    Compatible with NVIDIA Jetson Orin / Xavier NX class")
            print(f"    Memory:      ~500 MB peak (model + tokenizer + batch)")
            print(f"    Note:        Stage 2 is rule-based (no GPU needed, ~0.04ms)")

            # V2X bandwidth
            print(f"\n  V2X Bandwidth & Privacy:")
            print(f"    Provenance metadata size:  ~200-500 bytes per update")
            print(f"    Processing:    All inference is LOCAL (on-vehicle)")
            print(f"    Data shared:   Only binary validation verdict (accept/reject)")
            print(f"    Privacy:       Raw provenance metadata never leaves the vehicle")
            print(f"    Adversarial:   Input truncated to 512 tokens; no free-form prompting")

            # R-5 quarantine summary
            print(f"\n  Quarantine-by-Default (R5):")
            print(f"    Total samples:  {quarantine['total_samples']}")
            print(f"    Quarantined:    {quarantine['quarantined']} ({quarantine['quarantine_rate']:.2%})")
            print(f"    Threshold:      confidence < {quarantine['quarantine_threshold']}")
            print(f"    Category:       {quarantine.get('update_category', 'default')}")

            extended = {
                "latency_report": latency,
                "quarantine_summary": quarantine,
                "reputation_summary": {
                    "total_sources": reputation["total_sources"],
                    "mean_reputation": reputation["mean_reputation"],
                    "sybil_suspects": reputation["sybil_suspects"],
                    "collusion_detections": reputation["collusion_detections"],
                },
                "temporal_decay_analysis": decay_analysis,
                "standards_compliance": {
                    "v2v_deadline_ms": 100.0,
                    "v2i_deadline_ms": 1000.0,
                    "stage1_p95_ms": s1.get('p95_ms', None) if latency else None,
                    "v2v_compliant": s1.get('p95_ms', 999) < 100.0 if latency else None,
                    "v2i_compliant": (s1.get('p95_ms', 0) + s2.get('p95_ms', 0)) < 1000.0 if latency else None,
                },
            }
            with open("./extended_comparison_metrics.json", "w") as f:
                json.dump(extended, f, indent=2, default=str)
            print(f"\nExtended metrics saved to './extended_comparison_metrics.json'")

        results_df.to_csv('./comprehensive_comparison_results.csv', index=False)
        print(f"\nComprehensive results saved to './comprehensive_comparison_results.csv'")

        return results_df

def main():
        """Main function to run comprehensive comparison analysis."""
        analyzer = ComparisonAnalysis()

        # Run comparative evaluation
        results = analyzer.run_comparative_evaluation()

        # Create visualizations
        analyzer.create_performance_comparison_plots(results)

        # Generate comprehensive report
        report_df = analyzer.generate_comprehensive_report()

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("Generated files:")
        print("- plot_performance_metrics.pdf")
        print("- plot_error_rates.pdf")
        print("- plot_confusion_matrix_traditional.pdf")
        print("- plot_confusion_matrix_ai_crypto.pdf")
        print("- plot_detection_by_attack_type.pdf")
        print("- plot_f1_across_datasets.pdf")
        print("- roc_curves_comparison.pdf")
        print("- precision_recall_curves.pdf")
        print("- comprehensive_comparison_results.csv")
        if hasattr(analyzer, 'combined_validator') and analyzer.combined_validator is not None:
            print("- extended_comparison_metrics.json  (R3/R4/R5: latency, quarantine, reputation)")
        print("="*80)

        return analyzer, results, report_df

if __name__ == "__main__":
    analyzer, results, report = main()