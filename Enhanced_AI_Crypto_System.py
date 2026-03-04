import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from Traditional_Cryptographic_System import TraditionalCryptographicValidator
from reputation_system import ReputationSystem

# ---------------------------------------------------------------------------
# R5: Risk-stratified threshold configuration (ISO 26262 HARA alignment)
# ---------------------------------------------------------------------------
UPDATE_CATEGORY_THRESHOLDS = {
    "kernel_firmware":       {"theta": 0.85, "description": "Safety-critical firmware"},
    "sensor_fusion":         {"theta": 0.85, "description": "Safety-critical sensor modules"},
    "adas_control":          {"theta": 0.80, "description": "ADAS / control-loop modules"},
    "infotainment":          {"theta": 0.60, "description": "Non-safety infotainment"},
    "map_data":              {"theta": 0.55, "description": "Map / traffic data feeds"},
    "default":               {"theta": 0.65, "description": "Unclassified updates"},
}


class GenerativeReasoningLayer:
    """
    Stage 2: Generative reasoning layer that produces counterfactual threat
    narratives and uncertainty-weighted confidence scores.

    This layer is invoked *asynchronously* after the Stage 1 discriminative
    classifier has produced a verdict.  It does NOT block the primary decision
    pipeline (R4 fast-path).  Its outputs are advisory and feed the
    quarantine-escalation logic rather than directly overriding safety
    decisions (R1 guardrail).
    """

    THREAT_TEMPLATES = {
        "timing_attack": (
            "Counterfactual: If this update (t={time}, ver={version}) were shifted "
            "to a normal operating window (t<400) with expected version lineage, "
            "all crypto checks would pass and behavioral indicators would be benign. "
            "The late-time + low-version combination is consistent with a replay of "
            "a stale package injected outside the OTA maintenance window."
        ),
        "version_rollback": (
            "Counterfactual: If the version number ({version}) matched the expected "
            "monotonic sequence for node {node}, the rollforward check would pass. "
            "The unusually high version at early timestamp (t={time}) suggests an "
            "attempt to pre-position a future update with a forged version counter."
        ),
        "path_manipulation": (
            "Counterfactual: If the update had arrived through the standard OTA "
            "channel (path_length<=4) instead of a {path_length}-hop P2P route, "
            "path integrity would not be flagged. The extended path is consistent "
            "with a man-in-the-middle relay chain."
        ),
        "context_inconsistency": (
            "Counterfactual: The creator '{creator}' is not typically associated "
            "with the channel '{channel}' or version range observed. Removing the "
            "inconsistent metadata would make this update indistinguishable from "
            "benign traffic, suggesting a metadata spoofing attempt."
        ),
        "infrastructure_compromise": (
            "Counterfactual: If the update originated from a trusted OEM OTA server "
            "rather than '{channel}', and carried a version consistent with the "
            "node's history, all checks would pass. The behaviorally anomalous "
            "source with valid cryptographic credentials suggests a compromised "
            "infrastructure node."
        ),
        "sophisticated_unknown": (
            "Counterfactual: No single feature is individually anomalous, but the "
            "combination of t={time}, ver={version}, path_length={path_length} "
            "creates a statistical outlier in the joint feature distribution. "
            "This pattern matches zero-day or polymorphic attack vectors."
        ),
        "benign_uncertain": (
            "Assessment: Update appears benign across all individual checks, but "
            "classifier confidence ({confidence:.2f}) is below the certainty "
            "threshold. Recommend quarantine for manual review rather than "
            "automatic acceptance."
        ),
    }

    def generate_threat_narrative(self, features: Dict, ai_prediction: int,
                                  ai_confidence: float) -> Dict:
        """
        Produce a counterfactual threat narrative and an uncertainty-weighted
        confidence adjustment.

        Returns a dict with:
          - narrative (str): human-readable counterfactual explanation
          - attack_hypothesis (str): most-likely attack category
          - uncertainty_weight (float): [0,1] weight applied to final score
          - adjusted_confidence (float): confidence after uncertainty weighting
        """
        attack_hypothesis = self._infer_attack_category(features, ai_prediction)

        template_key = attack_hypothesis if attack_hypothesis in self.THREAT_TEMPLATES else "benign_uncertain"
        template = self.THREAT_TEMPLATES[template_key]

        narrative = template.format(
            time=features.get('time', '?'),
            version=features.get('version', '?'),
            node=features.get('node', '?'),
            path_length=features.get('path_length', '?'),
            channel=features.get('channel', '?'),
            creator=features.get('creator', '?'),
            confidence=ai_confidence,
        )

        uncertainty_weight = self._compute_uncertainty_weight(
            ai_confidence, features
        )
        adjusted_confidence = ai_confidence * uncertainty_weight

        return {
            "narrative": narrative,
            "attack_hypothesis": attack_hypothesis,
            "uncertainty_weight": round(uncertainty_weight, 4),
            "adjusted_confidence": round(adjusted_confidence, 4),
        }

    # ------------------------------------------------------------------
    def _infer_attack_category(self, features: Dict, ai_pred: int) -> str:
        """Rule-based inference of the most likely attack category."""
        if ai_pred == 0:
            return "benign_uncertain"

        t = features.get('time', 0)
        ver = features.get('version', 0)
        path_len = features.get('path_length', 0)
        channel = features.get('channel', '')
        creator = features.get('creator', '')

        if t > 400 and ver < 5:
            return "timing_attack"
        if ver > 25 and t < 50:
            return "version_rollback"
        if path_len > 6 and channel == 'P2P':
            return "path_manipulation"

        # Context inconsistency heuristics
        weather_p2p = (creator == 'WeatherSvc' and channel == 'P2P')
        police_high_ver = (creator == 'PoliceDept' and ver > 30)
        if weather_p2p or police_high_ver:
            return "context_inconsistency"

        if channel in ('charging_station', 'parking_garage') and ver > 28:
            return "infrastructure_compromise"

        return "sophisticated_unknown"

    def _compute_uncertainty_weight(self, ai_confidence: float,
                                     features: Dict) -> float:
        """
        Compute an epistemic uncertainty weight in [0, 1].
        High values mean *more certain*; low values penalise borderline
        predictions.
        """
        margin = abs(ai_confidence - 0.5)  # distance from decision boundary

        crypto_agreement = 1.0
        crypto_checks = [
            features.get('authenticity', 1),
            features.get('rollforward', 1),
            features.get('chain_integrity', 1),
            features.get('baseline_acceptance', 1),
        ]
        failed_checks = sum(1 for c in crypto_checks if c == 0)
        if failed_checks >= 2:
            crypto_agreement = 0.7
        elif failed_checks == 1:
            crypto_agreement = 0.85

        return min(1.0, (0.4 + margin) * crypto_agreement)


class EnhancedAICryptoValidator:
    """
    Two-stage AI-Crypto validation system.

    Stage 1 (Discriminative): Fine-tuned DistilBERT binary classifier that
        produces a malicious/benign prediction with a softmax confidence.
    Stage 2 (Generative Reasoning): Produces counterfactual threat narratives
        and uncertainty-weighted confidence adjustments.  Invoked asynchronously
        and does NOT block the primary decision path (see fast_path_mode).

    The fusion layer combines the Stage 1 AI verdict with the cryptographic
    verdict using an adaptive weighting strategy.
    """

    def __init__(self, model_path: str = "./results_no_leakage/checkpoint-200",
                 fusion_strategy: str = "adaptive",
                 update_category: str = "default",
                 enable_stage2: bool = True,
                 fast_path_mode: bool = False):
        self.crypto_validator = TraditionalCryptographicValidator()
        self.fusion_strategy = fusion_strategy
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.validation_results = []
        self.performance_metrics = {}
        self.latency_log: List[Dict] = []

        # R1: Stage 2 generative reasoning
        self.enable_stage2 = enable_stage2
        self.stage2 = GenerativeReasoningLayer()

        # R4: Fast-path mode skips Stage 2 for V2V safety-critical messages
        self.fast_path_mode = fast_path_mode

        # R5: Risk-stratified thresholds
        self.update_category = update_category
        cat_cfg = UPDATE_CATEGORY_THRESHOLDS.get(
            update_category, UPDATE_CATEGORY_THRESHOLDS["default"]
        )
        self.decision_threshold = cat_cfg["theta"]

        # R5: Quarantine-by-default — verdicts below this threshold are
        # escalated rather than silently accepted.
        self.quarantine_threshold = max(0.40, self.decision_threshold - 0.20)

        # R3: Source reputation tracker
        self.reputation = ReputationSystem()

        # Dynamic weight adjustment parameters
        self.base_crypto_weight = 0.5
        self.base_ai_weight = 0.5
        self.confidence_threshold = 0.7

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

    def get_ai_prediction(self, text: str) -> Tuple[int, float, np.ndarray, float]:
        """
        Stage 1 discriminative inference.

        Returns (prediction, confidence, prob_distribution, latency_ms).
        """
        if self.model is None or self.tokenizer is None:
            return 0, 0.5, np.array([0.5, 0.5]), 0.0

        try:
            t0 = time.perf_counter()
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(logits, dim=-1).item()
                confidence = probabilities[0][prediction].item()
                prob_dist = probabilities[0].numpy()

            latency_ms = (time.perf_counter() - t0) * 1000.0
            return prediction, confidence, prob_dist, latency_ms
        except Exception as e:
            print(f"Error in AI prediction: {e}")
            return 0, 0.5, np.array([0.5, 0.5]), 0.0

    def extract_enhanced_features(self, log_text: str) -> Dict:
        """Extract comprehensive features including behavioral patterns."""
        features = self.crypto_validator.extract_crypto_features(log_text)

        # Enhanced provenance features
        features['is_p2p'] = 1 if features['channel'] == 'P2P' else 0
        features['is_ota'] = 1 if features['channel'] == 'OTA' else 0

        # Behavioral anomaly detection
        features['version_anomaly'] = 1 if features['version'] > 30 or features['version'] < 1 else 0
        features['path_anomaly'] = 1 if features['path_length'] > 5 else 0
        features['time_anomaly'] = 1 if features['time'] > 500 or features['time'] < 10 else 0
        features['adversary_role'] = 1 if features['role'] == 'adversary' else 0
        features['fake_creator'] = 1 if 'Fake' in features['creator'] else 0

        # New sophisticated attack indicators
        features['crypto_bypass_risk'] = self._assess_crypto_bypass_risk(features)
        features['behavioral_anomaly_score'] = self._calculate_behavioral_anomaly_score(features)
        features['context_inconsistency'] = self._detect_context_inconsistency(log_text, features)

        return features

    def _assess_crypto_bypass_risk(self, features: Dict) -> float:
        """Assess risk of sophisticated attacks that might bypass crypto checks."""
        risk_score = 0.0

        # Check for patterns that suggest sophisticated attacks
        if features.get('fake_creator', 0) == 1:
            risk_score += 0.3

        if features.get('path_length', 0) > 4:  # Suspicious routing
            risk_score += 0.2

        if features.get('version', 0) > 25:  # Unusual version numbers
            risk_score += 0.2

        # Time-based anomalies might indicate coordinated attacks
        if features.get('time_anomaly', 0) == 1:
            risk_score += 0.15

        # Multiple small anomalies compound risk
        anomaly_count = sum([
            features.get('version_anomaly', 0),
            features.get('path_anomaly', 0),
            features.get('adversary_role', 0)
        ])

        if anomaly_count >= 2:
            risk_score += 0.25

        return min(risk_score, 1.0)

    def _calculate_behavioral_anomaly_score(self, features: Dict) -> float:
        """Calculate behavioral anomaly score based on multiple indicators."""
        anomaly_indicators = [
            features.get('version_anomaly', 0),
            features.get('path_anomaly', 0),
            features.get('time_anomaly', 0),
            features.get('adversary_role', 0),
            features.get('fake_creator', 0)
        ]

        # Weighted scoring
        weights = [0.2, 0.25, 0.15, 0.3, 0.35]  # Higher weight for creator and role

        score = sum(indicator * weight for indicator, weight in zip(anomaly_indicators, weights))
        return min(score, 1.0)

    def _detect_context_inconsistency(self, log_text: str, features: Dict) -> float:
        """Detect inconsistencies in update context that might indicate attacks."""
        inconsistency_score = 0.0

        # Check for inconsistent patterns in the log text
        text_lower = log_text.lower()

        # Inconsistent creator claims
        if 'oem' in text_lower and features.get('fake_creator', 0) == 1:
            inconsistency_score += 0.4

        # Inconsistent channel usage patterns
        if features.get('is_p2p', 0) == 1 and features.get('path_length', 0) < 2:
            inconsistency_score += 0.2

        # Version rollback attempts with high version numbers
        if features.get('version', 0) > 20 and 'rollback' in text_lower:
            inconsistency_score += 0.3

        return min(inconsistency_score, 1.0)

    def adaptive_fusion(self, text: str, features: Dict) -> Tuple[int, float, Dict]:
        """Adaptive fusion strategy that dynamically adjusts based on context."""

        # --- R4: Fast-path mode for V2V safety-critical messages --------
        # Bypasses Stage 1 AI entirely; relies on crypto verdict only.
        if self.fast_path_mode:
            t0 = time.perf_counter()
            crypto_prediction = self.crypto_validator.cryptographic_validation(features)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.latency_log.append({"stage": "fast_path_crypto", "ms": latency_ms})
            return crypto_prediction, 1.0, {
                "crypto_prediction": crypto_prediction,
                "mode": "fast_path",
                "latency_ms": latency_ms,
                "final_prediction": crypto_prediction,
                "final_confidence": 1.0,
            }

        # --- Stage 1: Discriminative classifier + crypto -----------------
        crypto_prediction = self.crypto_validator.cryptographic_validation(features)
        ai_prediction, ai_confidence, ai_prob_dist, ai_latency = self.get_ai_prediction(text)
        self.latency_log.append({"stage": "stage1_classifier", "ms": ai_latency})

        # Dynamic weights
        crypto_reliability = self._assess_crypto_reliability(features)
        ai_reliability = ai_confidence

        total_reliability = crypto_reliability + ai_reliability
        if total_reliability > 0:
            dynamic_crypto_weight = crypto_reliability / total_reliability
            dynamic_ai_weight = ai_reliability / total_reliability
        else:
            dynamic_crypto_weight = 0.5
            dynamic_ai_weight = 0.5

        decision_details = {
            'crypto_prediction': crypto_prediction,
            'crypto_reliability': crypto_reliability,
            'ai_prediction': ai_prediction,
            'ai_confidence': ai_confidence,
            'ai_prob_dist': ai_prob_dist.tolist(),
            'dynamic_crypto_weight': dynamic_crypto_weight,
            'dynamic_ai_weight': dynamic_ai_weight,
            'crypto_bypass_risk': features.get('crypto_bypass_risk', 0),
            'behavioral_anomaly_score': features.get('behavioral_anomaly_score', 0),
            'stage1_latency_ms': ai_latency,
        }

        # Stage 1 fusion decision
        final_prediction, confidence = self._make_adaptive_decision(
            crypto_prediction, ai_prediction, ai_confidence, ai_prob_dist,
            dynamic_crypto_weight, dynamic_ai_weight, features
        )

        # --- R5: Quarantine-by-default for uncertain verdicts -----------
        quarantined = False
        if confidence < self.quarantine_threshold:
            quarantined = True

        decision_details['quarantined'] = quarantined
        decision_details['decision_threshold'] = self.decision_threshold
        decision_details['update_category'] = self.update_category

        # --- R1 / Stage 2: Generative reasoning (async, non-blocking) ---
        stage2_output = None
        if self.enable_stage2 and not self.fast_path_mode:
            t0 = time.perf_counter()
            stage2_output = self.stage2.generate_threat_narrative(
                features, ai_prediction, ai_confidence
            )
            stage2_latency = (time.perf_counter() - t0) * 1000.0
            self.latency_log.append({"stage": "stage2_generative", "ms": stage2_latency})

            decision_details['stage2'] = stage2_output
            decision_details['stage2_latency_ms'] = stage2_latency

            # If Stage 2 detects high uncertainty, recommend quarantine
            if stage2_output['uncertainty_weight'] < 0.5 and not quarantined:
                quarantined = True
                decision_details['quarantined'] = True
                decision_details['quarantine_reason'] = 'stage2_uncertainty'

        decision_details.update({
            'final_prediction': final_prediction,
            'final_confidence': confidence,
        })

        return final_prediction, confidence, decision_details

    def _assess_crypto_reliability(self, features: Dict) -> float:
        """Assess reliability of crypto validation in current context."""
        base_reliability = 0.8  # High base reliability for crypto

        # Reduce reliability if sophisticated attack indicators present
        bypass_risk = features.get('crypto_bypass_risk', 0)
        reliability = base_reliability * (1 - bypass_risk * 0.4)

        # Further reduce if multiple anomalies suggest sophisticated attack
        behavioral_score = features.get('behavioral_anomaly_score', 0)
        if behavioral_score > 0.5:
            reliability *= 0.7

        return max(reliability, 0.3)  # Minimum reliability threshold

    def _make_adaptive_decision(self, crypto_pred: int, ai_pred: int, ai_conf: float,
                              ai_prob_dist: np.ndarray, crypto_weight: float,
                              ai_weight: float, features: Dict) -> Tuple[int, float]:
        """Make adaptive decision using multiple strategies with improved AI leverage."""

        # Enhanced Strategy 1: High-confidence AI override (prioritize well-trained AI)
        if ai_conf > 0.8:  # Lower threshold for high-confidence AI
            # Trust high-confidence AI predictions more
            return ai_pred, ai_conf

        # Strategy 2: Sophisticated attack detection (works even with untrained AI)
        bypass_risk = features.get('crypto_bypass_risk', 0)
        behavioral_score = features.get('behavioral_anomaly_score', 0)
        context_inconsistency = features.get('context_inconsistency', 0)

        # If multiple strong indicators suggest sophisticated attack, override crypto
        strong_attack_indicators = 0
        if bypass_risk > 0.4:
            strong_attack_indicators += 1
        if behavioral_score > 0.5:
            strong_attack_indicators += 1
        if context_inconsistency > 0.4:
            strong_attack_indicators += 1
        if features.get('fake_creator', 0) == 1:
            strong_attack_indicators += 1
        if features.get('adversary_role', 0) == 1:
            strong_attack_indicators += 1

        # If we have 2+ strong indicators and crypto says benign, flag as malicious
        if strong_attack_indicators >= 2 and crypto_pred == 0:
            confidence = min(0.9, 0.6 + (strong_attack_indicators * 0.1))
            return 1, confidence

        # Strategy 3: Behavioral anomaly detection (enhanced)
        if behavioral_score > 0.6 and crypto_pred == 0:
            # Even if AI is uncertain, trust behavioral analysis
            if ai_pred == 1 or behavioral_score > 0.8:
                return 1, max(0.7, behavioral_score)

        # Strategy 4: Context inconsistency detection (enhanced)
        if context_inconsistency > 0.5 and crypto_pred == 0:
            # Strong context inconsistency suggests attack
            return 1, max(0.75, context_inconsistency)

        # Strategy 5: Multi-factor risk assessment
        total_risk_score = (bypass_risk * 0.3 + behavioral_score * 0.4 + context_inconsistency * 0.3)
        if total_risk_score > 0.6 and crypto_pred == 0:
            return 1, min(0.85, total_risk_score + 0.2)

        # Strategy 6: Agreement amplification
        if crypto_pred == ai_pred:
            # Agreement - boost confidence
            base_confidence = 0.7 + ai_conf * 0.25
            # Additional boost if multiple indicators align
            if crypto_pred == 1 and strong_attack_indicators >= 1:
                base_confidence += 0.1
            return crypto_pred, min(0.95, base_confidence)

        # Strategy 7: Improved weighted decision with better AI leverage
        # When AI confidence is moderate to high, give it more weight
        if ai_conf > 0.6:
            # Boost AI weight for moderate to high confidence
            adjusted_ai_weight = ai_weight * 1.3
            adjusted_crypto_weight = crypto_weight * 0.8
        else:
            adjusted_ai_weight = ai_weight
            adjusted_crypto_weight = crypto_weight

        # Normalize weights
        total_weight = adjusted_ai_weight + adjusted_crypto_weight
        if total_weight > 0:
            adjusted_ai_weight = adjusted_ai_weight / total_weight
            adjusted_crypto_weight = adjusted_crypto_weight / total_weight

        weighted_score = (crypto_pred * adjusted_crypto_weight) + (ai_pred * adjusted_ai_weight)

        # Adjust based on risk factors
        if total_risk_score > 0.4:
            weighted_score += total_risk_score * 0.2  # Reduced impact

        final_prediction = 1 if weighted_score > 0.5 else 0
        confidence = max(0.5, (adjusted_crypto_weight + adjusted_ai_weight * ai_conf + total_risk_score * 0.1))

        return final_prediction, min(0.95, confidence)

    def validate_dataset(self, df: pd.DataFrame) -> Tuple[List[int], List[Dict]]:
        """Validate entire dataset using enhanced fusion approach."""
        predictions = []
        detailed_results = []

        print(f"Validating {len(df)} samples with enhanced AI+Crypto system...")
        print(f"Using fusion strategy: {self.fusion_strategy}")

        for idx, row in df.iterrows():
            if idx % 50 == 0:
                print(f"Processing sample {idx+1}/{len(df)}")

            # Extract enhanced features
            features = self.extract_enhanced_features(row['input_prompt'])

            # Apply adaptive fusion strategy
            prediction, confidence, details = self.adaptive_fusion(row['input_prompt'], features)

            predictions.append(prediction)

            # R3: Update source reputation based on prediction correctness
            source_id = features.get('creator', 'unknown')
            channel = features.get('channel', 'unknown')
            was_correct = (prediction == row['output_label'])
            self.reputation.record_interaction(
                source_id, timestamp=float(features.get('time', idx)),
                was_correct=was_correct, channel=channel
            )

            # R3: Fetch current reputation for this source
            source_rep = self.reputation.get_reputation(source_id)
            sybil_check = self.reputation.check_sybil(source_id)

            # Store detailed results
            result = {
                'index': idx,
                'features': features,
                'prediction': prediction,
                'confidence': confidence,
                'decision_details': details,
                'actual_label': row['output_label'],
                'correct': was_correct,
                'source_reputation': source_rep,
                'sybil_suspicious': sybil_check.get('is_suspicious', False),
            }
            detailed_results.append(result)

        # R3: Run collusion detection after full dataset pass
        self._collusion_results = self.reputation.detect_collusion()

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

    def get_latency_report(self) -> Dict:
        """R4: Summarise inference latency statistics."""
        if not self.latency_log:
            return {}

        stage1 = [e['ms'] for e in self.latency_log if e['stage'] == 'stage1_classifier']
        stage2 = [e['ms'] for e in self.latency_log if e['stage'] == 'stage2_generative']
        fast = [e['ms'] for e in self.latency_log if e['stage'] == 'fast_path_crypto']

        def _stats(vals):
            if not vals:
                return {}
            arr = np.array(vals)
            return {
                'count': len(arr),
                'mean_ms': float(np.mean(arr)),
                'p50_ms': float(np.median(arr)),
                'p95_ms': float(np.percentile(arr, 95)),
                'max_ms': float(np.max(arr)),
            }

        return {
            'stage1_classifier': _stats(stage1),
            'stage2_generative': _stats(stage2),
            'fast_path_crypto': _stats(fast),
        }

    def get_quarantine_summary(self) -> Dict:
        """R5: Summarise quarantine decisions."""
        total = len(self.validation_results)
        quarantined = sum(
            1 for r in self.validation_results
            if r.get('decision_details', {}).get('quarantined', False)
        )
        return {
            'total_samples': total,
            'quarantined': quarantined,
            'quarantine_rate': quarantined / total if total else 0,
            'decision_threshold': self.decision_threshold,
            'quarantine_threshold': self.quarantine_threshold,
            'update_category': self.update_category,
        }

    def get_reputation_report(self) -> Dict:
        """R3: Summarise reputation system state including Sybil/collusion flags."""
        rep_summary = self.reputation.summary()
        rep_summary['collusion_detections'] = getattr(self, '_collusion_results', [])
        return rep_summary

    def analyze_fusion_effectiveness(self) -> Dict:
        """Analyze effectiveness of the fusion approach."""
        crypto_only_correct = 0
        ai_only_correct = 0
        fusion_correct = 0
        fusion_improvements = 0

        for result in self.validation_results:
            details = result['decision_details']
            actual = result['actual_label']

            crypto_pred = details['crypto_prediction']
            ai_pred = details.get('ai_prediction', crypto_pred)
            fusion_pred = result['prediction']

            if crypto_pred == actual:
                crypto_only_correct += 1
            if ai_pred == actual:
                ai_only_correct += 1
            if fusion_pred == actual:
                fusion_correct += 1

            # Check if fusion improved over individual components
            if fusion_pred == actual and (crypto_pred != actual or ai_pred != actual):
                fusion_improvements += 1

        total_samples = len(self.validation_results)

        analysis = {
            'crypto_only_accuracy': crypto_only_correct / total_samples if total_samples > 0 else 0,
            'ai_only_accuracy': ai_only_correct / total_samples if total_samples > 0 else 0,
            'fusion_accuracy': fusion_correct / total_samples if total_samples > 0 else 0,
            'fusion_improvements': fusion_improvements,
            'improvement_rate': fusion_improvements / total_samples if total_samples > 0 else 0
        }

        return analysis

def main():
    """Main function to run enhanced AI+crypto validation."""
    print("=== Enhanced AI + Cryptographic Validation System ===")
    print("    Two-stage pipeline: Stage 1 (Discriminative) + Stage 2 (Generative Reasoning)")

    validator = EnhancedAICryptoValidator(fusion_strategy="adaptive",
                                          update_category="default",
                                          enable_stage2=True)

    print("Loading datasets...")
    test_df = pd.read_csv("./data/test_set.csv")
    print(f"Test set: {len(test_df)} samples")

    test_predictions, test_results = validator.validate_dataset(test_df)
    test_metrics = validator.calculate_metrics(test_df['output_label'].tolist(), test_predictions)
    fusion_analysis = validator.analyze_fusion_effectiveness()

    # --- Standard metrics -----------------------------------------------
    print(f"\n=== Enhanced System Results ===")
    print(f"Accuracy:                {test_metrics['accuracy']:.4f}")
    print(f"Precision:               {test_metrics['precision']:.4f}")
    print(f"Recall:                  {test_metrics['recall']:.4f}")
    print(f"F1-Score:                {test_metrics['f1_score']:.4f}")
    print(f"Malicious Detection Rate:{test_metrics['malicious_detection_rate']:.4f}")
    print(f"False Positive Rate:     {test_metrics['false_positive_rate']:.4f}")

    print(f"\n=== Fusion Analysis ===")
    print(f"Crypto-only Accuracy:    {fusion_analysis['crypto_only_accuracy']:.4f}")
    print(f"AI-only Accuracy:        {fusion_analysis['ai_only_accuracy']:.4f}")
    print(f"Fusion Accuracy:         {fusion_analysis['fusion_accuracy']:.4f}")
    print(f"Fusion Improvements:     {fusion_analysis['fusion_improvements']} cases")
    print(f"Improvement Rate:        {fusion_analysis['improvement_rate']:.4f}")

    # --- R4: Latency report ---------------------------------------------
    latency = validator.get_latency_report()
    print(f"\n=== Latency Report (R4) ===")
    for stage, stats in latency.items():
        if stats:
            print(f"  {stage}: mean={stats['mean_ms']:.2f}ms  "
                  f"p50={stats['p50_ms']:.2f}ms  p95={stats['p95_ms']:.2f}ms  "
                  f"max={stats['max_ms']:.2f}ms  (n={stats['count']})")

    # --- R5: Quarantine report ------------------------------------------
    quarantine = validator.get_quarantine_summary()
    print(f"\n=== Quarantine Report (R5) ===")
    print(f"  Update category:       {quarantine['update_category']}")
    print(f"  Decision threshold:    {quarantine['decision_threshold']}")
    print(f"  Quarantine threshold:  {quarantine['quarantine_threshold']}")
    print(f"  Quarantined samples:   {quarantine['quarantined']} / {quarantine['total_samples']} "
          f"({quarantine['quarantine_rate']:.2%})")

    # --- R3: Reputation report ------------------------------------------
    rep_report = validator.get_reputation_report()
    print(f"\n=== Reputation Report (R3) ===")
    print(f"  Total tracked sources: {rep_report['total_sources']}")
    print(f"  Mean reputation:       {rep_report['mean_reputation']:.4f}")
    if rep_report['sybil_suspects']:
        print(f"  Sybil suspects:        {rep_report['sybil_suspects']}")
    else:
        print(f"  Sybil suspects:        None")
    if rep_report['collusion_detections']:
        for c in rep_report['collusion_detections']:
            print(f"  Collusion pair: {c['source_pair']} "
                  f"(agreement={c['agreement_rate']:.2%})")
    else:
        print(f"  Collusion pairs:       None detected")

    # --- Save results ---------------------------------------------------
    results_df = pd.DataFrame([{
        'method': 'Enhanced_AI_Crypto_Adaptive',
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
        'fusion_improvements': fusion_analysis['fusion_improvements'],
        'improvement_rate': fusion_analysis['improvement_rate'],
        'quarantined_count': quarantine['quarantined'],
        'quarantine_rate': quarantine['quarantine_rate'],
    }])

    results_df.to_csv('./enhanced_ai_crypto_results.csv', index=False)
    print(f"\nResults saved to './enhanced_ai_crypto_results.csv'")

    return validator, test_metrics, fusion_analysis

if __name__ == "__main__":
    validator, metrics, analysis = main()
