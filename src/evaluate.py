import argparse
import logging

from pathlib import Path

import joblib
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score
)

sys.path.insert(0, str(Path(__file__).parent))

from data import MedicalDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_true, y_pred, output_path):
    """Confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Non-Medical', 'Medical'],
        yticklabels=['Non-Medical', 'Medical']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Confusion matrix saved to {output_path}")
    plt.close()


def plot_roc_curve(y_true, y_proba, output_path):
    """ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"ROC curve saved to {output_path}")
    plt.close()


def plot_precision_recall_curve(y_true, y_proba, output_path):
    """Precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Precision-recall curve saved to {output_path}")
    plt.close()

def plot_calibration_curve(y_true, y_proba, output_path, n_bins=10):
    """
    Plot calibration curve to check if predicted probabilities match reality.
    """
    # Bin predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute fraction of positives in each bin
    bin_sums = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)

    # Avoid division by zero
    bin_counts = np.maximum(bin_counts, 1)
    bin_true = bin_sums / bin_counts

    # Mean predicted probability in each bin
    bin_pred = np.array([
        y_proba[bin_indices == i].mean() if (bin_indices == i).sum() > 0 else 0
        for i in range(n_bins)
    ])

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(bin_pred, bin_true, 'o-', linewidth=2, label='Model')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Calibration curve saved to {output_path}")
    plt.close()

def analyze_errors(X_test, y_true, y_pred, y_proba, output_path):
    """Analyze misclassified samples."""
    errors = []

    for i, (text, true_label, pred_label, prob) in enumerate(zip(X_test, y_true, y_pred, y_proba)):
        if true_label != pred_label:
            errors.append({
                'text': text,
                'true_label': 'Medical' if true_label == 1 else 'Non-Medical',
                'pred_label': 'Medical' if pred_label == 1 else 'Non-Medical',
                'confidence': prob,
                'error_type': 'False Positive' if pred_label == 1 else 'False Negative'
            })

    error_df = pd.DataFrame(errors)
    error_df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Error analysis saved to {output_path}")

    # Summary
    logger.info(f"\nTotal errors: {len(errors)}")
    if len(errors) > 0:
        logger.info(f"  False Positives: {(error_df['error_type'] == 'False Positive').sum()}")
        logger.info(f"  False Negatives: {(error_df['error_type'] == 'False Negative').sum()}")

        logger.info("\nExample errors (first 3):")
        for idx, row in error_df.head(3).iterrows():
            logger.info(f"\n  Text: {row['text'][:100]}...")
            logger.info(f"  True: {row['true_label']} | Predicted: {row['pred_label']} | Confidence: {row['confidence']:.3f}")



def main(args):
    """Main EVAL pipeline."""

    # Create output directories
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MEDICAL TEXT CLASSIFIER - EVALUATION")
    logger.info("=" * 60)

    # 1. Load test data
    logger.info("\n[1/4] Loading test data...")

    if args.dataset_name:
        logger.info(f"Loading from HuggingFace: {args.dataset_name}")
        data_loader = MedicalDataLoader(
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            text_column=args.text_column,
            label_column=args.label_column,
            test_size=args.test_size,
            random_state=args.random_state
        )
    else:
        logger.info(f"Loading from CSV: {args.data}")
        data_loader = MedicalDataLoader(
            data_path=args.data,
            test_size=args.test_size,
            random_state=args.random_state
        )

    X_train, X_test, y_train, y_test, _ = data_loader.prepare_dataset()

    # 2. Load trained model and feature extractor
    logger.info("\n[2/4] Loading trained models...")
    model_dir = Path(args.model_dir)

    feature_extractor = joblib.load(model_dir / "feature_extractor.joblib")
    logger.info("Feature extractor loaded")

    # Load classifier
    from models import MedicalTextClassifier
    classifier = MedicalTextClassifier()
    classifier.load(str(model_dir))
    logger.info("Classifier loaded")

    # 3. Extract features and predict
    logger.info("\n[3/4] Running predictions...")
    X_test_features = feature_extractor.transform(X_test.tolist())
    y_proba = classifier.predict_proba(X_test_features)[:, 1]
    y_pred = (y_proba >= args.threshold).astype(int)

    # 4. Compute metrics
    logger.info("\n[4/4] Computing evaluation metrics...")

    # Classification report
    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION REPORT")
    logger.info("=" * 60)
    print(classification_report(
        y_test, y_pred,
        target_names=['Non-Medical', 'Medical'],
        digits=4
    ))

    # Key metrics
    f1_medical = f1_score(y_test, y_pred, pos_label=1)
    precision_medical = precision_score(y_test, y_pred, pos_label=1)
    recall_medical = recall_score(y_test, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_proba)

    logger.info("\n" + "=" * 60)
    logger.info("KEY METRICS (Medical Class)")
    logger.info("=" * 60)
    logger.info(f"F1 Score:       {f1_medical:.4f}")
    logger.info(f"Precision:      {precision_medical:.4f}")
    logger.info(f"Recall:         {recall_medical:.4f}")
    logger.info(f"ROC-AUC:        {roc_auc:.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("METRICS INTERPRETATION")
    logger.info("=" * 60)
    logger.info("F1 Score (Medical):")
    logger.info("  - Harmonic mean of precision and recall")
    logger.info("  - Best for imbalanced datasets (ignores true negatives)")
    logger.info("  - Target: >0.80 for production readiness")
    logger.info(f"  - Current: {f1_medical:.4f} {'✓ PASS' if f1_medical >= 0.80 else '✗ NEEDS IMPROVEMENT'}")

    logger.info("\nROC-AUC:")
    logger.info("  - Measures ranking quality across all thresholds")
    logger.info("  - Threshold-agnostic, handles class imbalance")
    logger.info("  - Target: >0.90 for reliable classification")
    logger.info(f"  - Current: {roc_auc:.4f} {'✓ PASS' if roc_auc >= 0.90 else '✗ NEEDS IMPROVEMENT'}")

    logger.info("\nPrecision (Medical):")
    logger.info("  - Of predicted medical texts, how many are actually medical?")
    logger.info("  - High precision = low false positive rate")
    logger.info(f"  - Current: {precision_medical:.4f} ({precision_medical*100:.1f}% of predictions are correct)")

    logger.info("\nRecall (Medical):")
    logger.info("  - Of all actual medical texts, how many did we catch?")
    logger.info("  - High recall = low false negative rate")
    logger.info(f"  - Current: {recall_medical:.4f} ({recall_medical*100:.1f}% of medical texts detected)")

    # 5. Generate visualizations
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 60)

    plot_confusion_matrix(y_test, y_pred, log_dir / "confusion_matrix.png")
    plot_roc_curve(y_test, y_proba, log_dir / "roc_curve.png")
    plot_precision_recall_curve(y_test, y_proba, log_dir / "precision_recall_curve.png")
    plot_calibration_curve(y_test, y_proba, log_dir / "calibration_curve.png")

    # 6. Error analysis
    logger.info("\n" + "=" * 60)
    logger.info("ERROR ANALYSIS")
    logger.info("=" * 60)
    analyze_errors(X_test, y_test, y_pred, y_proba, log_dir / "error_analysis.csv")

    # 7. Save evaluation summary
    summary = {
        'test_samples': len(X_test),
        'medical_samples': int(y_test.sum()),
        'non_medical_samples': int(len(y_test) - y_test.sum()),
        'f1_score_medical': float(f1_medical),
        'precision_medical': float(precision_medical),
        'recall_medical': float(recall_medical),
        'roc_auc': float(roc_auc),
        'threshold': args.threshold
    }

    import json
    with open(log_dir / "evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nEvaluation summary saved to {log_dir / 'evaluation_summary.json'}")

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nResults saved to: {log_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate German Medical Text classifier"
    )

    # Data source options (mutually exclusive because it can either be CSV or HF, not both)
    data_group = parser.add_mutually_exclusive_group(required=False)
    data_group.add_argument(
        "--data",
        type=str,
        default="data/medical_corpus.csv",
        help="Path to test data CSV"
    )
    data_group.add_argument(
        "--dataset-name",
        type=str,
        help="HuggingFace dataset name"
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold"
    )

    args = parser.parse_args()
    main(args)