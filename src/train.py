import argparse
import logging
from pathlib import Path
import joblib
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data import MedicalDataLoader
from features import GermanMedicalFeatureExtractor
from models import MedicalTextClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """Main Training pipeline."""

    # output_dirs
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MEDICAL TEXT CLASSIFIER (GERMAN) - TRAINING PIPELINE")
    logger.info("=" * 60)

    # 1. Load and preprocess data
    logger.info("\n[1/4] Loading data...")

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

    X_train, X_test, y_train, y_test, class_weights = data_loader.prepare_dataset()

    # 2. Extract features
    logger.info("\n[2/4] Extracting features...")
    feature_extractor = GermanMedicalFeatureExtractor(max_features=args.max_features)
    X_train_features = feature_extractor.fit_transform(X_train.tolist(), y_train.values)
    X_test_features = feature_extractor.transform(X_test.tolist())

    # Save feature extractor
    feature_extractor_path = model_dir / "feature_extractor.joblib"
    joblib.dump(feature_extractor, feature_extractor_path)
    logger.info(f"Feature extractor saved to {feature_extractor_path}")

    # 3. Train classifier
    logger.info("\n[3/4] Training LightGBM classifier...")
    classifier = MedicalTextClassifier(
        class_weights=class_weights,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    classifier.train(
        X_train_features,
        y_train.values,
        X_test_features,
        y_test.values,
        verbose=50
    )

    # Save classifier
    classifier.save(str(model_dir))
    logger.info(f"Classifier saved to {model_dir}")

    # 4. Display feature importance
    logger.info("\n[4/4] Feature importance analysis...")
    feature_names = feature_extractor.get_feature_names()
    top_features = classifier.get_feature_importance(feature_names, top_k=20)

    logger.info("\nTop 20 most important features:")
    for i, (feat_name, importance) in enumerate(top_features.items(), 1):
        logger.info(f"  {i:2d}. {feat_name:40s} {importance:10.2f}")

    # Save feature importance
    importance_path = model_dir / "feature_importance.joblib"
    joblib.dump(top_features, importance_path)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nModel artifacts saved to: {model_dir}/")
    logger.info("  - lightgbm_model.txt")
    logger.info("  - model_metadata.joblib")
    logger.info("  - feature_extractor.joblib")
    logger.info("  - feature_importance.joblib")
    logger.info("\nNext steps:")
    logger.info("  1. Run evaluation: make evaluate")
    logger.info("  2. Test predictions: make predict TEXT='Your sentence here'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train German medical text classifier"
    )

    # Data source options (mutually exclusive because it can either be CSV or HF, not both)
    data_group = parser.add_mutually_exclusive_group(required=False)
    data_group.add_argument(
        "--data",
        type=str,
        default="data/medical_corpus.csv",
        help="Path to training data CSV"
    )
    data_group.add_argument(
        "--dataset-name",
        type=str,
        help="HuggingFace dataset name (as given in exercise)"
    )

    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="HuggingFace dataset split to load (default: train)"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of text column in HuggingFace dataset (default: text)"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="is_medical",
        help="Name of label column in HuggingFace dataset (default: is_medical)"
    )

    # model & training options
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum TF-IDF features"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of boosting rounds"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth"
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()
    main(args)