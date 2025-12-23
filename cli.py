import click
import joblib
import sys
import time
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalClassifierPredictor:
    """Wrapper for loading models and running inference."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.feature_extractor = None
        self.classifier = None
        self._load_models()

    def _load_models(self):
        """Load all trained models."""
        logger.info(f"Loading models from {self.model_dir}...")

        try:
            # load feature extractor
            self.feature_extractor = joblib.load(
                self.model_dir / "feature_extractor.joblib"
            )

            # load classifier
            from models import MedicalTextClassifier
            self.classifier = MedicalTextClassifier()
            self.classifier.load(str(self.model_dir))

            logger.info("✓ Models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            logger.error("Please run training first: make train")
            sys.exit(1)

    def predict_single(
            self,
            text: str,
            threshold: float = 0.5
        ) -> dict:
            """
            Predict label for a single text sample.

            Args:
                text: Input text
                threshold: Classification threshold

            Returns:
                Dictionary with prediction results
            """
            start_time = time.time()

            # Extract features
            X = self.feature_extractor.transform([text])

            # Predict
            proba = self.classifier.predict_proba(X)[0]
            pred_label = 1 if proba[1] >= threshold else 0

            inference_time = time.time() - start_time

            result = {
                'text': text,
                'label': 'Medical' if pred_label == 1 else 'Non-Medical',
                'confidence': float(proba[1]),
                'inference_time_ms': inference_time * 1000
            }

            return result

#TODO : Add method for predict_batch (currently only single text CLI supported)

@click.group()
def cli():
    """German Medical Text Classifier CLI."""
    pass


@cli.command()
@click.option('--text', type=str, help='Text to classify')
@click.option('--input', type=click.Path(exists=True), help='File with texts (one per line)')
@click.option('--threshold', type=float, default=0.5, help='Classification threshold')
@click.option('--model-dir', type=str, default='models', help='Model directory')
def predict(text, input, output, threshold, model_dir):
    """
    Predict medical/non-medical label for text.

    Examples:
        Single prediction:
            python cli.py predict --text "Der Patient hat Fieber."
    """
    predictor = MedicalClassifierPredictor(model_dir=model_dir)

    # single prediction
    if text:
        result = predictor.predict_single(text, threshold=threshold)

        click.echo("\n" + "=" * 60)
        click.echo("PREDICTION RESULT")
        click.echo("=" * 60)
        click.echo(f"Text:       {result['text']}")
        click.echo(f"Label:      {result['label']}")
        click.echo(f"Confidence: {result['confidence']:.3f}")
        click.echo(f"Time:       {result['inference_time_ms']:.2f}ms")

    else:
        click.echo("Error: Provide either --text or --input")
        sys.exit(1)



@cli.command()
@click.option('--model-dir', type=str, default='models', help='Model directory')
def info(model_dir):
    """Display model information and feature importance."""
    model_dir = Path(model_dir)

    try:
        # Load metadata
        import json
        from models import MedicalTextClassifier

        classifier = MedicalTextClassifier()
        classifier.load(str(model_dir))

        feature_importance = joblib.load(model_dir / "feature_importance.joblib")

        click.echo("\n" + "=" * 60)
        click.echo("MODEL INFORMATION")
        click.echo("=" * 60)
        click.echo(f"Model directory: {model_dir}")
        click.echo(f"Class weights:   {classifier.class_weights}")
        click.echo(f"Features:        {classifier.model.num_feature()}")

        click.echo("\nTop 10 Most Important Features:")
        for i, (feat_name, importance) in enumerate(list(feature_importance.items())[:10], 1):
            click.echo(f"  {i:2d}. {feat_name:40s} {importance:10.2f}")

    except Exception as e:
        click.echo(f"Error loading model info: {e}")
        sys.exit(1)


def main():
    """Entry point for CLI."""
    cli()


if __name__ == '__main__':
    main()