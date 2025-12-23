.PHONY: install setup train train-hf evaluate predict clean help

PYTHON := python3
VENV := .venv
DATA_DIR := data
MODEL_DIR := models
TRAIN_DATA := $(DATA_DIR)/medical_corpus.csv

# HF Dataset Options
DATASET ?=
SPLIT ?= train
TEXT_COL ?= text
LABEL_COL ?= is_medical

help:
	@echo "Medical Classifier MVP - Makefile"
	@echo "==========================================="
	@echo "make install       - Install all dependencies using uv"
	@echo "make setup         - Download SpaCy German model"
	@echo "make train         - Train the LightGBM classifier (local CSV)"
	@echo "make train-hf      - Train from HuggingFace dataset (DEFAULT)"
	@echo "make evaluate      - Evaluate model performance"
	@echo "make predict       - Run inference on new samples"
	@echo "make clean         - Remove generated files"

install:
	@echo "Installing dependencies with uv..."
	uv sync
	@echo "✓ Dependencies installed"

train-hf: setup
	@echo "Training LightGBM classifier from HuggingFace dataset..."
	@if [ -z "$(DATASET)" ]; then \
		echo "Error: Please specify DATASET variable"; \
		echo "Example: make train-hf DATASET='username/medical-corpus'"; \
		exit 1; \
	fi
	@echo "Loading dataset: $(DATASET)"
	uv run python src/train.py \
		--dataset-name "$(DATASET)" \
		--dataset-split "$(SPLIT)" \
		--text-column "$(TEXT_COL)" \
		--label-column "$(LABEL_COL)"
	@echo "✓ Training complete. Models saved to $(MODEL_DIR)/"

evaluate:
	@echo "Evaluating model performance..."
	uv run python src/evaluate.py --data $(TRAIN_DATA)
	@echo "✓ Evaluation complete. Report saved to logs/"

predict:
	@echo "Running inference..."
	@echo "Usage: make predict TEXT='Your German sentence here'"
	@if [ -z "$(TEXT)" ]; then \
		uv run python cli.py predict --input data/test_samples.txt; \
	else \
		uv run python cli.py predict --text "$(TEXT)"; \
	fi

clean:
	@echo "Cleaning generated files..."
	rm -rf $(MODEL_DIR)/*.pkl $(MODEL_DIR)/*.joblib
	rm -rf logs/*.png logs/*.txt
	rm -rf __pycache__ src/__pycache__
	rm -rf .pytest_cache
	@echo "✓ Cleaned"