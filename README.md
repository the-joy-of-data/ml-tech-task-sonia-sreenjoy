# ML Tech Task Sonia Sreenjoy

## German Medical Text Classifier MVP

Binary classifier for detecting medical content in German text using LightGBM with engineered features.

---

### **Architecture**

```
Input Text
    ↓
Feature Extraction (5000+ features)
    ├─ TF-IDF (char n-grams 2-5 for German compounds)
    ├─ Medical vocabulary overlap score
    ├─ POS tag distributions (noun/verb/adj ratios)
    └─ Statistical features (length, lexical diversity)
    ↓
LightGBM Classifier (300 estimators)
    ├─ Handles class imbalance via scale_pos_weight
    ├─ Early stopping on validation set
    └─ Returns calibrated probabilities
    ↓
Output: {label, confidence, inference_time}
```

---

## 📦 **Installation**

### **Prerequisites**
- Python 3.11+
- `uv` package manager (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- 8GB RAM minimum (for SpaCy word vectors)

### **Setup**

```bash
# Clone repository
git clone <repo-url>
cd ml-tech-task-sonia-sreenjoy

# Install dependencies and SpaCy model
make install

# Activate venv
source .venv/bin/activate

# UV pip install the German model
uv pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.7.0/de_core_news_lg-3.7.0-py3-none-any.whl

# Verify installation
uv run python -c "import spacy; nlp = spacy.load('de_core_news_lg'); print('✓ German model loaded!')"
```

---

## 🚀 **Usage**

### **1. Prepare Training Data**

The training data is provided as a HuggingFace dataset:

#### **Source : HuggingFace Dataset**

Load directly from HuggingFace Hub:

```bash
# Train from public HuggingFace dataset
uv run python src/train.py \
    --dataset-name "username/german-medical-corpus" \
    --dataset-split "train"

# With custom column names
uv run python src/train.py \
    --dataset-name "username/medical-data" \
    --text-column "sentence" \
    --label-column "label"
```

**Requirements for HuggingFace datasets**:
- Dataset must have a text column (default: `text`)
- Dataset must have a binary label column (default: `is_medical`)
- Labels can be: bool, int (0/1), or strings ("medical"/"non-medical")

**Authentication** (for private datasets):
```bash
# Login to HuggingFace
huggingface-cli login

# Then train normally
uv run python src/train.py --dataset-name "SoniaSolutions/NLP-Tech-Task"
```

### **2. Train Model**

```bash
# HF shared dataset (requires authentication)
huggingface-cli login
make train-hf DATASET="SoniaSolutions/NLP-Tech-Task"

**Output**:
- `models/lightgbm_model.txt` - Trained classifier
- `models/feature_extractor.joblib` - TF-IDF + feature pipeline
- `models/feature_importance.joblib` - Feature rankings
```

**Training time**: ~2-3 minutes on Apple Macbook M1 Pro for 1k samples

---

### **3. Run Inference**

```bash
# Basic prediction
make predict TEXT="Der Patient hat nach der Operation eine Phlebothrombose im Bereich des Unterschenkels entwickelt."
```