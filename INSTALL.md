# Installation Guide - Exact Working Setup

This guide provides the **exact installation steps** that are guaranteed to work.

---

## ✅ **Prerequisites**

- **Python 3.11, or 3.12** (Python 3.9 NOT supported due to numpy ABI issues)
- **uv** package manager
- **8GB RAM** minimum

---

## 🚀 **Step-by-Step Installation**

### **1. Verify Python Version**

```bash
python3 --version
# Should show: Python 3.11.x, or 3.12.x

# If you have Python 3.9 or older, upgrade first:
# macOS: brew install python@3.11
# Ubuntu: sudo apt install python3.11
```

---

### **2. Clone/Navigate to Project**

```bash
cd /path/to/ml-tech-task-sonia-sreenjoy
```

---

### **3. Remove Any Existing Environment**

```bash
rm -rf .venv
rm -rf __pycache__ src/__pycache__
```

---

### **4. Install Dependencies with uv**

```bash
# Install uv if not already installed
pip install uv

# Install all dependencies (this uses pyproject.toml)
uv sync

# This should install:
# - numpy 1.26.4
# - spacy 3.7.4 with exact matching dependencies
# - lightgbm 4.3.0
# - All other packages with locked versions
```

---

### **5. Download SpaCy German Model**

```bash
# Activate venv
source .venv/bin/activate

# UV pip install the German model
uv pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.7.0/de_core_news_lg-3.7.0-py3-none-any.whl

# Verify installation
python -c "import spacy; nlp = spacy.load('de_core_news_lg'); print('✓ German model loaded!')"
```

---

### **6. Verify Complete Installation**

```bash
# Test all imports (VERY IMPORTANT TO RUN BEFORE TRAINING OR EVALUATION)
python -c "
import numpy
import pandas
import spacy
import lightgbm
import sklearn
from datasets import load_dataset

print(f'✓ numpy: {numpy.__version__}')
print(f'✓ pandas: {pandas.__version__}')
print(f'✓ spacy: {spacy.__version__}')
print(f'✓ lightgbm: {lightgbm.__version__}')
print(f'✓ sklearn: {sklearn.__version__}')
print('✓ All dependencies working!')
"
```

Expected output:
```
✓ numpy: 1.26.4
✓ pandas: 2.2.1
✓ spacy: 3.7.4
✓ lightgbm: 4.3.0
✓ sklearn: 1.4.2
✓ All dependencies working!
```

---

### **7. Test Training**

```bash
# From HuggingFace dataset
make train-hf DATASET="SoniaSolutions/NLP-Tech-Task"
```

---

## ✅ **Quick Verification Checklist**

- [ ] Python 3.11+ installed
- [ ] uv installed (`pip install uv`)
- [ ] Dependencies installed (`uv sync`)
- [ ] SpaCy model downloaded (direct wheel URL)
- [ ] Imports work (run verification script)
- [ ] Training works (`make train-hf`)

---
