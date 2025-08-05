# Detection of Sarcastic News Headlines

A machine learning project focused on the detection of sarcasm in news headlines using advanced natural language processing (NLP) techniques. This repository contains our code, report, and presentation for the comparative study of **BiLSTM**, **BERT**, and **RoBERTa** architectures.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models and Methodology](#models-and-methodology)
- [Results](#results)
- [Getting Started](#getting-started)
- [How to Run](#how-to-run)


---

## Project Overview

Detecting sarcasm in news headlines is a challenging task due to the subtle and complex nature of sarcastic language. With the emergence of transformer-based architectures, there is significant potential to improve sarcasm detection systems.  
This project compares the performance of three models:
- **Bidirectional LSTM (BiLSTM)**
- **BERT (Bidirectional Encoder Representations from Transformers)**
- **RoBERTa (Robustly optimized BERT approach)**

Our goal is to benchmark these models on a curated sarcasm detection dataset and evaluate their effectiveness in capturing the nuanced cues of sarcastic discourse.

---

## Dataset

A curated dataset of news headlines labeled for sarcasm was used.  
*Details about the dataset source, size, and preprocessing steps can be added here if you wish to make the repo more complete and reusable.*

---

## Models and Methodology

### 1. Bidirectional LSTM (Baseline)
- **Architecture**: Embedding layer → 2-layer BiLSTM (hidden size: 256, dropout: 0.5) → Linear layer → Sigmoid activation.
- **Tokenizer**: `basic_english`
- **Loss Function**: Binary Cross Entropy
- **Optimizer**: Adam (lr=0.001)
- **Training**: 15 epochs

### 2. BERT
- **Pre-trained Model**: `bert-base-uncased` from HuggingFace Transformers
- **Architecture**: BERT → Dropout (0.3) → Linear (768, 1)
- **Loss Function**: Binary Cross Entropy
- **Optimizer**: Adam (lr=1e-5)
- **Training**: 15 epochs

### 3. RoBERTa
- **Pre-trained Model**: `roberta-base` from HuggingFace Transformers
- **Architecture**: RoBERTa → Dropout (0.3) → Linear (768, 1)
- **Loss Function**: Binary Cross Entropy
- **Optimizer**: Adam (lr=1e-5)
- **Training**: 15 epochs

Each model was trained and validated using a 70:30 split.

---

## Results

**Summary Table (Validation Data):**

| Metric         | BiLSTM | BERT   | RoBERTa |
|----------------|--------|--------|---------|
| Precision      | 0.82–0.84 | 0.92–0.93 | 0.97–0.98 |
| Recall         | 0.80–0.86 | 0.91–0.94 | 0.96–0.99 |
| F1-Score       | 0.82–0.84 | 0.92–0.93 | 0.97–0.98 |
| ROC-AUC        | 0.91   | 0.98   | 1.00    |
| **Accuracy**   | 0.83   | 0.92   | 0.97    |

- **RoBERTa** achieved the highest accuracy and ROC-AUC, slightly outperforming BERT.
- **BERT** also performed very well, demonstrating the power of transformer models for this task.
- **BiLSTM**, while a solid baseline, lagged behind the transformer-based models.

**Test Scores (Kaggle):**
- RoBERTa: 0.97099
- BERT: 0.97012
- BiLSTM: 0.94409

---

### Prerequisites
- Python 3.8+
- PyTorch
- HuggingFace Transformers
- torchtext
- numpy, pandas, scikit-learn, matplotlib

### Installation
```bash
git clone https://github.com/<your-username>/Detection-of-Sarcastic-News-Headlines.git
cd Detection-of-Sarcastic-News-Headlines
pip install -r requirements.txt
```
## How to Run

1. **Data Preparation:**  
   Place your dataset in the `/data` directory (update the code/data path as needed).

2. **Training:**  
   To train a specific model, run the following commands in your terminal:

   ```bash
   python train.py --model bilstm    # For BiLSTM
   python train.py --model bert      # For BERT
   python train.py --model roberta   # For RoBERTa 


