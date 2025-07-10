# FineTune_C-_CodeT5
# 🛡️ AI-Powered C++ Vulnerability Analyzer

This project is a two-stage AI-based tool for identifying and explaining vulnerabilities in C++ code. It leverages a classification model (`Roberta`) to detect unsafe code and a sequence-to-sequence model (`CodeT5`) to generate natural language explanations.

## 🚀 Features

- ✅ **C++ Vulnerability Classification** using fine-tuned `Roberta`
- 🧠 **Explanation Generation** using fine-tuned `Salesforce/CodeT5`
- 📊 **Multi-chunk Inference** for long functions with overlapping windows
- 🖥️ **Streamlit Web App** for interactive analysis
- 💾 **Checkpoint-Based Training** with auto-saving every 15% progress
- 🔄 **Fine-tuned on Juliet Test Suite dataset**

## 📁 Project Structure

├── app.py # Streamlit frontend
├── Trainer.py # Model training script (CodeT5)
├── Tokenizer.py # Dataset loading and tokenization
├── parsed_juliet_codet5_blocks/ # Input JSONL files (source + explanation)
├── tokenized_juliet_dataset/ # Tokenized dataset for training
├── cpp_linter_model/ # Roberta classification model
├── codet5_explainer_model/ # Trained CodeT5 explanation model
├── codet5_explainer_checkpoints/# Intermediate CodeT5 checkpoints
└── README.md


---

## 🧠 Models

### 🔹 Vulnerability Classifier
- Model: `RobertaForSequenceClassification`
- Input: Raw C++ function
- Output: Binary label (Safe / Vulnerable)

### 🔹 Explanation Generator
- Model: `Salesforce/codet5-base` (fine-tuned)
- Input: Detected vulnerable code
- Output: Natural language explanation of the vulnerability

---

## 🧪 Dataset

- ✅ Based on the **Juliet Test Suite v1.3** for C/C++
- ✅ Preprocessed into:
  - `source`: C++ code block
  - `explanation`: Natural language description

## 📦 Dependencies

```bash
pip install torch transformers datasets streamlit

## 🧠 Models

### 🔹 Vulnerability Classifier
- Model: `RobertaForSequenceClassification`
- Input: Raw C++ function
- Output: Binary label (Safe / Vulnerable)

### 🔹 Explanation Generator
- Model: `Salesforce/codet5-base` (fine-tuned)
- Input: Detected vulnerable code
- Output: Natural language explanation of the vulnerability

## 🧪 Dataset

- ✅ Based on the **Juliet Test Suite v1.3** for C/C++
- ✅ Preprocessed into:
  - `source`: C++ code block
  - `explanation`: Natural language description

## 📦 Dependencies

```bash
pip install torch transformers datasets streamlit
