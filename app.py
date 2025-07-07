import torch
import streamlit as st
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# Paths to your trained models
clf_model_path = "D:/Python/DL/cpp_linter_model"
exp_model_path = "D:/Python/DL/codet5_explainer_model"

# Load tokenizers
clf_tokenizer = RobertaTokenizerFast.from_pretrained(clf_model_path)
exp_tokenizer = AutoTokenizer.from_pretrained(exp_model_path)

# Set device manually
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🚫 DO NOT use device_map or low_cpu_mem_usage here
clf_model = RobertaForSequenceClassification.from_pretrained(clf_model_path)
clf_model.to(device)

exp_model = AutoModelForSeq2SeqLM.from_pretrained(exp_model_path)
exp_model.to(device)

# 🔽 Streamlit UI
st.title("🛡️ AI-Powered C++ Vulnerability Analyzer")

code = st.text_area("Paste your C++ function code here:")

if st.button("🔍 Analyze Code"):
    if not code.strip():
        st.warning("Please paste some C++ code to analyze.")
    else:
        max_length = 512
        stride = 256
        tokens = clf_tokenizer(
            code,
            return_overflowing_tokens=True,
            max_length=max_length,
            stride=stride,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        chunk_predictions = []
        explanations = []

        st.info(f"🔢 Total chunks: {input_ids.size(0)}")
        progress = st.progress(0)

        for i in range(input_ids.size(0)):
            with torch.no_grad():
                clf_out = clf_model(input_ids[i:i+1], attention_mask=attention_mask[i:i+1])
            pred = torch.argmax(clf_out.logits, dim=-1).item()
            label = "✅ Safe" if pred == 0 else "❌ Vulnerable"
            chunk_predictions.append(label)

            if pred == 1:
                chunk_text = clf_tokenizer.decode(input_ids[i], skip_special_tokens=True)
                input_exp = exp_tokenizer(
                    chunk_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                ).to(device)

                with torch.no_grad():
                    generated_ids = exp_model.generate(
                        **input_exp,
                        max_length=128,
                        num_beams=4,
                        early_stopping=True,
                    )
                explanation = exp_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            else:
                explanation = "No issues found."

            explanations.append(explanation)
            progress.progress((i + 1) / input_ids.size(0))

        st.success("✅ Analysis Completed!")

        for idx, (label, explanation) in enumerate(zip(chunk_predictions, explanations)):
            with st.expander(f"🔍 Chunk {idx + 1}: {label}"):
                st.markdown(f"**Explanation:** {explanation}")
