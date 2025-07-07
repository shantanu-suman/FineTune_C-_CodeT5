import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Path to your trained model
model_path = "D:/Python/DL/codet5_explainer_model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")

# Example vulnerable C++ function
cpp_code = """void bad() {
    char dataBuffer[100];
    strcpy(dataBuffer, "This string is too long!");
    printf("%s\\n", dataBuffer);
}"""

# Tokenize the code
inputs = tokenizer(
    cpp_code,
    return_tensors="pt",
    max_length=512,
    truncation=True,
    padding="max_length"
).to(model.device)

# Generate explanation
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

# Decode and print
explanation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("🧠 Explanation:", explanation)
