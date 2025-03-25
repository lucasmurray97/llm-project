import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    DPRContextEncoder, DPRContextEncoderTokenizer,
    AdamW
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from tqdm import tqdm
import ir_datasets
import random
import matplotlib.pyplot as plt
import os

# --- Configuración general ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 100
NUM_NEGATIVES = 8

# --- Cargar modelos ---
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").eval().to(device)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# --- LoRA ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "key"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)
question_encoder = get_peft_model(question_encoder, lora_config).to(device)
question_encoder.train()

# --- Cargar datos ---
print("Construyendo dataset...")

queries = torch.load("queries_random.pt")
random_neg = torch.load("random_docs.pt")
positives = torch.load("positive_docs.pt")
print("Number of queries: ", len(queries))
print("Number of random docs: ", len(random_neg))
print("Number of positive docs: ", len(positives))

train_data = []
for q in positives:
    q_id, emb = positives[q]
    q_text = queries[q_id]
    neg_docs = random.sample(random_neg, NUM_NEGATIVES)
    
    train_data.append({
        "query": q_text,
        "positive": emb,
        "negatives": neg_docs
    })

hf_dataset = Dataset.from_list(train_data)
# divide in train/val
train_size = int(0.8 * len(hf_dataset))
val_size = len(hf_dataset) - train_size
hf_dataset = hf_dataset.train_test_split(test_size=val_size, shuffle=True)


# --- Collate function ---
def collate_fn(batch):
    queries = [item["query"] for item in batch]
    pos_docs = [item["positive"] for item in batch]
    neg_docs = [neg for item in batch for neg in item["negatives"]]

    q_tokens = question_tokenizer(queries, return_tensors="pt", padding=True, truncation=True)
    return q_tokens, pos_docs, neg_docs

dataloader = DataLoader(hf_dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(hf_dataset["test"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
optimizer = AdamW(question_encoder.parameters(), lr=2e-5)

# --- Entrenamiento con Early Stopping ---
print("Entrenando...")

loss_holder = []
val_loss_holder = []
best_loss = float('inf')
patience_counter = 0
# Get current path
save_path = os.getcwd()

for epoch in range(EPOCHS):
    question_encoder.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        q_inputs, p_inputs, n_inputs = batch
        q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
        pos_emb = torch.Tensor(p_inputs).to(device)
        neg_emb = torch.Tensor(n_inputs).to(device)

        q_emb = question_encoder(**q_inputs).pooler_output
        scores_pos = torch.matmul(q_emb, pos_emb.T)
        scores_neg = torch.matmul(q_emb, neg_emb.T)

        logits = torch.cat([scores_pos.diag().unsqueeze(1), scores_neg], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(device)

        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    # add in validation
    question_encoder.eval()
    with torch.no_grad():
        val_loss = 0
        for batch in val_dataloader:
            q_inputs, p_inputs, n_inputs = batch
            q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
            pos_emb = torch.Tensor(p_inputs).to(device)
            neg_emb = torch.Tensor(n_inputs).to(device)

            q_emb = question_encoder(**q_inputs).pooler_output
            scores_pos = torch.matmul(q_emb, pos_emb.T)
            scores_neg = torch.matmul(q_emb, neg_emb.T)

            logits = torch.cat([scores_pos.diag().unsqueeze(1), scores_neg], dim=1)
            labels = torch.zeros(logits.size(0), dtype=torch.long).to(device)

            loss = F.cross_entropy(logits, labels)
            val_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
    loss_holder.append(avg_loss)
    val_loss_holder.append(avg_val_loss)

    # Check for improvement
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        patience_counter = 0
        question_encoder.save_pretrained("models")
        question_tokenizer.save_pretrained("models")


# --- Gráfico de pérdida ---
plt.plot(loss_holder)
plt.plot(val_loss_holder)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig(f"{save_path}/loss.png")

