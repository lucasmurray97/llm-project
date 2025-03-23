import torch
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
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

# Configuración general
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Cargar dataset
dataset = ir_datasets.load("msmarco-passage/dev")

# 2. Cargar modelos y tokenizers

context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_encoder.eval().to(device)


# 4. Construir dataset (pares pregunta + embedding de documento relevante)
train_data_random = []
train_data_positive = {}

# Get random queries:
q_rels = {i.query_id: i.doc_id for i in dataset.qrels_iter()}
queries_random = {}
inverse_queries_random = {}
MAX_QUERIES = 1000
for query in tqdm(dataset.queries_iter(), total=dataset.queries_count()):
    if query.query_id in q_rels.keys():
        text = query.text
        # get embeddings for positive documents
        num = random.random()
        if num < 0.1:
            text = query.text
            # get first relevant doc
            doc_id = q_rels[query.query_id]
            queries_random[query.query_id] = [text, doc_id]
            inverse_queries_random[doc_id] = query.query_id
            if len(queries_random) >= MAX_QUERIES:
                break
MAX_SAMPLES = 10000
# Save queries_random
torch.save(queries_random, "queries_random.pt")
print("Generating pool of random documents")
n = 0
m = 0
for doc in tqdm(dataset.docs_iter(), total = dataset.docs_count()):
    if doc.doc_id in inverse_queries_random.keys():
        with torch.no_grad():
            doc_text = doc.text
            context_inputs = context_tokenizer(doc_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            doc_embedding = context_encoder(**context_inputs).pooler_output.squeeze().cpu()
            train_data_positive[doc.doc_id] = (inverse_queries_random[doc_id], doc_embedding)
            m += 1
            if m >= MAX_QUERIES:
                break
    else:
        if n < MAX_SAMPLES:
            num = random.random()
            if num < 0.1:
                doc_text = doc.text
                context_inputs = context_tokenizer(doc_text, return_tensors="pt", truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    doc_embedding = context_encoder(**context_inputs).pooler_output.squeeze().cpu()
                    train_data_random.append(doc_embedding)
                    n += 1
        else:
            continue
# Save the pool of random documents
torch.save(train_data_random, "random_docs.pt")
torch.save(train_data_positive, "positive_docs.pt")
