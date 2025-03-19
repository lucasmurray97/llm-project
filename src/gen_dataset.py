import json
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
import faiss
import numpy as np
import torch
import ir_datasets
from tqdm import tqdm
import random

# Load the TREC CAsT 2020 dataset
dataset = ir_datasets.load("msmarco-passage/dev")  # Use train set
qrels = {}

print("Loading qrels...")
n = 0
for qrel in dataset.qrels_iter():
    if qrel.query_id not in qrels:
        qrels[qrel.query_id] = set()
        n += 1
    qrels[qrel.query_id].add(qrel.doc_id)
    if n == 100:
        break
# get rid of empty queries
qrels = {k: v for k, v in qrels.items() if v}
print(len(qrels), "qrels loaded!")

print("Looking for relevant docs")
# Collect relevant docs from qrels
relevant_docs = set()
for query_id, doc_ids in qrels.items():
    relevant_docs.update(doc_ids)

print(len(relevant_docs), "relevant documents found!")

# Select some irrelevant docs
all_docs = set(doc.doc_id for doc in dataset.docs_iter())  # All available doc IDs
irrelevant_docs = list(all_docs - relevant_docs)  # Remove relevant ones
print(len(irrelevant_docs), "irrelevant documents found!")
num_irrelevant = min(len(irrelevant_docs), 2 * len(relevant_docs))  # 5x irrelevant docs
selected_irrelevant_docs = set(irrelevant_docs[:num_irrelevant])

# Save the filtered document IDs to a JSON file
filtered_doc_ids = {"relevant": list(relevant_docs), "irrelevant": list(selected_irrelevant_docs)}
with open("filtered_doc_ids_ms.json", "w") as f:
    json.dump(filtered_doc_ids, f)

print("Filtered document IDs saved!")
