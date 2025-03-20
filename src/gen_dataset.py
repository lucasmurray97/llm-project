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
    if n == 1000:
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

# set from filtered doc ids
relevant_docs = set(filtered_doc_ids["relevant"]) 
selected_irrelevant_docs = set(filtered_doc_ids["irrelevant"])

# Merge docs
all_selected_docs = relevant_docs.union(selected_irrelevant_docs)

print("Filtered document IDs saved!")

# Generating embeddings for documents

# Load DPR question encoder
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# Load DPR context encoder
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")


doc_embeddings = []
doc_ids = []

for doc in tqdm(dataset.docs_iter(), desc="Processing Filtered Docs", total=dataset.docs_count()):
    if doc.doc_id in relevant_docs or doc.doc_id in selected_irrelevant_docs:
        inputs = context_tokenizer(doc.text, return_tensors="pt", truncation=True, max_length=512)
        doc_embedding = context_encoder(**inputs).pooler_output.detach()
        
        doc_embeddings.append(doc_embedding)
        doc_ids.append(doc.doc_id)

# Iterate over queries
filtered_queries = list(qrels.keys())
q_embeddings = []
q_rel_docs = {}
for query in dataset.queries_iter():
    if query.query_id in filtered_queries:
        q_rel_docs[query.query_id] = qrels[query.query_id]
        inputs = question_tokenizer(query.text, return_tensors="pt", truncation=True, max_length=128)
        q_embedding = question_encoder(**inputs).pooler_output.detach()
        q_embeddings.append(q_embedding)

# Convert lists to tensors
doc_embeddings = torch.cat(doc_embeddings)  # Shape: (N, 768)
q_embeddings = torch.cat(q_embeddings)  # Shape: (M, 768)

# Save as .pt files
torch.save(doc_embeddings, "doc_embeddings.pt")
torch.save(doc_ids, "doc_ids.pt")
torch.save(q_embeddings, "q_embeddings.pt")
# Save q_rel_docs dictionary
torch.save(q_rel_docs, "q_rel_docs.pt")


# Sanity check:
# Load the tensors
doc_embeddings = torch.load("doc_embeddings.pt")
doc_ids = torch.load("doc_ids.pt")
q_embeddings = torch.load("q_embeddings.pt")
q_rel_docs = torch.load("q_rel_docs.pt")

# Check 
filtered_queries = list(q_rel_docs.keys())
queries = []
for query in dataset.queries_iter():
    if query.query_id in filtered_queries:
        queries.append(query)
        if len(queries) == len(filtered_queries):
            break

for query in queries:
    relevant_docs = qrels.get(query.query_id, set())
    inter_doc_ids = relevant_docs.intersection(set(doc_ids))
    inter_gen_docs = relevant_docs.intersection(set(all_selected_docs))
    if len(inter_doc_ids) != len(relevant_docs) or len(inter_gen_docs) != len(relevant_docs):
        print(f"No relevant docs for query {query.query_id}")
        print(len(inter_doc_ids), len(relevant_docs), len(inter_gen_docs))
        raise ValueError("No relevant docs found!")