from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
import faiss
import numpy as np
import torch
import ir_datasets
from tqdm import tqdm
import random
import json

# Load the TREC CAsT 2020 dataset
dataset = ir_datasets.load("msmarco-passage/dev")  # Use train set

# Load DPR question encoder
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# Load DPR context encoder
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")


# Load the tensors
doc_embeddings = torch.load("doc_embeddings.pt")
doc_ids = torch.load("doc_ids.pt")
q_embeddings = torch.load("q_embeddings.pt")
qrels = torch.load("q_rel_docs.pt")


# Convert to FAISS
filtered_doc_embeddings = np.vstack(doc_embeddings)
dimension = filtered_doc_embeddings.shape[1]
faiss.normalize_L2(doc_embeddings.numpy())
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(filtered_doc_embeddings)

print(f"Number of docs in FAISS: {faiss_index.ntotal}")
print(f"Number of doc IDs: {len(doc_ids)}")



def retrieve_dpr(query, k=10):
    # print("Query: ", query)
    # Encode query
    inputs = question_tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
    query_embedding = question_encoder(**inputs).pooler_output.detach().numpy()

    # Search in FAISS
    D, I = faiss_index.search(query_embedding, k)
    
    # Return document IDs instead of passages
    return [(doc_ids[i], D[0][j]) for j, i in enumerate(I[0])]



def compute_top_k_accuracy(retrieved_docs, relevant_docs, k):
    # print("Retrieved Docs: ", retrieved_docs)
    """
    Computes the Top-K accuracy for retrieval results.
    
    :param retrieved_docs: List of retrieved document IDs.
    :param relevant_docs: Set of relevant document IDs.
    :param k: The value of K (e.g., 20 or 100).
    :return: 1 if at least one relevant document is in the top-K results, else 0.
    """
    return int(any(docid in relevant_docs for docid in retrieved_docs[:k]))

def evaluate_top_k(model_func, dataset, k=20):
    print("Evaluating Top-K")
    """
    Evaluates a retrieval model using Top-K accuracy.
    
    :param model_func: Function that retrieves documents for a query.
    :param dataset: The dataset containing queries and relevance judgments.
    :param k: The value of K (e.g., 20 or 100).
    :param num_queries: Number of queries to test on.
    :return: The Top-K accuracy score.
    """
    top_k_total = 0
    count = 0

    for query in tqdm(dataset, desc="Evaluating", total=len(dataset)):
        
        retrieved_docs = [doc[0] for doc in model_func(query.text, k=k)]
        relevant_docs = qrels.get(query.query_id, set())  # Ensure qrels uses doc IDs
        # Check if relelevant docs are in the all the docs


        # print(f"Query: {query.text}")
        # print(f"Retrieved Docs: {retrieved_docs}")
        # print(f"Relevant Docs: {relevant_docs}")
        top_k_total += compute_top_k_accuracy(retrieved_docs, relevant_docs, k)
        count += 1

    return top_k_total / count  # Compute average accuracy

# Compute Top-20 and Top-100 accuracy
queries = []
for query in dataset.queries_iter():
    if query.query_id in qrels.keys():
        queries.append(query)
        if len(queries) == len(qrels):
            break

# Check if relevant docs are in the all the doc_ids iterate over the queries
for query in queries:
    relevant_docs = set(qrels[query.query_id])
    # print(f"Intersection: {relevant_docs.intersection(set(doc_ids))}")

top_5_dpr = evaluate_top_k(retrieve_dpr, queries, k=5)
print(f"DPR Top-5 Accuracy: {top_5_dpr:.4f}")
top_20_dpr = evaluate_top_k(retrieve_dpr, queries, k=20)
print(f"DPR Top-20 Accuracy: {top_20_dpr:.4f}")
top_100_dpr = evaluate_top_k(retrieve_dpr, queries, k=100)
print(f"DPR Top-100 Accuracy: {top_100_dpr:.4f}")

# Save the results
results = {
    "DPR Top-5": top_5_dpr,
    "DPR Top-20": top_20_dpr,
    "DPR Top-100": top_100_dpr
}

with open("results_dpr.json", "w") as f:
    json.dump(results, f)
