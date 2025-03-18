from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
import faiss
import numpy as np
import torch
import ir_datasets
from tqdm import tqdm

# Load the TREC CAsT 2020 dataset
dataset = ir_datasets.load("trec-cast/v1/2020")
qrels = {}

for qrel in dataset.qrels_iter():
    if qrel.query_id not in qrels:
        qrels[qrel.query_id] = set()
    qrels[qrel.query_id].add(qrel.doc_id)  # Use `doc_id` instead of `relevant_doc_ids`


# Load DPR question encoder
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# Load DPR context encoder
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")


passages = [
    "COVID-19 symptoms include fever, cough, and difficulty breathing.",
    "Common symptoms are fever, tiredness, and dry cough.",
    "Some may experience loss of taste or smell."
]

# Encode passages
doc_embeddings = []
for passage in passages:
    inputs = context_tokenizer(passage, return_tensors="pt")
    doc_embedding = context_encoder(**inputs).pooler_output.detach().numpy()
    doc_embeddings.append(doc_embedding)

# Convert to FAISS index
doc_embeddings = np.vstack(doc_embeddings)
dimension = doc_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(doc_embeddings)


def retrieve_dpr(query, k=10):
    # Encode query
    inputs = question_tokenizer(query, return_tensors="pt")
    query_embedding = question_encoder(**inputs).pooler_output.detach().numpy()

    # Search in FAISS
    D, I = faiss_index.search(query_embedding, k)
    
    # Return results
    return [(passages[i], D[0][j]) for j, i in enumerate(I[0])]


def compute_top_k_accuracy(retrieved_docs, relevant_docs, k):
    """
    Computes the Top-K accuracy for retrieval results.
    
    :param retrieved_docs: List of retrieved document IDs.
    :param relevant_docs: Set of relevant document IDs.
    :param k: The value of K (e.g., 20 or 100).
    :return: 1 if at least one relevant document is in the top-K results, else 0.
    """
    return int(any(docid in relevant_docs for docid in retrieved_docs[:k]))

def evaluate_top_k(model_func, dataset, k=20, num_queries=100):
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

    for query in tqdm(dataset.queries_iter()):
        if count >= num_queries:
            break
        
        retrieved_docs = [doc[0] for doc in model_func(query.raw_utterance, k=k)]
        relevant_docs = qrels.get(query.query_id, set())

        top_k_total += compute_top_k_accuracy(retrieved_docs, relevant_docs, k)
        count += 1

    return top_k_total / count  # Compute average accuracy

# Compute Top-20 and Top-100 accuracy

top_20_dpr = evaluate_top_k(retrieve_dpr, dataset, k=20)
top_100_dpr = evaluate_top_k(retrieve_dpr, dataset, k=100)


# Print results
print(f"DPR Top-20 Accuracy: {top_20_dpr:.4f}")
print(f"DPR Top-100 Accuracy: {top_100_dpr:.4f}")

