##### Code for the evaluation of BM25 ########

import json
import nltk
import polars as pl
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from joblib import Parallel, delayed
import ir_datasets
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy

############# Dataset loading + relevant doc selection


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
num_irrelevant = min(len(irrelevant_docs), 3 * len(relevant_docs))  # 5x irrelevant docs
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

#######################################

############ Preprocessing of the data

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load spaCy model for lemmatization
nlp = spacy.load("en_core_web_sm")

dataset = ir_datasets.load("msmarco-passage/dev")  # Adjust dataset if needed

with open("filtered_doc_ids_ms.json", "r") as f:
    filtered_doc_ids = json.load(f)

relevant_docs = set(filtered_doc_ids["relevant"])
selected_irrelevant_docs = set(filtered_doc_ids["irrelevant"])
all_selected_docs = relevant_docs.union(selected_irrelevant_docs)

def preprocess(text):
    # Tokenize and remove stopwords
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize using spaCy
    doc = nlp(" ".join(tokens))
    lemmatized_tokens = [token.lemma_ for token in doc if token.is_alpha]  # Only keep alphabetic words

    return lemmatized_tokens

doc_texts = {}
for doc in tqdm(dataset.docs_iter(), desc="Processing Docs"):
    if doc.doc_id in all_selected_docs:
        doc_texts[doc.doc_id] = preprocess(doc.text)

with open("preprocessed_docs.jsonl", "w") as f:
    for doc_id, tokens in doc_texts.items():
        f.write(json.dumps({"doc_id": doc_id, "tokens": tokens}) + "\n")

print("Preprocessed documents saved as 'preprocessed_docs.jsonl'.")

############################

########### BM25 functions

doc_ids = []
tokenized_corpus = []

with open("preprocessed_docs.jsonl", "r") as f:
    for line in f:
        doc = json.loads(line)
        doc_ids.append(doc["doc_id"])
        tokenized_corpus.append(doc["tokens"])

# Initialize BM25
bm25 = BM25Okapi(tokenized_corpus)
print(f"BM25 initialized with {len(tokenized_corpus)} documents.")


# BM25 Query Function
def retrieve_bm25(query, k=10):
    query_tokens = preprocess(query)  # Use same preprocessing as docs
    scores = bm25.get_scores(query_tokens)
    
    # Get top-k results
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(doc_ids[i], scores[i]) for i in top_k_indices]

################################



################# Evaluate BM25 for the top-K accuracy


# Evaluate Top-K accuracy for BM25
def evaluate_top_k_bm25(model_func, dataset, qrels, k=20):
    print("Evaluating Top-K (BM25)")
    top_k_total = 0
    count = 0

    # Define compute_top_k_accuracy function
    def compute_top_k_accuracy(retrieved_docs, relevant_docs, k):
        """Computes top-k accuracy."""
        num_relevant_retrieved = len(set(retrieved_docs[:k]) & relevant_docs)
        if len(relevant_docs) == 0:  # Handle empty relevance sets
            return 0.0  
        return num_relevant_retrieved / min(k, len(relevant_docs))

    for query in tqdm(dataset, desc="Evaluating BM25", total=len(dataset)):
        retrieved_docs = [doc[0] for doc in model_func(query.text, k=k)]
        relevant_docs = qrels.get(query.query_id, set())
        top_k_total += compute_top_k_accuracy(retrieved_docs, relevant_docs, k)
        count += 1

    return top_k_total / count

# Select queries for evaluation
queries = [query for query in dataset.queries_iter() if query.query_id in qrels.keys()]

# Compute Top-20 and Top-100 accuracy for BM25
top_20_bm25 = evaluate_top_k_bm25(retrieve_bm25, queries, qrels, k=20)
top_100_bm25 = evaluate_top_k_bm25(retrieve_bm25, queries, qrels, k=100)

# Print results
print(f"BM25 Top-20 Accuracy: {top_20_bm25:.4f}")
print(f"BM25 Top-100 Accuracy: {top_100_bm25:.4f}")