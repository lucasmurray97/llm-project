from pyserini.search import SimpleSearcher



# Initialize the searcher with the index directory
searcher = SimpleSearcher('lucene-index.msmarco-v1-passage.20221004.252b5e/')

# Set BM25 parameters
searcher.set_bm25(k1=0.9, b=0.4)

# Example query
query = "What are the symptoms of COVID-19?"
hits = searcher.search(query, k=5)

# Display results
for i, hit in enumerate(hits):
    print(f"{i+1}. {hit.docid}: {hit.contents} (Score: {hit.score})")

def retrieve_bm25(query, k=10):
    hits = searcher.search(query, k)
    return [(hit.docid, hit.contents, hit.score) for hit in hits]

# Example
query = "What are the symptoms of COVID-19?"
results = retrieve_bm25(query)

# Print results
for i, (docid, text, score) in enumerate(results):
    print(f"{i+1}. Doc ID: {docid} | Score: {score}")
    print(f"   {text}\n")
