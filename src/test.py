import ir_datasets

# Load the TREC CAsT 2020 dataset
dataset = ir_datasets.load("trec-cast/v1/2020")
for query in dataset.queries_iter():
    print(query.query_id, query.raw_utterance)
    break  # Remove this break to iterate over all queries












