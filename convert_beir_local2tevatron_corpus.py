from tqdm import tqdm
from datasets import Dataset

import beir
from beir.datasets.data_loader import GenericDataLoader

from tevatron.datasets.dataset import load_dataset

dataset_name = "bioasq"

corpus, queries, qrels = GenericDataLoader(data_folder="/home/y247xie/00_data/beir/{}".format(dataset_name)).load(split="train")

corpus_dataset = {"docid": [], "title": [], "text": []}

for docid in tqdm(corpus):
    corpus_dataset["docid"].append(docid)
    corpus_dataset["title"].append(corpus[docid]["title"])
    corpus_dataset["text"].append(corpus[docid]["text"])

# corpus.save_to_disk("{}_corpus.hf".format(dataset_name))

# train_dataset = {"query_id":[], "query": [], "positive_passages":[], "negative_passages":[]}

# for qid in tqdm(qrels):
#     row = {}
#     train_dataset["query_id"].append(qid)
#     train_dataset["query"].append(queries[qid])
#     pos_passages = []
#     neg_passages = []

#     for docid in qrels[qid]:
#         score = qrels[qid][docid]
#         if score == 1:
#             pos_passages.append(
#                 {
#                     "docid": docid, 
#                     "title":corpus[docid]["title"], 
#                     "text":corpus[docid]["text"]
#                 }
#             )
#         else:
#             pos_passages.append({
#                     "docid": docid, 
#                     "title":corpus[docid]["title"], 
#                     "text":corpus[docid]["text"]
#                 })
#     train_dataset["positive_passages"].append(pos_passages)
#     train_dataset["negative_passages"].append(neg_passages)
    
dataset = Dataset.from_dict(corpus_dataset)
dataset.save_to_disk("tev-beir-datasets/{}_corpus.hf".format(dataset_name))
