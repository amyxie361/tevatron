from tqdm import tqdm
from datasets import Dataset

import beir
from beir.datasets.data_loader import GenericDataLoader

from tevatron.datasets.dataset import load_dataset

dataset_name = "bioasq"

corpus, queries, qrels = GenericDataLoader(data_folder="/home/y247xie/00_data/beir/{}".format(dataset_name)).load(split="train")

train_dataset = {"query_id":[], "query": [], "positive_passages":[], "negative_passages":[]}

for qid in tqdm(qrels):
    row = {}
    train_dataset["query_id"].append(qid)
    train_dataset["query"].append(queries[qid])
    pos_passages = []
    neg_passages = []

    for docid in qrels[qid]:
        score = qrels[qid][docid]
        if score == 1:
            pos_passages.append(
                {
                    "docid": docid, 
                    "title":corpus[docid]["title"], 
                    "text":corpus[docid]["text"]
                }
            )
        else:
            pos_passages.append({
                    "docid": docid, 
                    "title":corpus[docid]["title"], 
                    "text":corpus[docid]["text"]
                })
    train_dataset["positive_passages"].append(pos_passages)
    train_dataset["negative_passages"].append(neg_passages)
    
dataset = Dataset.from_dict(train_dataset)
dataset.save_to_disk("tev-beir-datasets/{}_training.hf".format(dataset_name))
