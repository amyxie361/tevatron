python -m tevatron.faiss_retriever \
--query_reps query_emb.pkl \
--passage_reps 'corpus_emb.*.pkl' \
--depth 100 \
--batch_size -1 \
--save_text \
--save_ranking_to rank.txt
