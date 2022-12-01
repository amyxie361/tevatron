python -m tevatron.faiss_retriever \
--query_reps queries_emb_scifact_segments2.pt \
--passage_reps corpus_emb_scifact_segments2.pt \
--depth 20 \
--batch_size -1 \
--save_text \
--save_ranking_to run.scifact_segments2.dev.txt
