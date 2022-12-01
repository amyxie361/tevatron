#for samples in 10 50 100 1000
# msmarco-fiqa-sample1000-lr1e-4-bs4-epoch20 -> 1e-4 bs 4 2 1, epoch 20 40
for samples in 1000
do
for dataset in bioasq
do
for lr in 1e-5
do
for bs in 64 #32 16 8 4 2 1
do
for epoch in 20 #40
do

# --dataset_name Tevatron/beir-corpus:${dataset} \

export EXP_PATH=exps/few-shot/${dataset}/msmarco-${dataset}-sample${samples}-lr${lr}-bs${bs}-epoch${epoch}
mkdir -p ${EXP_PATH}/corpus_embeddings/
for s in $(seq -f "%02g" 0 99)
do
python -m tevatron.driver.encode \
  --output_dir=${EXP_PATH}/output \
  --model_name_or_path ${EXP_PATH} \
  --fp16 \
  --per_device_eval_batch_size  \
  --dataset_name tev-beir-datasets/${dataset}_corpus.hf  \
  --encode_num_shard 100 \
  --encode_shard_index ${s} \
  --p_max_len 512 \
  --encoded_save_path ${EXP_PATH}/corpus_embeddings/corpus_emb.${s}.pkl
done

export EVAL=test
python -m tevatron.driver.encode \
  --output_dir=${EXP_PATH}/output \
  --model_name_or_path ${EXP_PATH} \
  --fp16 \
  --per_device_eval_batch_size 512 \
  --dataset_name tev-beir-datasets/${dataset}_${EVAL}.hf \
  --encode_is_qry \
  --q_max_len 64 \
  --encoded_save_path ${EXP_PATH}/queries_emb_${EVAL}.pt

python -m tevatron.faiss_retriever \
  --query_reps ${EXP_PATH}/queries_emb_${EVAL}.pt \
  --passage_reps "${EXP_PATH}/corpus_emb.*.pkl" \
  --depth 100 \
  --batch_size -1 \
  --save_text \
  --save_ranking_to ${EXP_PATH}/run.${dataset}.${EVAL}.txt

python -m tevatron.utils.format.convert_result_to_trec \
  --input ${EXP_PATH}/run.${dataset}.${EVAL}.txt \
  --output ${EXP_PATH}/run.${dataset}.${EVAL}.trec 

python -m pyserini.eval.trec_eval -c -mrecip_rank -mndcg_cut.10 trec_qrels/qrels.beir-v1.0.0-${dataset}.${EVAL}.txt ${EXP_PATH}/run.${dataset}.${EVAL}.trec > ${EXP_PATH}/results.test.txt

#rm ${EXP_PATH}/corpus_emb.pt

done
done
done
done
done
