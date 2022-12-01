#for samples in 10 50 100 1000
# msmarco-fiqa-sample1000-lr1e-4-bs4-epoch20 -> 1e-4 bs 4 2 1, epoch 20 40
for samples in 1000 100 50 10
do
for dataset in fiqa
do
for lr in 3e-7 3e-6 1e-6
do
for bs in 8 16 32 64
do
for epoch in 100
do

export SAVE_EP=10
export SAVE_STEP=$((${samples} * ${SAVE_EP} / ${bs}))
echo ${SAVE_STEP}

export EXP_PATH=exps/few-shot/${dataset}/msmarco-${dataset}-sample${samples}-lr${lr}-bs${bs}-epoch${epoch}
if test -f "${EXP_PATH}/results.test.txt"; then
	continue
fi

python -m tevatron.driver.train \
  --output_dir ${EXP_PATH} \
  --model_name_or_path model_msmarco \
  --save_steps ${SAVE_STEP} \
  --dataset_name tev-beir-datasets/${dataset}_training.hf \
  --fp16 \
  --per_device_train_batch_size ${bs} \
  --max_train_samples ${samples} \
  --train_n_passages 1 \
  --learning_rate ${lr} \
  --q_max_len 64 \
  --p_max_len 128 \
  --num_train_epochs ${epoch} \
  --logging_steps 1000 \
  --overwrite_output_dir

# test after each saved epoch ckpt
for ep in 1 2 3 4 5 6 7 8 9 10
do

export CURR_STEP=$((${ep} * ${SAVE_STEP}))
export CURR_PATH=${EXP_PATH}/checkpoint-${CURR_STEP}

if test -f "${CURR_PATH}/results.test.txt"; then
    continue
fi

python -m tevatron.driver.encode \
  --output_dir=${CURR_PATH}/output \
  --model_name_or_path ${CURR_PATH} \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --dataset_name Tevatron/beir-corpus:${dataset} \
  --p_max_len 512 \
  --encoded_save_path ${CURR_PATH}/corpus_emb.pt

export EVAL=test
python -m tevatron.driver.encode \
  --output_dir=${CURR_PATH}/output \
  --model_name_or_path ${CURR_PATH} \
  --fp16 \
  --per_device_eval_batch_size 256 \
  --dataset_name Tevatron/beir:${dataset}/${EVAL} \
  --encode_is_qry \
  --q_max_len 64 \
  --encoded_save_path ${CURR_PATH}/queries_emb_${EVAL}.pt

python -m tevatron.faiss_retriever \
  --query_reps ${CURR_PATH}/queries_emb_${EVAL}.pt \
  --passage_reps ${CURR_PATH}/corpus_emb.pt \
  --depth 20 \
  --batch_size -1 \
  --save_text \
  --save_ranking_to ${CURR_PATH}/run.${dataset}.${EVAL}.txt

python -m tevatron.utils.format.convert_result_to_trec \
  --input ${CURR_PATH}/run.${dataset}.${EVAL}.txt \
  --output ${CURR_PATH}/run.${dataset}.${EVAL}.trec

python -m pyserini.eval.trec_eval -c -mrecip_rank -mndcg_cut.10 \
    trec_qrels/qrels.beir-v1.0.0-${dataset}.${EVAL}.txt \
    ${CURR_PATH}/run.${dataset}.${EVAL}.trec > ${CURR_PATH}/results.test.txt

rm ${CURR_PATH}/*.pt
rm ${CURR_PATH}/*.bin

done
done
done
done
done
done
