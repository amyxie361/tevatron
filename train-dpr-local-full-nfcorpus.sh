#for samples in 10 50 100 1000
# 1e-4 8 4 2 1 20 40
for samples in 1000
do
for dataset in nfcorpus
do
for lr in 1e-3 1e-6 1e-7
do
for bs in 64 32 16 8 4 2 1
do
for epoch in 20 40
do
python -m tevatron.driver.train \
  --output_dir exps/few-shot/${dataset}/msmarco-${dataset}-sample${samples}-lr${lr}-bs${bs}-epoch${epoch} \
  --model_name_or_path model_msmarco \
  --save_steps 10000 \
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
done
done
done
done
done
