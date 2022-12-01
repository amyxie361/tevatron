for samples in 10 50 100 1000 
do
for dataset in fiqa bioasq nfcorpus
do
python -m tevatron.driver.train \
  --output_dir msmarco-${dataset}-$samples \
  --model_name_or_path model_msmarco \
  --save_steps ${samples} \
  --dataset_name tev-beir-datasets/${dataset}_training.hf \
  --fp16 \
  --per_device_train_batch_size 65 \
  --max_train_samples 100 \
  --train_n_passages 1 \
  --learning_rate 1e-5 \
  --q_max_len 64 \
  --p_max_len 128 \
  --num_train_epochs 40 \
  --logging_steps 500 \
  --overwrite_output_dir
done
done
