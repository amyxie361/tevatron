for bs in 64
do
for lr in 1e-5
do
CUDA_VISIBLE_DEVICES=2 python -m tevatron.driver.train \
  --output_dir model_scifact_xyq_bs${bs}_lr${lr} \
  --model_name_or_path facebook/dpr-question_encoder-single-nq-base \
  --save_steps 2000 \
  --dataset_name Tevatron/scifact \
  --fp16 \
  --per_device_train_batch_size ${bs} \
  --train_n_passages 2 \
  --learning_rate ${lr} \
  --q_max_len 64 \
  --grad_cache \
  --p_max_len 512 \
  --num_train_epochs 50 \
  --gc_p_chunk_size 8 \
  --logging_steps 10 \
  --overwrite_output_dir

done
done
