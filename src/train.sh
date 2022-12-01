for bs in 16 8
do
for lr in 1e-5 3e-5 3e-6
do
CUDA_VISIBLE_DEVICES=2 python -m tevatron.driver.train \
  --output_dir model_scifact_xyq_bs${bs}_lr${lr} \
  --model_name_or_path facebook/dpr-question_encoder-single-nq-base \
  --save_steps 2000 \
  --dataset_name Tevatron/msmarco-passage \
  --fp16 \
  --per_device_train_batch_size ${bs} \
  --train_n_passages 8 \
  --learning_rate ${lr} \
  --q_max_len 32 \
  --p_max_len 256 \
  --num_train_epochs 10 \
  --logging_steps 500 \
  --overwrite_output_dir

done
done

