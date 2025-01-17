python -m tevatron.driver.train \
  --output_dir scifact_segment_test \
  --model_name_or_path bert-base-uncased \
  --save_steps 5000 \
  --dataset_name Tevatron/msmarco-passage \
  --segment_training True \
  --fp16 \
  --per_device_train_batch_size 4 \
  --train_n_passages 1 \
  --learning_rate 1e-5 \

  --q_max_len 64 \
  --p_max_len 128 \
  --num_train_epochs 2 \
  --gc_p_chunk_size 8 \
  --logging_steps 200 \
  --overwrite_output_dir
