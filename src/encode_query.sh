python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path model_msmarco \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name Tevatron/msmarco-passage/dev \
  --encoded_save_path query_emb.pkl \
  --q_max_len 32 \
  --encode_is_qry
