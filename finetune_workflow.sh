DATA_DIR=./out/cuhk_end_to_end
MODEL="hf-hub:timm/ViT-SO400M-14-SigLIP2-378"
PRETRAINED="dfn5b"
BATCH_SIZE=8
SERIES_EMBEDDING_BATCH_SIZE=128
LR=1e-5
WD=0.1
EPOCHS=3
WARMUP=100
WORKERS=4
LOGS_DIR=./logs
DEVICE=cuda

echo -e "\n=== Finetune on series ==="
python -m open_clip_train.main \
  --save-frequency 3 \
  --zeroshot-frequency 0 \
  --dataset-type series_json \
  --train-data "$DATA_DIR/data/train_series.json" \
  --val-data "$DATA_DIR/data/val_series.json" \
  --model "$MODEL" \
  --batch-size $BATCH_SIZE \
  --series-embedding-batch-size $SERIES_EMBEDDING_BATCH_SIZE \
  --lr $LR \
  --wd $WD \
  --epochs $EPOCHS \
  --warmup $WARMUP \
  --workers $WORKERS \
  --device $DEVICE \
  --logs "$LOGS_DIR" \
  --siglip \
  --name cuhk-series

echo -e "\n=== Finetune on subseries ==="
python -m open_clip_train.main \
  --save-frequency 3 \
  --zeroshot-frequency 0 \
  --dataset-type series_json \
  --train-data "$DATA_DIR/subseries_split_data/train_series.json" \
  --val-data "$DATA_DIR/subseries_split_data/val_series.json" \
  --model "$MODEL" \
  --batch-size $BATCH_SIZE \
  --series-embedding-batch-size $SERIES_EMBEDDING_BATCH_SIZE \
  --lr $LR \
  --wd $WD \
  --epochs $EPOCHS \
  --warmup $WARMUP \
  --workers $WORKERS \
  --device $DEVICE \
  --logs "$LOGS_DIR" \
  --siglip \
  --name cuhk-subseries
