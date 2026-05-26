DATA_DIR=/media/prasanna/4TB/work_data/step1_processed/DFN5B-CLIP-ViT-H-14-378/cuhk
MODEL="hf-hub:apple/DFN5B-CLIP-ViT-H-14-378"
BATCH_SIZE=8
SERIES_EMBEDDING_BATCH_SIZE=128
LR=1e-5
WD=0.1
EPOCHS=30
WARMUP=100
WORKERS=4
LOGS_DIR=/media/prasanna/4TB/work_data/step2_finetune_logs/DFN5B-CLIP-ViT-H-14-378/cuhk
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
  --name cuhk-series \
  --report-to tensorboard

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
  --name cuhk-subseries \
  --report-to tensorboard
