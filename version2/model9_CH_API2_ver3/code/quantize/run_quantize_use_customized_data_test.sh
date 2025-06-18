cd ../com/ 

python quantize_finetune_model.py \
  --model ../../float/model_9_CH_ver1.pth \
  --quantized_model_name quantized_model_9_CH_ver1.pth\
  --quant_mode test\
  --batchsize 1 \
  --quantize True \
  --quantize_output_dir ../../quantized/ \
  --finetune True \
  --deploy True \
  --train_dataset_path ../../data/My_dataset/subset_train_mel_spec \
  --val_dataset_path ../../data/My_dataset/val_mel_spec_padding_original_audio \
