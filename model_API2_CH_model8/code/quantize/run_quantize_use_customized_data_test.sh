cd ../com/ 

python quantize_finetune_model.py \
  --model ../../float/model_8_ver_epoch15_LR001.pth \
  --quantized_model_name My_quantized_model_8_ver_epoch15_LR001.pth\
  --quant_mode test\
  --batchsize 1 \
  --quantize True \
  --quantize_output_dir ../../quantized/ \
  --finetune True \
  --deploy True \
  --train_dataset_path ../../data/My_dataset/vitis_ai_train_subset1000 \
  --val_dataset_path ../../data/My_dataset/vitis_ai_valid_CH \
