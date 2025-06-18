cd ../com/ 

python quantize_finetune_model.py \
  --model ../../float/model_13_ENG_ver1.pth \
  --quantized_model_name quantized_model_13_ENG_ver1.pth \
  --quant_mode calib\
  --batchsize 50\
  --quantize True \
  --quantize_output_dir ../../quantized/ \
  --finetune True \
  --deploy True \
  --train_dataset_path ../../data/My_dataset/subset_train \
  --val_dataset_path ../../data/My_dataset/subset_val \
