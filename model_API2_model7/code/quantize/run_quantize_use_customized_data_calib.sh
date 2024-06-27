cd ../com/ 

python quantize_finetune_model.py \
  --model ../../float/model_7.pth \
  --quantized_model_name my_quantized_model_7.pth \
  --quant_mode calib\
  --batchsize 200\
  --quantize True \
  --quantize_output_dir ../../quantized/ \
  --finetune True \
  --deploy True \
  --train_dataset_path ../../data/My_dataset/train_spec_LATrain_audio_shuffle23_NOT_preprocessing \
  --val_dataset_path ../../data/My_dataset/valid_spec_LATrain_audio_shuffle4_NOT_preprocessing \
