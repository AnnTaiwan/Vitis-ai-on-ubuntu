#!/bin/bash

echo "It's going to plot the audios in one folder into a bunch of mel_spec.img(s)"
if [ "$#" -lt 3 ]; then
    echo "Please follow this format: sh plot_mel_spec_just5sec_CH_dataset.sh <AUDIO_FOLDER_name> <TXT_FOLDER_name> <IMG_FOLDER_name> <label_name: 'spoof' or 'bonafide'>"
    exit 1
fi

AUDIO_FOLDER=$1
TXT_FOLDER=$2
IMG_FOLDER=$3
label=$4
# Iterate over each file in the audio folder
for audio_file in "$AUDIO_FOLDER"/*; do
    # Run the mel-spectrogram calculation for each file
    ./cal_mel_spec_ver5_just5sec "$audio_file" "$TXT_FOLDER"
done

echo "Done! Generated txt files in $TXT_FOLDER from $AUDIO_FOLDER"

# Iterate over each .txt file in the txt folder
for txt_file in "$TXT_FOLDER"/*.txt; do
    # Get the base name of the txt file (without the directory and extension)
    base_name=$(basename "$txt_file" .txt)
    # Correct string concatenation for the image name
    image_name="${label}_image_${base_name}"
    # Run the plot script for each txt file and save the image
    ./plot_mel_spec_from_txt_ver2 "$txt_file" "$IMG_FOLDER/$image_name.png"
    rm $txt_file
done

echo "Done! Generated images in $IMG_FOLDER from $TXT_FOLDER"

