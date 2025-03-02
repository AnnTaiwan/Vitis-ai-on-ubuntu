from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import os

from P_model_7 import IMAGE_SIZE, CNN_model7
from SpectrogramDataset import SpectrogramDataset

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

BATCH_SIZE_TEST = 50

dict1 = {
	"Image_name":[],
	"output_0":[],
	"output_1":[]
}
# Transformer
test_transformer = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

def get_test_dataloader(data_dir):
    test_image_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
    test_labels = [dev_df[dev_df["filename"] == os.path.basename(path)[5:-4]]["target"].values[0] for path in test_image_paths]
    test_dataloader = torch.utils.data.DataLoader(SpectrogramDataset(test_image_paths, test_labels, test_transformer),
                                  batch_size=BATCH_SIZE_TEST, shuffle=False, pin_memory=True)
    return test_dataloader

def print_tensor_hook(module, input, output):
    print(f'Layer: {module.__class__.__name__}')
    print(f'Output tensor shape: {output.shape}')
    if(len(output.shape) >= 3): 
        for channel in range(output.shape[1]):  # Iterate over channels
            print(f'Channel {channel}: {output[:, channel, :min(10, output.shape[2])].detach().cpu().numpy()}')
    else:
        print(f'Linear layer {output.shape}: {output.detach().cpu().numpy()}')


def register_hooks(model):
    for layer in model.children():
        layer.register_forward_hook(print_tensor_hook)
        if hasattr(layer, 'children'):
            register_hooks(layer)

def test_model(model, test_dataloader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    y_true = np.array([])
    y_pred = np.array([])
    with torch.no_grad():
        for _, (image, label) in enumerate(test_dataloader):
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = criterion(output, label)
            test_loss += loss.item() * image.size(0)

            # record the outputs of each columns
            dict1["output_0"].extend(output[:, 0].tolist())
            dict1["output_1"].extend(output[:, 1].tolist())
        	
            probs = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(probs, 1)

            y_pred = np.append(y_pred, predicted.cpu().numpy())
            y_true = np.append(y_true, label.cpu().numpy())

            total += label.size(0)

    avg_loss = test_loss / total

    print(f"Test Loss: {avg_loss:.4f}")

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"Precision: {precision * 100:.2f}")
    print(f"Recall: {recall * 100:.2f}")
    print(f"F1-score: {f1 * 100:.2f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Test on {device}.")
    dev_df = pd.read_csv("ASV_spoof2019/train_info.csv")
    model = CNN_model7()
    model.to(device)
    state_dict = torch.load('../../../float/model_7.pth')
    
    model.load_state_dict(state_dict)
    criterion = nn.CrossEntropyLoss()

    # add marks to print the layer data info
    # register_hooks(model) 

    # image_folder_path = r"C:\Users\User\Desktop\code\dataset_testing"
    #image_folder_path = "../../../data/My_dataset/spec_LAEval_audio_shuffle1_NOT_preprocessing"
    image_folder_path = "../../../data/My_dataset/valid_spec_LATrain_audio_shuffle4_NOT_preprocessing"

	# record the Image_name
    dict1["Image_name"] = [filename for filename in os.listdir(image_folder_path)]
	
	
    test_dataloader = get_test_dataloader(image_folder_path)
    print(f"Loaded test data from {image_folder_path}.")

    print("Testing...")
    test_model(model, test_dataloader, criterion)
    
    # turn dict1 into csv file
    output_df = pd.DataFrame(dict1)
    csv_file_path = f"output_model7_{os.path.basename(image_folder_path)}.csv"
    output_df.to_csv(csv_file_path, index=False)
    print(f"Data has been successfully exported to {csv_file_path}")
