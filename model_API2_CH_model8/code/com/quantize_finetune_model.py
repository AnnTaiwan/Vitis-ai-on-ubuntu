import os
import torch
import torch.nn as nn
from pytorch_nndct.apis import torch_quantizer
import sys
import argparse
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image


IMAGE_SIZE = 128
transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Original Size: (640, 480)
        transforms.ToTensor()
    ])
# do data augmentation on spoof data
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),
    # transforms.RandomAffine(degrees=0, translate=(0.2, 0), scale=None, shear=0),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])
# 移除圖片周圍空白處
def crop_with_points(image_path):
    points = [(79, 57), (575, 428), (575, 57), (79, 426)]
    # Load the image
    img = Image.open(image_path)
    # original shape is 640*480
    # Define the four points
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]

    # Find the bounding box for cropping
    left = min(x1, x4)
    upper = min(y1, y2)
    right = max(x2, x3)
    lower = max(y3, y4)
    # Crop the image
    cropped_img = img.crop((left, upper, right, lower)) # 79, 57, 575, 426
	# After cropping, shape is 496*369
    return cropped_img

# when training, do data augmentation on spoof data
def generate_dataset(image_folder_path, df, batch_size=20, train_or_valid=True):
    image_paths = [os.path.join(image_folder_path, filename) for filename in os.listdir(image_folder_path)]

    # Load Labels
    labels = [df[df["file_name"] == os.path.basename(path)]["label"].values[0] for path in image_paths]
    
    # Apply Transformations
    images = [transform(crop_with_points(path).convert('RGB')) for path in image_paths]
    '''
    if train_or_valid:
        # Get the path name starting with "spoof"
        spoof_images_paths = [path for path in image_paths if os.path.basename(path).startswith("spoof")]
        # Do some data augmentation on spoof data
        aug_images = [augment_transform(crop_with_points(path).convert('RGB')) for path in spoof_images_paths]
        images += aug_images
        # Update the label due to increasing spoof data
        labels += [1 for i in range(len(spoof_images_paths))]
    '''
    # Create TensorDataset
    dataset = torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))

    # Create DataLoader for train and validation sets
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train_or_valid, pin_memory=True)

    return data_loader
    
# define CNN model
class CNN_model8(nn.Module):
    def __init__(self):
        super(CNN_model8, self).__init__()
        self.input_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 8, 5, stride=1), # kernel = 5*5
                nn.ReLU(),
                nn.BatchNorm2d(8), # 添加批次正規化層，對同一channel作正規化
                nn.MaxPool2d(2, stride=2) 
            )
        ])
        
        conv_filters = [12,30,16,8] # [12,16,12,8]
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(8, 12, 1),
                nn.ReLU(),
                nn.BatchNorm2d(12)
            ),
            nn.Sequential(
                nn.Conv2d(12, 12, 3),
                nn.ReLU(),
                nn.BatchNorm2d(12)
            ),
            nn.MaxPool2d(2, stride=2)
        ])
        for i in range(1, len(conv_filters)):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(conv_filters[i-1], conv_filters[i], 1),
                nn.ReLU(),
                nn.BatchNorm2d(conv_filters[i])
            ))
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(conv_filters[i], conv_filters[i], 3),
                nn.ReLU(),
                nn.BatchNorm2d(conv_filters[i])
            )
            )
            self.conv_layers.append(
                nn.MaxPool2d(2, stride=2)
            )
        # final layer output above is (8, 108, 108) 93312
        self.class_layers = nn.ModuleList([
            nn.Sequential(
                # Flatten layers
                nn.Linear(8*2*2, 2),       
            )
        ])
        
    def forward(self, x):
        for layer in self.input_layers:
            x = layer(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(-1, 8*2*2)
        for layer in self.class_layers:
            x = layer(x)
        return x  
        

def evaluate(model, val_loader, loss_fn, device):
    model.eval()
    model = model.to(device)
    total_samples = len(val_loader.dataset)
    total_loss = 0
    total_correct = 0
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)
        total_correct += (predicted == labels).sum().item()
        total_loss += loss.item() * images.size(0)
        
    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples
    
    return avg_accuracy, avg_loss



def quantize(model, quantized_model_name, quant_mode, batchsize, quantize_output_dir, finetune, deploy, train_dataset_path, val_dataset_path, df, device):
    # Construct paths
    quant_model_path = os.path.join(quantize_output_dir, quantized_model_name)

    # Override batch size if in test mode
    if quant_mode != 'test' and deploy:
        deploy = False
        print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
    if deploy and (batchsize != 1):
        print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
        batchsize = 1
    
    
    # Generate a random input tensor for quantization
    rand_in = torch.randn([batchsize, 3, IMAGE_SIZE, IMAGE_SIZE])

    # Create quantizer
    quantizer = torch_quantizer(quant_mode, model, rand_in, output_dir=quantize_output_dir)
    quantized_model = quantizer.quant_model

    
    # Load Images from a Folder
    image_folder_path = val_dataset_path
    print(f"Loading validating data from {image_folder_path}.")
    valid_dataloader = generate_dataset(image_folder_path, df, batch_size = batchsize, train_or_valid = False)
    
    # to get loss value after evaluation
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    # fast finetune model or load finetuned parameter before test
    if finetune == True:
        if quant_mode == 'calib':
            print("Start to finetune the quantized model")
            
            # Load Images from a Folder
            image_folder_path = train_dataset_path
            print(f"Loading training data from {image_folder_path}.")
            train_dataloader = generate_dataset(image_folder_path, df, batch_size = batchsize, train_or_valid = True)
            
            quantizer.fast_finetune(evaluate, (quantized_model, train_dataloader, loss_fn, device))
            quantized_model = quantizer.quant_model # Update quantized_model to use the fine-tuned parameters
            
        elif quant_mode == 'test':
            quantizer.load_ft_param()
            quantized_model = quantizer.quant_model # Update quantized_model to use the fine-tuned parameters
    		
    		
    # start to evaluate the model
    avg_accuracy = 0
    avg_loss = 0
    
    avg_accuracy, avg_loss = evaluate(quantized_model, valid_dataloader, loss_fn, device)
    
	# evaluation result
    print(f"avg_accuracy : {avg_accuracy}, avg_loss : {avg_loss}")
	
	# handle quantization result, export config
    if quant_mode == 'calib':
        quantizer.export_quant_config()
        # Save the quantized model
        torch.save(quantized_model.state_dict(), quant_model_path)
    if deploy:
        quantizer.export_xmodel(deploy_check=True, output_dir=quantize_output_dir)
	
    return

def main():
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}.")
    # load data info
    train_df = pd.read_csv("train_code/chinese_dataset_info/chinese_audio_info.csv")
	
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',  type=str, default='../../float/model_6.pth',    help='Path to float model')
    ap.add_argument('--quantized_model_name',  type=str, default='quantized_model_6.pth',    help='Quantized model name')
    ap.add_argument('--quant_mode', type=str, default='calib', choices=['calib', 'test'], help='Quantization mode (calib or test). Default is calib')
    ap.add_argument('--batchsize',  type=int, default=100, help='Testing batch size - must be an integer. Default is 100')
    ap.add_argument('--quantize',  type=bool, default=True, help='Whether to do quantization, default is True')
    ap.add_argument('--quantize_output_dir',  type=str, default='../../quantized/', help='Directory to save the quantized model')
    ap.add_argument('--finetune',  type=bool, default=True, help='Whether to do the finetune befroe quantization')
    ap.add_argument('--deploy',  type=bool, default=True, help='Whether to export xmodel for deployment')
    ap.add_argument('--train_dataset_path',  type=str, default='../../data/My_dataset/train_spec_LATrain_audio_shuffle23_NOT_preprocessing', help='Give the train dataset path')
    ap.add_argument('--val_dataset_path',  type=str, default='../../data/My_dataset/valid_spec_LATrain_audio_shuffle4_NOT_preprocessing', help='Give the validating dataset path')
    args = ap.parse_args()

    print('\n----------------------------------------')
    print('PyTorch version:', torch.__version__)
    print('Python version:', sys.version)
    print('----------------------------------------')
    print('Command line options:')
    print('--model:', args.model)
    print('--quantized_model_name:', args.quantized_model_name)
    print('--quant_mode:', args.quant_mode)
    print('--batchsize:', args.batchsize)
    print('--quantize:', args.quantize)
    print('--quantize_output_dir:', args.quantize_output_dir)
    print('--finetune:', args.finetune)
    print('--deploy:', args.deploy)
    print('--train_dataset_path:', args.train_dataset_path)
    print('--val_dataset_path:', args.val_dataset_path)
    
    
    print('----------------------------------------')

    # Load the model and weights
    model = CNN_model8()
    model.to(device)
    print("Loading model")
    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict)

    if args.quantize: # Check if quantization is requested
        print("Quantizing model")
        quantize(model, 
                 args.quantized_model_name, 
                 args.quant_mode, 
                 args.batchsize, 
                 args.quantize_output_dir, 
                 args.finetune, 
                 args.deploy, 
                 args.train_dataset_path, 
                 args.val_dataset_path, 
                 train_df, 
                 device)
                 
        print('Quantization finished. Results saved in:', args.quantize_output_dir)

if __name__ == "__main__":
    main()

