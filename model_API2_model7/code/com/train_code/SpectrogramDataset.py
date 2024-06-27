from PIL import Image
import numpy as np
import torch

class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames    # 資料集的所有檔名
        self.labels = labels          # 影像的標籤
        self.transform = transform    # 影像的轉換方式
 
    def __len__(self):
        return len(self.filenames)    # return DataSet 長度
 
    def __getitem__(self, idx):       # idx: Inedx of filenames
        # Transform image
        image = self.transform(self._crop_with_points(self.filenames[idx]).convert('RGB')) 
        label = np.array(self.labels[idx])
        return image, label           # return 模型訓練所需的資訊
    
    def _crop_with_points(self, image_path):
    
	    points = [(79, 57), (575, 428), (575, 57), (79, 426)]
	    # Load the image
	    img = Image.open(image_path)
	    
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
	    cropped_img = img.crop((left, upper, right, lower))

	    return cropped_img
