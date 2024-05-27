import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# Load the image
image_path = 'R.png'
image = Image.open(image_path).convert("RGB")

# Define the image transformations
preprocess = transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

# Load pre-trained DeepLabV3 model and set it to evaluation mode
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# Perform model inference on CPU
with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

# Assuming the clothing class in the segmentation map is labeled as '15' (this can vary)
mask = output_predictions == 15  # Change '15' to the actual class index for clothing
mask = mask.byte().cpu().numpy()  # Convert to numpy array

# Convert to binary mask (0 or 255)
mask = mask * 255
mask = mask.astype(np.uint8)

# Resize mask to the original image size
original_size = image.size[::-1]  # (height, width)
mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

# Optional: Apply morphological operations to clean up the mask
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Save the mask
mask_image = Image.fromarray(mask)
mask_image.save('D:\Website\masked.png')

# Display the original image and the mask for verification
mask_image.show()
image.show()
