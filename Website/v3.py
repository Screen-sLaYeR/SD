import torch
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np
import cv2

# Load the image
image_path = 'v3+inp.png'
image = Image.open(image_path).convert("RGB")

# Define the image transformations without resizing
preprocess = transforms.Compose([
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
clothing_class = 12  # Change '15' to the actual class index for clothing
mask = output_predictions == clothing_class
mask = mask.byte().cpu().numpy()  # Convert to numpy array

# Convert to binary mask (0 or 255)
mask = mask * 255
mask = mask.astype(np.uint8)

# Optional: Apply morphological operations to clean up the mask
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Ensure the mask has the same dimensions as the original image
mask_image = Image.fromarray(mask).convert("L")

# Create an RGBA version of the original image and mask
original_image_rgba = image.convert("RGBA")

# Create a red overlay where the mask is
red_overlay = Image.new("RGBA", original_image_rgba.size, (255, 0, 0, 0))
red_overlay.paste((255, 0, 0, 128), (0, 0), mask_image)

# Combine the original image with the red overlay
combined_image = Image.alpha_composite(original_image_rgba, red_overlay)

# Save the combined image
combined_image.save('D:\\Website\\combined_image_v3.png')

# Display the combined image for verification
combined_image.show()
