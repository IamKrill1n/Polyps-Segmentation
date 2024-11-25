import argparse
import torch
import cv2
import numpy as np
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

color_dict= {0: (0, 0, 0),
             1: (255, 0, 0),
             2: (0, 255, 0)}
def mask_to_rgb(mask, color_dict):
    output = np.zeros((mask.shape[0], mask.shape[1], 3))

    for k in color_dict.keys():
        output[mask==k] = color_dict[k]

    return np.uint8(output)  

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    return parser.parse_args()

def main():
    args = parse_args()
    image_path = args.image_path

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = smp.UnetPlusPlus(
        encoder_name='resnet34', 
        encoder_weights=None, 
        in_channels=3, 
        classes=3
    )
    checkpoint = torch.load('model_checkpoint\model.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # Define transforms
    transform = Compose([
        Resize(256, 256),
        Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Load and preprocess the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image_rgb)
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Convert mask to RGB
    mask = mask_to_rgb(mask, color_dict)
    # Resize mask back to original size
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2RGB)

    # Save the mask
    cv2.imwrite('output.jpg', mask_resized)

if __name__ == '__main__':
    main()