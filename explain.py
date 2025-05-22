import cv2
import numpy as np
import torch
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, FullGrad
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

CAM_METHODS = {
    "gradcam": GradCAM,
    "hirescam": HiResCAM,
    "scorecam": ScoreCAM,
    "gradcam++": GradCAMPlusPlus,
    "ablationcam": AblationCAM,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "fullgrad": FullGrad
}

def generate_cam_visualization(model, image_path, method='gradcam', target_class=None, device='cuda'):
    # Get normalization from trained model
    mean = model.datamodule.mean
    std = model.datamodule.std
    
    # Process image
    image = process_image(image_path, mean, std)
    
    # Get CAM method
    cam_class = CAM_METHODS.get(method.lower(), GradCAM)
    
    # Generate CAM
    target_layers = [model.model.layer4[-1]]
    with cam_class(model=model.model, target_layers=target_layers, use_cuda=(device=='cuda')) as cam:
        grayscale_cam = cam(input_tensor=image.unsqueeze(0).to(device),
                          targets=[ClassifierOutputTarget(target_class)] if target_class else None)
    
    # Overlay on original image
    rgb_img = cv2.imread(image_path)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    visualization = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)
    
    return visualization

def process_image(image_path, mean, std):
    # Read the image in grayscale mode (single channel)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # No need to normalize, the images are already normalized
    #image = np.float32(image) / 255
    # Add a channel dimension so that shape is (1, H, W)
    image = np.expand_dims(image, axis=2)
    # Convert H x W x 1 to 1 x H x W
    image = np.transpose(image, (2, 0, 1))
    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((299, 299)),
        transforms.Normalize(mean, std)
    ])(image)
    return tensor
