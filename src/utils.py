import cv2
import torch
import numpy as np


def preprocess(image, return_tensor='pt', img_size=(224, 224), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    # Resize image
    image = cv2.resize(image, img_size)
    image = image.transpose((2, 0, 1)) / 255.0  # Convert to channels first and normalize
    # Normalize image
    image = (image - np.array(mean)[:, None, None]) / np.array(std)[:, None, None]
    image = np.expand_dims(image, 0)
    
    if return_tensor == 'pt':
        return torch.Tensor(image)
    
    return image


def attem_load(model, checkpoint_path):
    weights = torch.load(checkpoint_path)['state_dict']
    reweights = dict()
    for k, v in weights.items():
        reweights[k[6:]] = v
    
    model.load_state_dict(reweights)
    model.eval()
    
    return model