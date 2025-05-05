import matplotlib.pyplot as plt
import torch

def plot_cxr_images(images, titles, figsize=(12, 4)):
    """
    Plots CXR images (as PyTorch tensors) in grayscale side by side with titles.
    
    Parameters:
    - images: List of 2D (or 3D) torch.Tensor objects (CXR images).
    - titles: List of strings, one for each image.
    - figsize: Tuple (width, height) for the matplotlib figure.
    """
    if len(images) != len(titles):
        raise ValueError("The number of images must match the number of titles.")
    
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        img = images[i]
        
        img = img.detach().cpu().numpy()
        
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(titles[i], fontsize=10)
        ax.axis('off')

    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.tight_layout(pad=0.5)
    plt.show()