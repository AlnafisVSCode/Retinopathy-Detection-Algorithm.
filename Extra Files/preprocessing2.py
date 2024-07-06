import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np

def compute_dataset_statistics(loader):
    sum_r, sum_g, sum_b = 0.0, 0.0, 0.0
    sum_sq_r, sum_sq_g, sum_sq_b = 0.0, 0.0, 0.0
    total_pixels = 0
    processed_images = 0

    for images, _ in loader:
        r = images[:, 0, :, :]
        g = images[:, 1, :, :]
        b = images[:, 2, :, :]

        sum_r += r.sum().item()
        sum_g += g.sum().item()
        sum_b += b.sum().item()

        sum_sq_r += (r ** 2).sum().item()
        sum_sq_g += (g ** 2).sum().item()
        sum_sq_b += (b ** 2).sum().item()

        total_pixels += r.numel()

        processed_images += images.size(0)
        print(f"Processed {processed_images} images so far...")

    mean_r = sum_r / total_pixels
    mean_g = sum_g / total_pixels
    mean_b = sum_b / total_pixels

    std_r = ((sum_sq_r / total_pixels) - (mean_r ** 2)) ** 0.5
    std_g = ((sum_sq_g / total_pixels) - (mean_g ** 2)) ** 0.5
    std_b = ((sum_sq_b / total_pixels) - (mean_b ** 2)) ** 0.5

    return [mean_r, mean_g, mean_b], [std_r, std_g, std_b]

# Define transform to convert images to tensors
to_tensor_transform = transforms.Compose([
    transforms.Resize((960,540)),
    transforms.ToTensor()
])

# Load dataset with ImageFolder
dataset = ImageFolder(root='train', transform=to_tensor_transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=64)

"""

960,540
mean = np.array([0.2641, 0.1858, 0.1350])                                                                               
std = np.array([0.3007, 0.2171, 0.1713])

"""

mean, std = compute_dataset_statistics(data_loader)

print(f'mean = np.array([{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}])')
print(f'std = np.array([{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}])')
