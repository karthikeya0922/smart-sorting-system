"""
Data Augmentation Utilities for Waste Classification
Provides augmentation transforms using albumentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_transforms(image_size=224):
    """
    Get training augmentation pipeline.
    
    Args:
        image_size (int): Target image size (default: 224)
    
    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    return A.Compose([
        # Resize
        A.Resize(image_size, image_size),
        
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
        
        # Color transforms
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        
        # Noise and blur
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        
        # Normalization (ImageNet stats)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        
        # Convert to tensor
        ToTensorV2()
    ])


def get_val_transforms(image_size=224):
    """
    Get validation/test augmentation pipeline (no augmentation, just preprocessing).
    
    Args:
        image_size (int): Target image size (default: 224)
    
    Returns:
        albumentations.Compose: Preprocessing pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def visualize_augmentation(image_path, num_samples=9):
    """
    Visualize augmentation effects on a single image.
    
    Args:
        image_path (str): Path to image
        num_samples (int): Number of augmented samples to generate
    """
    import matplotlib.pyplot as plt
    
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get transform (without normalization for visualization)
    transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
        A.GaussNoise(p=0.2),
    ])
    
    # Create subplot
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    fig.suptitle('Data Augmentation Examples', fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i == 0:
            # Show original
            augmented = A.Resize(224, 224)(image=image)['image']
            ax.set_title('Original')
        else:
            # Show augmented
            augmented = transform(image=image)['image']
            ax.set_title(f'Augmented {i}')
        
        ax.imshow(augmented)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset/augmentation_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Augmentation visualization saved to: dataset/augmentation_samples.png")


if __name__ == "__main__":
    # Example usage
    print("Data Augmentation Module")
    print("=" * 60)
    print()
    print("Available transforms:")
    print("  - Training: Flip, Rotate, ColorJitter, Noise, Blur")
    print("  - Validation: Resize + Normalize only")
    print()
    print("To visualize augmentation:")
    print("  >>> from dataset.augment import visualize_augmentation")
    print("  >>> visualize_augmentation('path/to/image.jpg')")
    print()
