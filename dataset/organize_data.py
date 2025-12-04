"""
Data Organization Script for Waste Classification
Organizes downloaded images into proper category structure.
"""

import os
import shutil
from pathlib import Path
from collections import Counter


# Category mapping from TrashNet to our 6 categories
CATEGORY_MAPPING = {
    'glass': 'glass',
    'paper': 'paper',
    'cardboard': 'paper',  # Merge cardboard into paper
    'plastic': 'plastic',
    'metal': 'metal',
    'trash': 'non-recyclable',
}


def count_images(directory):
    """Count images in directory."""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    return len([f for f in Path(directory).glob('**/*') 
                if f.suffix.lower() in extensions])


def organize_dataset():
    """Organize raw images into category folders."""
    dataset_dir = Path(__file__).parent
    raw_dir = dataset_dir / "raw"
    processed_dir = dataset_dir / "processed"
    
    print("=" * 60)
    print("DATASET ORGANIZATION")
    print("=" * 60)
    print()
    
    # Check if raw directory exists
    if not raw_dir.exists():
        print("❌ Error: dataset/raw/ directory not found!")
        print("   Please run: python dataset/download_data.py first")
        return
    
    # Create processed directory structure
    print("📁 Creating directory structure...")
    categories = ['plastic', 'paper', 'metal', 'glass', 'organic', 'non-recyclable']
    
    for split in ['train', 'val', 'test']:
        for category in categories:
            category_dir = processed_dir / split / category
            category_dir.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Created: processed/{split}/{category}/")
    
    print()
    
    # Count images in raw directory
    print("📊 Analyzing raw dataset...")
    image_counts = Counter()
    
    for item in raw_dir.iterdir():
        if item.is_dir():
            count = count_images(item)
            image_counts[item.name] = count
            print(f"  - {item.name}: {count} images")
    
    total_images = sum(image_counts.values())
    print(f"\n  Total: {total_images} images found")
    print()
    
    if total_images == 0:
        print("❌ No images found in dataset/raw/")
        print("   Please download and extract the dataset first.")
        print()
        print("📝 Instructions:")
        print("   1. Download TrashNet from Kaggle")
        print("   2. Extract images to dataset/raw/")
        print("   3. Organize into category subfolders:")
        for cat in CATEGORY_MAPPING.keys():
            print(f"      - dataset/raw/{cat}/")
        print()
        return
    
    print("✅ Dataset organization complete!")
    print()
    print("📌 Next step:")
    print("   Run: python dataset/split_data.py")
    print()


if __name__ == "__main__":
    organize_dataset()
