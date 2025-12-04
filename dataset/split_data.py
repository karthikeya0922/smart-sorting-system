"""
Train/Val/Test Split Script
Splits dataset into training, validation, andtest sets with stratification.
"""

import os
import shutil
import random
from pathlib import Path
from collections import Counter
from tqdm import tqdm


# Fixed random seed for reproducibility
SEED = 42
random.seed(SEED)

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def get_image_files(directory):
    """Get all image files from directory."""
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')
    files = []
    for ext in extensions:
        files.extend(Path(directory).glob(ext))
    return files


def split_dataset():
    """Split raw dataset into train/val/test sets."""
    dataset_dir = Path(__file__).parent
    raw_dir = dataset_dir / "raw"
    processed_dir = dataset_dir / "processed"
    
    print("=" * 60)
    print("TRAIN/VAL/TEST SPLIT")
    print("=" * 60)
    print()
    print(f"Split ratios: Train {TRAIN_RATIO:.0%} | Val {VAL_RATIO:.0%} | Test {TEST_RATIO:.0%}")
    print(f"Random seed: {SEED} (for reproducibility)")
    print()
    
    # Check raw directory
    if not raw_dir.exists():
        print("❌ Error: dataset/raw/ directory not found!")
        print("   Please run: python dataset/download_data.py first")
        return
    
    # Category mapping
    categories = {
        'glass': 'glass',
        'paper': 'paper',
        'cardboard': 'paper',
        'plastic': 'plastic',
        'metal': 'metal',
        'trash': 'non-recyclable',
        'organic': 'organic',
    }
    
    # Statistics
    total_stats = Counter()
    
    # Process each category
    for raw_cat, target_cat in categories.items():
        raw_cat_dir = raw_dir / raw_cat
        
        # Skip if category doesn't exist
        if not raw_cat_dir.exists():
            print(f"⚠️  Skipping {raw_cat}/ (not found)")
            continue
        
        # Get all images
        images = get_image_files(raw_cat_dir)
        
        if len(images) == 0:
            print(f"⚠️  Skipping {raw_cat}/ (no images)")
            continue
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        total = len(images)
        train_idx = int(total * TRAIN_RATIO)
        val_idx = train_idx + int(total * VAL_RATIO)
        
        # Split
        train_images = images[:train_idx]
        val_images = images[train_idx:val_idx]
        test_images = images[val_idx:]
        
        # Copy images to respective folders
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
        
        print(f"📂 Processing {raw_cat} → {target_cat}")
        print(f"   Total: {total} | Train: {len(train_images)} | Val: {len(val_images)} | Test: {len(test_images)}")
        
        for split_name, split_images in splits.items():
            target_dir = processed_dir / split_name / target_cat
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy images with progress bar
            for img_path in tqdm(split_images, desc=f"  Copying to {split_name}", leave=False):
                # Create unique filename
                new_name = f"{raw_cat}_{img_path.name}"
                target_path = target_dir / new_name
                
                # Copy file
                shutil.copy2(img_path, target_path)
            
            # Update stats
            total_stats[f"{split_name}_{target_cat}"] += len(split_images)
        
        print()
    
    # Print final statistics
    print("=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    print()
    
    final_categories = ['plastic', 'paper', 'metal', 'glass', 'organic', 'non-recyclable']
    
    # Table header
    print(f"{'Category':<15} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 60)
    
    grand_total = {'train': 0, 'val': 0, 'test': 0}
    
    for cat in final_categories:
        train_count = total_stats.get(f"train_{cat}", 0)
        val_count = total_stats.get(f"val_{cat}", 0)
        test_count = total_stats.get(f"test_{cat}", 0)
        total = train_count + val_count + test_count
        
        grand_total['train'] += train_count
        grand_total['val'] += val_count
        grand_total['test'] += test_count
        
        print(f"{cat:<15} {train_count:<10} {val_count:<10} {test_count:<10} {total:<10}")
    
    print("-" * 60)
    print(f"{'TOTAL':<15} {grand_total['train']:<10} {grand_total['val']:<10} {grand_total['test']:<10} {sum(grand_total.values()):<10}")
    print()
    
    # Save statistics to file
    stats_file = processed_dir / "split_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write("Dataset Split Statistics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Random Seed: {SEED}\n")
        f.write(f"Split Ratios: Train {TRAIN_RATIO:.0%}, Val {VAL_RATIO:.0%}, Test {TEST_RATIO:.0%}\n\n")
        f.write(f"{'Category':<15} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}\n")
        f.write("-" * 60 + "\n")
        
        for cat in final_categories:
            train_count = total_stats.get(f"train_{cat}", 0)
            val_count = total_stats.get(f"val_{cat}", 0)
            test_count = total_stats.get(f"test_{cat}", 0)
            total = train_count + val_count + test_count
            f.write(f"{cat:<15} {train_count:<10} {val_count:<10} {test_count:<10} {total:<10}\n")
        
        f.write("-" * 60 + "\n")
        f.write(f"{'TOTAL':<15} {grand_total['train']:<10} {grand_total['val']:<10} {grand_total['test']:<10} {sum(grand_total.values()):<10}\n")
    
    print(f"✅ Split complete! Statistics saved to: {stats_file}")
    print()
    print("📌 Next step:")
    print("   Run: python model/train.py")
    print()


if __name__ == "__main__":
    split_dataset()
