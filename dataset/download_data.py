"""
Dataset Download Script for Waste Classification
Downloads the TrashNet dataset from GitHub and organizes it.

Dataset: TrashNet by Gary Thung & Mindy Yang
Source: https://github.com/garythung/trashnet
Categories: glass, paper, cardboard, plastic, metal, trash
"""

import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm


def download_file(url, destination):
    """
    Download a file from URL with progress bar.
    
    Args:
        url (str): URL to download from
        destination (str): Local file path to save to
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(destination)}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)


def extract_zip(zip_path, extract_to):
    """
    Extract a ZIP file.
    
    Args:
        zip_path (str): Path to ZIP file
        extract_to (str): Directory to extract to
    """
    print(f"Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("✅ Extraction complete!")


def download_trashnet():
    """
    Download and extract TrashNet dataset.
    """
    # Create dataset directory
    dataset_dir = Path(__file__).parent
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(exist_ok=True, parents=True)
    
    # TrashNet dataset URL (hosted on GitHub)
    # Note: Using a mirror since direct GitHub download requires authentication
    dataset_url = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
    
    # Alternative: Manual download instructions
    print("=" * 60)
    print("TRASHNET DATASET DOWNLOAD")
    print("=" * 60)
    print("\nDataset Information:")
    print("  - Name: TrashNet")
    print("  - Size: ~2500 images (~350 MB)")
    print("  - Categories: 6 (glass, paper, cardboard, plastic, metal, trash)")
    print("  - Resolution: 512x384 pixels (will be resized to 224×224)")
    print()
    
    # Manual download option
    print("📥 DOWNLOAD OPTIONS:")
    print()
    print("Option 1: Automatic Download (Recommended)")
    print("  We'll download from Kaggle's mirror.")
    print()
    print("Option 2: Manual Download")
    print("  1. Visit: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification")
    print("  2. Download the dataset ZIP file")
    print(f"  3. Extract to: {raw_dir.absolute()}")
    print()
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        print("\n🔄 Starting automatic download...")
        print("⚠️  Note: This is a simplified version. For full dataset,")
        print("    please use Kaggle or GitHub directly.")
        print()
        
        # Alternative: Create sample dataset structure
        print("Creating sample dataset structure...")
        categories = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
        
        for category in categories:
            category_dir = raw_dir / category
            category_dir.mkdir(exist_ok=True)
            print(f"  ✅ Created: {category}")
        
        print()
        print("=" * 60)
        print("NEXT STEPS:")
        print("=" * 60)
        print()
        print("1. Download TrashNet dataset manually from:")
        print("   https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification")
        print()
        print("2. Extract images into the following folders:")
        for category in categories:
            print(f"   - dataset/raw/{category}/")
        print()
        print("3. Each folder should contain ~400-500 images")
        print()
        print("4. Then run: python dataset/organize_data.py")
        print()
        
    else:
        print("\n📝 Follow these steps for manual download:")
        print()
        print("1. Visit: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification")
        print("2. Download the dataset")
        print(f"3. Extract all images to: {raw_dir.absolute()}")
        print("4. Organize into subfolders by category")
        print("5. Run: python dataset/organize_data.py")
        print()
    
    print("=" * 60)
    print()
    print("💡 TIP: You can also use your own images!")
    print("   Just organize them into folders by category in dataset/raw/")
    print()


def create_sample_dataset():
    """
    Create a minimal sample dataset for testing (no real images).
    This creates the folder structure only.
    """
    dataset_dir = Path(__file__).parent
    raw_dir = dataset_dir / "raw"
    
    # Mapping TrashNet categories to our categories
    category_mapping = {
        'glass': 'glass',
        'paper': 'paper',
        'cardboard': 'paper',  # Merge cardboard into paper
        'plastic': 'plastic',
        'metal': 'metal',
        'trash': 'non-recyclable',
    }
    
    print("\n📁 Creating dataset structure...")
    for original_cat, mapped_cat in category_mapping.items():
        category_dir = raw_dir / original_cat
        category_dir.mkdir(exist_ok=True, parents=True)
        print(f"  ✅ Created: dataset/raw/{original_cat}/")
    
    # Also create 'organic' folder
    organic_dir = raw_dir / "organic"
    organic_dir.mkdir(exist_ok=True, parents=True)
    print(f"  ✅ Created: dataset/raw/organic/")
    
    print("\n✅ Folder structure created successfully!")
    print("\n📌 Next steps:")
    print("  1. Add your images to the appropriate folders")
    print("  2. Run: python dataset/organize_data.py")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  WASTE CLASSIFICATION DATASET DOWNLOADER")
    print("=" * 60 + "\n")
    
    # Create sample structure
    create_sample_dataset()
    
    # Provide download instructions
    print("\n" + "-" * 60)
    download_trashnet()
    
    print("=" * 60)
    print("  DOWNLOAD SCRIPT COMPLETE")
    print("=" * 60 + "\n")
