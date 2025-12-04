"""
Test script to validate project structure and code syntax.
Run this to check if everything is set up correctly.
"""

import os
import sys
from pathlib import Path


def check_file_exists(filepath):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✅ {filepath}")
        return True
    else:
        print(f"❌ {filepath} - NOT FOUND")
        return False


def check_python_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, filepath, 'exec')
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {filepath}: {e}")
        return False


def test_project_structure():
    """Test if all required files exist."""
    print("=" * 60)
    print("TESTING PROJECT STRUCTURE")
    print("=" * 60)
    print()
    
    required_files = [
        'README.md',
        'requirements.txt',
        'LICENSE',
        '.gitignore',
        'PROJECT_PLAN.md',
        'PROJECT_STRUCTURE.md',
        'dataset/download_data.py',
        'dataset/organize_data.py',
        'dataset/augment.py',
        'dataset/split_data.py',
        'model/config.py',
        'model/model_architecture.py',
        'model/utils.py',
    ]
    
    all_exist = True
    for filepath in required_files:
        if not check_file_exists(filepath):
            all_exist = False
    
    print()
    if all_exist:
        print("✅ All required files exist!")
    else:
        print("❌ Some files are missing!")
    
    return all_exist


def test_python_syntax():
    """Test Python files for syntax errors."""
    print()
    print("=" * 60)
    print("TESTING PYTHON SYNTAX")
    print("=" * 60)
    print()
    
    python_files = [
        'dataset/download_data.py',
        'dataset/organize_data.py',
        'dataset/augment.py',
        'dataset/split_data.py',
        'model/config.py',
        'model/model_architecture.py',
        'model/utils.py',
    ]
    
    all_valid = True
    for filepath in python_files:
        if Path(filepath).exists():
            if check_python_syntax(filepath):
                print(f"✅ {filepath} - Valid syntax")
            else:
                all_valid = False
        else:
            print(f"⚠️  {filepath} - File not found")
            all_valid = False
    
    print()
    if all_valid:
        print("✅ All Python files have valid syntax!")
    else:
        print("❌ Some Python files have syntax errors!")
    
    return all_valid


def test_model_architecture():
    """Test if model architecture can be defined (without dependencies)."""
    print()
    print("=" * 60)
    print("TESTING MODEL ARCHITECTURE (Syntax Check)")
    print("=" * 60)
    print()
    
    try:
        # Just check if the file can be compiled
        with open('model/model_architecture.py', 'r') as f:
            code = f.read()
        
        # Check for key components
        checks = {
            'WasteClassifier class': 'class WasteClassifier' in code,
            'forward method': 'def forward' in code,
            'create_model function': 'def create_model' in code,
            'MobileNetV2 reference': 'mobilenet_v2' in code,
        }
        
        all_passed = True
        for check_name, passed in checks.items():
            if passed:
                print(f"✅ {check_name} found")
            else:
                print(f"❌ {check_name} missing")
                all_passed = False
        
        print()
        if all_passed:
            print("✅ Model architecture structure looks good!")
        else:
            print("❌ Some components are missing!")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error checking model architecture: {e}")
        return False


def test_config():
    """Test if config file is properly structured."""
    print()
    print("=" * 60)
    print("TESTING CONFIGURATION")
    print("=" * 60)
    print()
    
    try:
        with open('model/config.py', 'r') as f:
            code = f.read()
        
        configs = [
            'MODEL_CONFIG',
            'TRAIN_CONFIG',
            'DATA_CONFIG',
            'CLASS_NAMES',
        ]
        
        all_found = True
        for config in configs:
            if config in code:
                print(f"✅ {config} defined")
            else:
                print(f"❌ {config} missing")
                all_found = False
        
        print()
        if all_found:
            print("✅ Configuration looks complete!")
        else:
            print("❌ Some configurations are missing!")
        
        return all_found
        
    except Exception as e:
        print(f"❌ Error checking config: {e}")
        return False


def test_readme():
    """Test if README has essential sections."""
    print()
    print("=" * 60)
    print("TESTING DOCUMENTATION")
    print("=" * 60)
    print()
    
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        sections = [
            '## Overview',
            '## Installation',
            '## Quick Start',
            '## Usage',
            'requirements.txt',
        ]
        
        all_found = True
        for section in sections:
            if section.lower() in content.lower():
                print(f"✅ {section} section found")
            else:
                print(f"⚠️  {section} section might be missing")
        
        print(f"\n📊 README.md is {len(content)} characters long")
        print("✅ Documentation exists!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking README: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  PROJECT VALIDATION TEST SUITE")
    print("=" * 60 + "\n")
    
    results = {
        'Structure': test_project_structure(),
        'Syntax': test_python_syntax(),
        'Model': test_model_architecture(),
        'Config': test_config(),
        'Documentation': test_readme(),
    }
    
    print()
    print("=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    print()
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<20} {status}")
    
    print()
    all_passed = all(results.values())
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print()
        print("✅ Project structure is valid")
        print("✅ Python syntax is correct")
        print("✅ Ready for next steps")
        print()
        print("📌 Next steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Download dataset: python dataset/download_data.py")
        print("   3. Create training script: model/train.py")
        print("   4. Create webcam app: app/webcam_detect.py")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Please review the errors above.")
    
    print()
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
