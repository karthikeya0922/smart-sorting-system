# Python Installation Guide for Windows

## Quick Installation Steps

### Option 1: Download from Official Website (Recommended)

1. **Download Python 3.11 or 3.10**:
   - Visit: https://www.python.org/downloads/
   - Click the yellow "Download Python 3.x.x" button
   - This will download `python-3.x.x-amd64.exe`

2. **Run the Installer**:
   - Double-click the downloaded `.exe` file
   - ⚠️ **IMPORTANT**: Check ✅ "Add python.exe to PATH" at the bottom!
   - Click "Install Now" (recommended) or "Customize installation"
   - Wait for installation to complete

3. **Verify Installation**:
   ```powershell
   # Open a NEW PowerShell/Command Prompt window
   python --version
   # Should show: Python 3.x.x
   
   pip --version
   # Should show: pip 23.x.x or similar
   ```

### Option 2: Using Windows Store (Alternative)

1. Open Microsoft Store
2. Search for "Python 3.11" or "Python 3.10"
3. Click "Get" or "Install"
4. Python will be automatically added to PATH

### Option 3: Using Chocolatey (For Advanced Users)

```powershell
# If you have Chocolatey installed:
choco install python --version=3.11.0
```

---

## After Installation

### 1. Verify Python is Working
```powershell
# Open a NEW terminal window (important!)
python --version
pip --version
```

### 2. Navigate to Your Project
```powershell
cd "C:\Users\Admin\Desktop\sorting system"
```

### 3. Create Virtual Environment
```powershell
python -m venv venv
```

### 4. Activate Virtual Environment
```powershell
.\venv\Scripts\activate
# You should see (venv) before your prompt
```

### 5. Install Project Dependencies
```powershell
pip install -r requirements.txt
```

This will install:
- PyTorch
- OpenCV
- Albumentations
- Matplotlib
- All other dependencies (~1-2 GB download)

### 6. Run Validation Test
```powershell
python test_project.py
```

---

## Troubleshooting

### Python command not found
- **Solution**: You need to restart your terminal after installation
- Or Python wasn't added to PATH during installation
- Reinstall Python and check "Add to PATH" ✅

### pip command not found
- **Solution**: Run `python -m pip --version` instead

### Permission errors during pip install
- **Solution**: Use virtual environment (step 3 above)
- Or run PowerShell as Administrator

### SSL certificate errors
- **Solution**: Update pip first:
  ```powershell
  python -m pip install --upgrade pip
  ```

---

## Recommended Python Version

**Best for this project**: Python 3.10.x or 3.11.x
- ✅ Python 3.10.11 (stable, recommended)
- ✅ Python 3.11.5 (newer, faster)
- ❌ Python 3.12+ (some packages may not support yet)
- ❌ Python 3.9 or older (too old)

---

## Next Steps After Installation

1. ✅ Install Python
2. ✅ Verify installation
3. ✅ Create virtual environment
4. ✅ Install dependencies
5. ✅ Run test_project.py
6. 📥 Download dataset
7. 🚀 Train model
8. 📹 Run webcam app

---

**Once installed, run this complete setup**:
```powershell
# 1. Navigate to project
cd "C:\Users\Admin\Desktop\sorting system"

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Test installation
python test_project.py
```

---

**Installation Time**: 5-10 minutes  
**Download Size**: ~50 MB (Python) + 1-2 GB (dependencies)
