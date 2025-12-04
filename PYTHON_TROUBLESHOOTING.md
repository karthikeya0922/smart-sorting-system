# Python Installation Troubleshooting

## ❌ Issue Detected: Python Not in PATH

Your Python is installed but not accessible from the command line.

## 🔧 Solution Options

### Option 1: Use the Python Launcher (Quick Fix)

Instead of `python`, use `py`:

```powershell
# Check version
py --version

# Run scripts
py test_project.py

# Use pip
py -m pip install -r requirements.txt
```

### Option 2: Add Python to PATH (Permanent Fix)

#### Method A: Reinstall Python with PATH
1. Uninstall current Python (Control Panel → Programs → Uninstall)
2. Download Python again from python.org
3. **IMPORTANT**: Check ✅ "Add Python to PATH" during installation
4. Complete installation
5. Restart terminal

#### Method B: Manually Add to PATH
1. Find Python installation location (usually):
   - `C:\Users\Admin\AppData\Local\Programs\Python\Python311\`
   - `C:\Python311\`
   - `C:\Users\Admin\AppData\Local\Microsoft\WindowsApps\`

2. Add to PATH:
   - Press `Windows + X` → System
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - Under "User variables", select "Path" → Edit
   - Click "New" and add Python directory
   - Click "New" again and add Scripts directory
   - Click OK on all windows
   - **Restart your terminal**

### Option 3: Use Full Path (Temporary)

Find where Python is installed and use full path:
```powershell
# Example (adjust path as needed):
C:\Users\Admin\AppData\Local\Programs\Python\Python311\python.exe --version
```

## ✅ Quick Test Commands

Try these in order until one works:

```powershell
# Try 1: Python launcher
py --version

# Try 2: Direct python3
python3 --version

# Try 3: Check Windows Apps
where py

# Try 4: Check if in AppData
dir $env:LOCALAPPDATA\Programs\Python
```

## 🚀 Next Steps (Once Python Works)

```powershell
# 1. Verify Python
py --version

# 2. Verify pip
py -m pip --version

# 3. Navigate to project
cd "C:\Users\Admin\Desktop\sorting system"

# 4. Create virtual environment
py -m venv venv

# 5. Activate virtual environment
.\venv\Scripts\activate

# 6. Install dependencies
pip install -r requirements.txt

# 7. Run validation
python test_project.py
```

## 📌 Most Likely Solution

**For most users, this works immediately**:
```powershell
# Use 'py' instead of 'python'
py --version
py test_project.py
py -m pip install -r requirements.txt
```

---

**Let me know which command works and I'll guide you through the next steps!**
