# Bombus Detection Project .gitignore

# Large data files
*.mp4
*.avi
*.mov
*.mkv
data/raw/
*.pth
*.pkl

# Processed images (too many small files)
data/processed/frames/positive/
data/processed/frames/negative/

# Keep directory structure but ignore contents
data/processed/frames/positive/.gitkeep
data/processed/frames/negative/.gitkeep

# Excel files with potentially sensitive data
Collection\ Observations*.xlsx

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Jupyter Notebook
.ipynb_checkpoints

# PyTorch
*.pth
*.pt

# Matplotlib
*.png
*.pdf
*.svg
!README_images/*.png

# Analysis outputs
analysis_output/
analysis_results/
training_history.png
training_results.json

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# macOS
.DS_Store
.AppleDouble
.LSOverride
Icon
._*
.DocumentRevisions-V100
.fseventsd
.Spotlight-V100
.TemporaryItems
.Trashes
.VolumeIcon.icns
.com.apple.timemachine.donotpresent
.AppleDB
.AppleDesktop
Network Trash Folder
Temporary Items
.apdisk

# Windows
Thumbs.db
ehthumbs.db
Desktop.ini
$RECYCLE.BIN/
*.cab
*.msi
*.msix
*.msm
*.msp
*.lnk

# Logs
*.log
logs/

# Temporary files
tmp/
temp/