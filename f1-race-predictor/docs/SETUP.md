# Setup Guide

## Prerequisites

**Required:**
- Modern web browser (Chrome 90+, Firefox 88+, Safari 14+)
- Text editor (VS Code, Sublime, Notepad++)

**Optional:**
- Python 3.8+ (for data processing)
- Git (for version control)
- Web server (for development)

## Installation Methods

### Method 1: Direct Download (Easiest)

1. Download the ZIP file
2. Extract to your desired location
3. Open `index.html` in your browser
4. Done!

### Method 2: Git Clone
```bash
git clone https://github.com/yourusername/f1-race-predictor.git
cd f1-race-predictor
open index.html
```

### Method 3: Local Web Server
```bash
# Python
python -m http.server 8000

# Node.js
npm install -g http-server
http-server -p 8000

# Access at: http://localhost:8000
```

## File Setup

### 1. Create Directory Structure
```bash
mkdir -p f1-race-predictor/{css,js,data,python,docs,assets/icons}
cd f1-race-predictor
```

### 2. Copy Files

Place all files in their respective directories according to the structure.

### 3. Verify Installation

Open `index.html` and check:
- [ ] Page loads without errors
- [ ] All styles applied correctly
- [ ] Sliders work
- [ ] Predictions update
- [ ] Presets load

## Data Processing

### Setup Python Environment
```bash
cd python
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download F1 Data

1. Go to [Kaggle F1 Dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
2. Download ZIP file
3. Extract CSV files to `data_raw/` directory

### Process Data
```bash
python process_data.py
# Creates JSON files in data/ directory

python train_model.py
# Validates model performance
```

## Configuration

### Edit Config File
```javascript
// js/config.js
const CONFIG = {
    MODEL_VERSION: '2.0',
    // Adjust weights
    WEIGHTS: {
        GRID_POSITION: 40,  // Your custom weight
        RECENT_FORM: 30,
    }
};
```

### Customize Styles
```css
/* css/styles.css */
:root {
    --blue: #your-color;
    --bg-primary: #your-bg;
}
```

## Troubleshooting

### Page Doesn't Load

**Solution:**
- Check browser console (F12)
- Verify all files are in correct directories
- Clear browser cache

### Sliders Don't Update

**Solution:**
- Check JavaScript console for errors
- Ensure all JS files are loaded
- Refresh page (Ctrl+R / Cmd+R)

### Styles Look Wrong

**Solution:**
- Verify `css/styles.css` is loaded
- Check for syntax errors
- Try different browser

### Python Scripts Fail

**Solution:**
- Check Python version (3.8+)
- Install dependencies: `pip install -r requirements.txt`
- Verify CSV files exist in `data_raw/`

## Development Setup

### VS Code

1. Install extensions:
   - Live Server
   - Prettier
   - ESLint

2. Open project folder
3. Start Live Server
4. Auto-reload on changes

### Git Setup
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin your-repo-url
git push -u origin main
```

## Deployment

### GitHub Pages
```bash
# Push to GitHub
git push origin main

# Enable Pages in Settings
# Select main branch
# Access at: username.github.io/f1-race-predictor
```

### Netlify

1. Drag and drop folder to Netlify
2. Or connect GitHub repository
3. Auto-deploy on push

### Vercel
```bash
npm install -g vercel
vercel
# Follow prompts
```

## Performance Optimization

### Minify Files
```bash
# CSS
cssnano styles.css > styles.min.css

# JavaScript
uglifyjs predictor.js -o predictor.min.js
```

### Compress Images
```bash
# Using imagemin
imagemin assets/* --out-dir=assets/compressed
```

### Enable Caching

Add to `.htaccess`:
```apache
<IfModule mod_expires.c>
    ExpiresActive On
    ExpiresByType text/css "access plus 1 year"
    ExpiresByType application/javascript "access plus 1 year"
</IfModule>
```

## Testing

### Browser Testing

Test on:
- [ ] Chrome (Windows, Mac, Linux)
- [ ] Firefox
- [ ] Safari (Mac, iOS)
- [ ] Edge
- [ ] Mobile browsers

### Functionality Testing

- [ ] All sliders move smoothly
- [ ] Predictions update instantly
- [ ] Presets load correctly
- [ ] Export/import works
- [ ] Responsive on mobile

### Performance Testing

- [ ] Page loads in <2 seconds
- [ ] No JavaScript errors
- [ ] Smooth animations
- [ ] Memory usage acceptable

## Maintenance

### Update Data
```bash
# Download new CSV files
# Run processing
python process_data.py

# Commit changes
git add data/
git commit -m "Update data to 2024"
git push
```

### Update Dependencies
```bash
# Python
pip list --outdated
pip install --upgrade package-name

# JavaScript (if using npm)
npm update
```

## Backup

### Regular Backups
```bash
# Create backup
tar -czf f1-predictor-backup-$(date +%Y%m%d).tar.gz f1-race-predictor/

# Or use Git
git commit -am "Backup $(date)"
git push
```

## Support

For issues:
1. Check [Troubleshooting](#troubleshooting)
2. Search [GitHub Issues](https://github.com/yourusername/f1-race-predictor/issues)
3. Create new issue with:
   - Browser version
   - Error message
   - Steps to reproduce

---

**Setup Time**: ~10 minutes  
**Difficulty**: Easy  
**Support**: Available via GitHub Issues