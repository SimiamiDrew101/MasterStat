# MasterStat Quick Start Guide

Get MasterStat up and running in under 5 minutes!

## Option 1: Download Pre-Built Installer (Easiest!)

Download ready-to-use installers for your platform:

| Platform | Download |
|----------|----------|
| **macOS (Apple Silicon)** | `MasterStat-1.0.0-arm64.dmg` |
| **macOS (Intel)** | `MasterStat-1.0.0.dmg` |
| **Windows** | `MasterStat Setup 1.0.0.exe` |
| **Linux** | `MasterStat-1.0.0-arm64.AppImage` |

**Requirements:** Python 3.11+ must be installed on your system.

---

## Option 2: Run from Source

### Prerequisites
- Node.js v18+ ([Download](https://nodejs.org/))
- Python 3.11+ ([Download](https://python.org/downloads/))

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SimiamiDrew101/MasterStat.git
   cd MasterStat
   ```

2. **Install dependencies:**
   ```bash
   npm install
   cd backend && pip install -r requirements.txt && cd ..
   cd frontend && npm install && cd ..
   ```

3. **Launch the Electron app:**
   ```bash
   npm run electron
   ```

4. **That's it!** The app opens in a native window with the backend automatically started.

---

## Option 3: Development Mode

Run backend and frontend separately for development work with hot-reload.

### Terminal 1: Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

### Terminal 2: Frontend
```bash
cd frontend
npm install
npm run dev
```

### Access Points
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

---

## Try Your First Analysis

### Example 1: Response Surface Optimization

1. Navigate to **Response Surface** from the home page
2. Choose **Central Composite Design** (CCD)
3. Enter 3 factors with names and ranges
4. Click **Generate Design** to create the experiment matrix
5. Enter response data (or use sample data)
6. Click **Fit Model** to analyze
7. View 3D surface plots and contour plots
8. Run **Desirability Optimization** to find optimal settings

### Example 2: Reliability Analysis

1. Navigate to **Reliability Analysis**
2. Enter time-to-event data with censoring indicators (1=event, 0=censored)
3. Click **Fit Life Distribution** to compare Weibull, lognormal, exponential
4. Generate **Kaplan-Meier** survival curves
5. Add covariates and run **Cox Regression** for hazard ratios

### Example 3: Predictive Modeling

1. Navigate to **Predictive Modeling**
2. Load or paste your dataset
3. Select response variable and predictors
4. Click **Model Comparison** to run all methods:
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - Lasso/Ridge/Elastic Net
5. Compare R-squared, RMSE, MAE across models
6. View variable importance plots

### Example 4: Graph Builder

1. Navigate to **Graph Builder**
2. Load or paste data
3. Click a column, then click a zone (X, Y, Color, Size) to assign
4. Select chart type: Scatter, Bar, Box, Histogram, Heatmap
5. Customize and export your visualization

### Example 5: Linked Data Explorer

1. Navigate to **Linked Explorer**
2. Load your dataset
3. Select X and Y variables for scatter plot
4. **Brush select** points by clicking and dragging
5. Watch selection sync across histogram, box plot, and data table
6. View selection statistics

---

## Features Overview

MasterStat provides 24 statistical modules:

| Category | Modules |
|----------|---------|
| **DOE** | Experiment Wizard, RSM (CCD, Box-Behnken), Factorial, Custom Design, Mixture, Block Designs, Bayesian DOE, Optimal Designs |
| **Analysis** | ANOVA, Mixed Models, GLM, Nonlinear Regression, Hypothesis Testing |
| **Reliability** | Life Distribution, Kaplan-Meier, Cox PH, Accelerated Life Testing |
| **Quality** | Control Charts, Process Capability, Gauge R&R |
| **Predictive** | Decision Trees, Random Forest, Gradient Boosting, Regularized Regression |
| **Visualization** | Graph Builder, Linked Explorer, Prediction Profiler |

---

## Troubleshooting

### App Won't Start

```bash
# Check Python version
python3 --version  # Should be 3.11+

# Kill any existing processes
pkill -f electron
pkill -f uvicorn

# Reinstall dependencies
npm install
cd backend && pip install -r requirements.txt && cd ..

# Try again
npm run electron
```

### Port Already in Use

```bash
# Check what's using the ports
lsof -i :5173
lsof -i :8000

# Kill processes
kill -9 $(lsof -ti:5173)
kill -9 $(lsof -ti:8000)
```

### Backend Errors

```bash
# Check if all Python packages are installed
cd backend
pip install -r requirements.txt

# Test backend directly
python -m uvicorn app.main:app --reload --port 8000

# Check http://localhost:8000/health
```

---

## Building Desktop Apps

To build installers for distribution:

```bash
# Build for current platform
npm run dist

# Platform-specific
npm run build:mac      # macOS DMG + ZIP
npm run build:win      # Windows NSIS + Portable
npm run build:linux    # Linux AppImage + deb
```

Output: `dist-electron/`

---

## Next Steps

- Explore all 24 statistical modules from the home page
- Check API documentation at http://localhost:8000/docs
- Save your work using the **Sessions** button in the header
- Export results to PDF, Excel, or CSV
- Support the project at [Ko-fi](https://ko-fi.com/MasterStat)

---

## Tips

- **Sessions persist:** Your analysis sessions are saved in the browser and can be exported/imported
- **Dark mode:** The entire app uses a modern dark theme
- **Keyboard shortcuts:** Use Tab to navigate, Enter to submit
- **Export options:** Most results can be exported to PDF, Excel, CSV, or PNG
- **API access:** All features are available via REST API at http://localhost:8000/docs

---

**Need help?** Open an issue on [GitHub](https://github.com/SimiamiDrew101/MasterStat/issues)
