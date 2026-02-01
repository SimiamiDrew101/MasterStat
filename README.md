# MasterStat

**Professional-grade statistical analysis and Design of Experiments (DOE) platform**

Free, open-source desktop application for researchers, engineers, and data scientists. Comprehensive statistical analysis suite built with React, FastAPI, Electron, and modern visualization libraries.

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Ko-fi](https://img.shields.io/badge/Support-Ko--fi-FF5E5B?logo=ko-fi&logoColor=white)](https://ko-fi.com/MasterStat)

[Features](#features) | [Download](#download) | [Quick Start](#quick-start) | [Documentation](#documentation) | [Contributing](#contributing)

---

## Download Desktop App

**Pre-built installers for all platforms (v1.0.0):**

| Platform | Download | Notes |
|----------|----------|-------|
| **macOS (Apple Silicon)** | `MasterStat-1.0.0-arm64.dmg` | M1/M2/M3 Macs |
| **macOS (Intel)** | `MasterStat-1.0.0.dmg` | Intel Macs |
| **Windows** | `MasterStat Setup 1.0.0.exe` | Windows 10/11 (64-bit) |
| **Windows Portable** | `MasterStat 1.0.0.exe` | No installation required |
| **Linux** | `MasterStat-1.0.0-arm64.AppImage` | ARM64 systems |
| **Linux (Debian)** | `masterstat_1.0.0_arm64.deb` | Debian/Ubuntu ARM64 |

> **Note:** Requires Python 3.11+ installed on your system for the statistical backend.

---

## Features

### Design of Experiments (DOE)

| Module | Description |
|--------|-------------|
| **Experiment Wizard** | Step-by-step guided design with DSD, Plackett-Burman, confounding analysis |
| **Response Surface (RSM)** | CCD, Box-Behnken, optimization, 3D surfaces, contour plots |
| **Factorial Designs** | Full 2^k, 3^k, fractional factorial, screening designs |
| **Custom Design** | D-optimal, I-optimal, A-optimal designs with constraints |
| **Mixture Designs** | Simplex-lattice, simplex-centroid, extreme vertices, ternary plots |
| **Block Designs** | RCBD, Latin squares, BIBD, incomplete blocks |
| **Bayesian DOE** | MCMC, posterior optimization, HDI, convergence diagnostics |
| **Optimal Designs** | Space-filling, coordinate exchange algorithms |

### Statistical Analysis

| Module | Description |
|--------|-------------|
| **ANOVA** | One-way, two-way, repeated measures with Tukey, Bonferroni, Scheffe |
| **Mixed Models** | Split-plot, nested designs, random effects, variance components, BLUPs |
| **GLM** | Poisson, Binomial, Gamma, Negative Binomial regression |
| **Nonlinear Regression** | Curve fitting, growth models, convergence analysis |
| **Hypothesis Testing** | t-tests, F-tests, Z-tests with confidence intervals |
| **Model Validation** | PRESS statistic, k-fold CV, adequacy scoring, diagnostic tests |

### Reliability & Survival Analysis

| Feature | Description |
|---------|-------------|
| **Life Distribution** | Weibull, lognormal, exponential distribution fitting |
| **Kaplan-Meier** | Non-parametric survival curves, log-rank tests |
| **Cox Proportional Hazards** | Semi-parametric regression for censored data |
| **Accelerated Life Testing** | Arrhenius, power law models for stress testing |
| **Reliability Test Planning** | Sample size calculations for demonstration tests |

### Quality Control & MSA

| Feature | Description |
|---------|-------------|
| **Control Charts** | X-bar, R, S, P, C, CUSUM, EWMA charts |
| **Process Capability** | Cp, Cpk, Pp, Ppk with confidence intervals |
| **Gauge R&R** | Crossed and nested MSA studies |
| **Attribute Agreement** | Kappa, Kendall's W for categorical data |
| **Western Electric Rules** | Automatic out-of-control detection |

### Predictive Modeling

| Method | Purpose |
|--------|---------|
| **Decision Trees (CART)** | Interpretable segmentation |
| **Random Forest** | Variable importance, robust prediction |
| **Gradient Boosting** | High-accuracy prediction |
| **Regularized Regression** | Lasso, Ridge, Elastic Net for variable selection |
| **Model Comparison** | Automated comparison across all methods |

### Interactive Visualization

| Feature | Description |
|---------|-------------|
| **Graph Builder** | Drag-and-drop chart creation (scatter, bar, box, histogram, heatmap) |
| **Linked Explorer** | Linked plots with brushing, selection sync across views |
| **3D Surfaces** | Interactive response surfaces with rotation and zoom |
| **Contour Plots** | Overlay contours for multi-response optimization |
| **Ternary Plots** | Triangular plots for mixture designs |
| **Diagnostic Plots** | Residuals, Q-Q, Cook's distance, leverage |

### Multi-Response Optimization

| Feature | Description |
|---------|-------------|
| **Desirability Functions** | Geometric mean, minimum, weighted sum methods |
| **Overlay Contours** | Find feasible regions for multiple responses |
| **Optimization Profiler** | Interactive factor profiling and sensitivity analysis |

### Data Management

- **Excel-like Table Interface** - Edit data inline with copy/paste support
- **Missing Data Imputation** - Mean, median, mode, interpolation, MICE
- **Outlier Detection** - IQR method, Z-score method, visual diagnostics
- **Data Transformation** - Log, square root, Box-Cox, standardization
- **Session Persistence** - IndexedDB storage, export/import sessions

### Export Capabilities

- **PDF** - Publication-ready reports with embedded figures
- **Excel** - Multi-sheet workbooks with formatted tables
- **CSV/TSV** - Standard data formats
- **JSON** - Session export/import for reproducibility
- **PNG/SVG** - High-resolution figures (300-1200 DPI)

---

## Quick Start

### Option 1: Download Desktop App (Recommended)

1. Download the installer for your platform from the releases
2. Install and launch MasterStat
3. The app automatically starts the Python backend

**Requirements:** Python 3.11+ must be installed ([Download](https://python.org/downloads/))

### Option 2: Run from Source

```bash
# Clone repository
git clone https://github.com/SimiamiDrew101/MasterStat.git
cd MasterStat

# Install dependencies
npm install
cd backend && pip install -r requirements.txt && cd ..
cd frontend && npm install && cd ..

# Launch Electron app
npm run electron
```

### Option 3: Development Mode

```bash
# Terminal 1: Backend
cd backend
python -m uvicorn app.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

**Access points:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

---

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React 18.2, Vite 5.0, TailwindCSS 3.4, Plotly.js 3.3, Recharts |
| **Backend** | Python 3.11+, FastAPI 0.109, Uvicorn, SciPy, statsmodels, pandas, scikit-learn, lifelines |
| **Desktop** | Electron 28.1, cross-platform (macOS, Windows, Linux) |
| **Persistence** | IndexedDB via Dexie.js 4.2 (client-side session storage) |
| **Visualization** | Plotly.js (3D surfaces, contours, ternary), Recharts (charts) |

---

## Architecture

```
MasterStat/
├── backend/                 # Python/FastAPI statistical engine
│   ├── app/
│   │   ├── api/            # 24 API modules
│   │   ├── utils/          # model_validation.py, report_generator.py
│   │   └── main.py         # FastAPI entry point
│   └── requirements.txt
├── frontend/                # React/Vite web interface
│   ├── src/
│   │   ├── pages/          # 24 page components
│   │   ├── components/     # 90+ reusable components
│   │   ├── contexts/       # SessionContext.jsx, SelectionContext.jsx
│   │   └── utils/          # sessionManager.js, validation, etc.
│   └── package.json
├── electron/                # Electron desktop wrapper
│   ├── main.js             # Main process
│   └── preload.js          # Security layer
└── package.json            # Root config (Electron builds)
```

**Project Statistics:**
- 24 statistical API modules
- 24 page components
- 90+ React components
- 50+ visualization types
- ~93,000 lines of code
- Cross-platform desktop builds

---

## Example Workflows

### 1. Response Surface Optimization

1. Navigate to **Response Surface** -> Generate CCD or Box-Behnken design
2. Import experimental data -> Fit quadratic model
3. View 3D surface and contour plots
4. **Validate model** with PRESS and k-fold CV
5. Run desirability optimization -> Export report

### 2. Reliability Analysis

1. Navigate to **Reliability Analysis**
2. Enter time-to-event data with censoring indicators
3. Fit life distributions (Weibull, lognormal, exponential)
4. Generate Kaplan-Meier survival curves
5. Run Cox regression with covariates -> View hazard ratios

### 3. Predictive Modeling Comparison

1. Navigate to **Predictive Modeling**
2. Load dataset with response variable
3. Run **Model Comparison** across all methods
4. Compare R-squared, RMSE, MAE across models
5. View variable importance from Random Forest

### 4. Mixture Design with Ternary Plots

1. Navigate to **Mixture Design**
2. Define 3+ components with constraints
3. Generate extreme vertices design
4. Fit Scheffe model to response data
5. View **ternary contour plot** with optimal region

### 5. Graph Builder

1. Navigate to **Graph Builder**
2. Load or paste data
3. Drag columns to X, Y, Color zones (or click to assign)
4. Select chart type (scatter, bar, box, histogram, heatmap)
5. Customize appearance and export

### 6. Linked Data Exploration

1. Navigate to **Linked Explorer**
2. Load dataset
3. Select variables for scatter plot
4. **Brush select** points in scatter -> See highlighting in histogram and box plot
5. View selection statistics and filtered data table

---

## Building Desktop Apps

```bash
# Build for current platform
npm run dist

# Platform-specific builds
npm run build:mac      # macOS (DMG + ZIP, arm64 + x64)
npm run build:win      # Windows (NSIS installer + Portable)
npm run build:linux    # Linux (AppImage + deb)
```

Output location: `dist-electron/`

---

## API Reference

### Design Generation Endpoints
```
POST /api/rsm/ccd/generate
POST /api/rsm/box-behnken/generate
POST /api/rsm/dsd/generate
POST /api/factorial/generate
POST /api/custom-design/generate
POST /api/mixture/simplex-lattice
POST /api/mixture/extreme-vertices/generate
```

### Analysis Endpoints
```
POST /api/anova/analyze
POST /api/rsm/fit-model
POST /api/glm/fit
POST /api/reliability/kaplan-meier
POST /api/reliability/cox-ph
POST /api/predictive-modeling/model-comparison
```

### Quality Control Endpoints
```
POST /api/quality-control/control-chart
POST /api/quality-control/capability
POST /api/msa/gauge-rr
```

Full API documentation available at http://localhost:8000/docs when running the backend.

---

## Contributing

Contributions welcome! See our guidelines:

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/MasterStat.git

# Create feature branch
git checkout -b feature/your-feature

# Make changes, test locally
npm run dev  # or npm run electron

# Commit and push
git commit -m "feat: Description of feature"
git push origin feature/your-feature

# Open Pull Request
```

**Code Guidelines:**
- React functional components with hooks
- TailwindCSS dark mode (slate-800, slate-700 backgrounds)
- FastAPI endpoints with Pydantic models
- Plotly dark theme (paper_bgcolor: #1e293b, plot_bgcolor: #0f172a)

---

## Support

MasterStat is free and open-source. Support the project:

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/MasterStat)

**Other ways to help:**
- Star this repository
- Report bugs via [GitHub Issues](https://github.com/SimiamiDrew101/MasterStat/issues)
- Share with colleagues and students
- Contribute code or documentation

---

## License

**Creative Commons Attribution 4.0 International (CC BY 4.0)**

You are free to share and adapt for any purpose, even commercially, with attribution.

See [LICENSE](LICENSE) for details.

**Citation:**
```
MasterStat: Professional statistical analysis and Design of Experiments platform
https://github.com/SimiamiDrew101/MasterStat
```

---

## Acknowledgments

Built with:
- **Frontend:** React, Vite, Plotly.js, TailwindCSS, Lucide Icons, Dexie.js
- **Backend:** FastAPI, SciPy, statsmodels, pandas, NumPy, scikit-learn, lifelines, ReportLab
- **Desktop:** Electron, electron-builder
- **Development:** Claude Code

---

<div align="center">

**Made with care for the research and engineering community**

[Back to Top](#masterstat)

</div>
