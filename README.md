# MasterStat

**Professional-grade statistical analysis and Design of Experiments (DOE) platform**

Free, open-source desktop application for researchers, engineers, and data scientists. Built with React, FastAPI, Electron, and modern visualization libraries.

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Ko-fi](https://img.shields.io/badge/Support-Ko--fi-FF5E5B?logo=ko-fi&logoColor=white)](https://ko-fi.com/MasterStat)

[Features](#features) • [Download](#download) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Contributing](#contributing)

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

### Core Statistical Analysis

| Module | Description |
|--------|-------------|
| **Experiment Wizard** | Step-by-step guided design with DSD, Plackett-Burman, confounding analysis |
| **Response Surface (RSM)** | CCD, Box-Behnken, optimization, 3D surfaces, contour plots |
| **ANOVA** | One-way, two-way, repeated measures with Tukey, Bonferroni, Scheffé |
| **Factorial Designs** | Full 2^k, 3^k, fractional factorial, screening designs |
| **Mixed Models** | Split-plot, nested designs, random effects, variance components |
| **Quality Control** | X-bar, R, S, P, C charts, process capability (Cp, Cpk) |
| **Nonlinear Regression** | Curve fitting, growth models, convergence analysis |
| **Bayesian DOE** | MCMC, posterior optimization, HDI, convergence diagnostics |

### Advanced Features (Tier 2 - NEW)

| Feature | Description |
|---------|-------------|
| **Model Validation** | PRESS statistic, k-fold cross-validation, adequacy scoring (0-100), diagnostic tests |
| **Multi-Response Optimization** | 3 desirability methods (geometric mean, minimum, weighted sum), overlay contour plots |
| **Session Management** | Save/load analysis sessions, export/import JSON, search and filter history |
| **Design Preview** | Interactive 3D visualization of experimental designs with power curves |

### Data Management

- **Excel-like Table Interface** - Edit data inline with copy/paste support
- **Missing Data Imputation** - Mean, median, mode, interpolation, MICE
- **Outlier Detection** - IQR method, Z-score method, visual diagnostics
- **Data Transformation** - Log, square root, Box-Cox, standardization
- **Session Persistence** - IndexedDB storage, export/import sessions

### Visualizations

- Interactive 3D response surfaces with rotation and zoom
- Contour plots with optimization paths and constraints
- **Overlay contour plots** for multi-response optimization
- Diagnostic plots (residuals, Q-Q, Cook's distance, leverage)
- Main effects and interaction plots
- Power curves for sample size planning
- Confounding diagrams with resolution badges

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
| **Backend** | Python 3.11+, FastAPI 0.109, Uvicorn, SciPy, statsmodels, pandas |
| **Desktop** | Electron 28.1, cross-platform (macOS, Windows, Linux) |
| **Persistence** | IndexedDB via Dexie.js 4.2 (client-side session storage) |
| **Visualization** | Plotly.js (3D surfaces, contours), Recharts (charts) |

---

## Architecture

```
MasterStat/
├── backend/                 # Python/FastAPI statistical engine
│   ├── app/
│   │   ├── api/            # 14 API modules (rsm, anova, factorial, etc.)
│   │   ├── utils/          # model_validation.py, report_generator.py
│   │   └── main.py         # FastAPI entry point
│   └── requirements.txt
├── frontend/                # React/Vite web interface
│   ├── src/
│   │   ├── pages/          # 18 page components
│   │   ├── components/     # 90+ reusable components
│   │   ├── contexts/       # SessionContext.jsx
│   │   └── utils/          # sessionManager.js, validation, etc.
│   └── package.json
├── electron/                # Electron desktop wrapper
│   ├── main.js             # Main process
│   └── preload.js          # Security layer
└── package.json            # Root config (Electron builds)
```

**Project Statistics:**
- 18 statistical modules
- 90+ React components
- 14 backend API modules
- 50+ visualization types
- ~35,000 lines of code
- Cross-platform desktop builds

---

## Example Workflows

### 1. Response Surface Optimization

1. Navigate to **Response Surface** → Generate CCD or Box-Behnken design
2. Import experimental data → Fit quadratic model
3. View 3D surface and contour plots
4. **Validate model** with PRESS and k-fold CV
5. Run desirability optimization → Export report

### 2. Multi-Response Optimization (NEW)

1. Fit models for multiple responses (Yield, Purity, Cost)
2. Add each to desirability specifications
3. Set goals (maximize, minimize, target) and weights
4. Choose compositing method (geometric mean recommended)
5. View **overlay contour plot** to find sweet spot

### 3. Session Management (NEW)

1. Complete any analysis
2. Click **Save Session** → Enter name
3. Access saved sessions via **Sessions** button in header
4. Load, rename, delete, or export sessions as JSON

### 4. Model Validation (NEW)

1. Fit a model (ANOVA, Factorial, RSM, etc.)
2. Click **Validate Model** button
3. Review adequacy score (0-100)
4. Check PRESS statistic and k-fold CV results
5. Review diagnostic tests (normality, homoscedasticity)

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
- **Backend:** FastAPI, SciPy, statsmodels, pandas, NumPy, ReportLab
- **Desktop:** Electron, electron-builder
- **Development:** Claude Code

---

<div align="center">

**Made with care for the research and engineering community**

[Back to Top](#masterstat)

</div>
