# MasterStat

**Professional-grade statistical analysis and Design of Experiments (DOE) platform**

Free, open-source web application for researchers, engineers, and data scientists. Built with React, FastAPI, and modern visualization libraries.

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Ko-fi](https://img.shields.io/badge/Support-Ko--fi-FF5E5B?logo=ko-fi&logoColor=white)](https://ko-fi.com/MasterStat)

[Features](#features) • [Quick Start](#quick-start) • [Installation](#installation) • [Documentation](#documentation) • [Contributing](#contributing)

---

## Features

### Core Statistical Analysis Modules

- **Experiment Wizard** - Step-by-step guided experimental design with goal-based recommendations
- **Data Preprocessing** - Transform, clean, detect outliers, handle missing values with multiple imputation methods
- **Protocol Generator** - Create randomized, blinded experimental protocols with PDF export
- **Experiment Planning** - Sample size calculation, power analysis, effect size estimation
- **Hypothesis Testing** - t-tests, F-tests, Z-tests, Mann-Whitney, Kruskal-Wallis with confidence intervals
- **ANOVA** - One-way, two-way, repeated measures with post-hoc tests (Tukey, Bonferroni, Scheffé)
- **Factorial Designs** - Full 2^k, 3^k, fractional factorial, Plackett-Burman, screening designs
- **Block Designs** - RCBD, Latin squares, Graeco-Latin squares, incomplete blocks
- **Mixed Models** - Split-plot, nested designs, repeated measures, random effects, variance components
- **Response Surface Methodology** - Central Composite Design, Box-Behnken, optimization, steepest ascent
- **Mixture Design** - Simplex-centroid, simplex-lattice, constrained mixture regions
- **Robust Design** - Taguchi methods, parameter design, noise factors, signal-to-noise ratios
- **Bayesian DOE** - MCMC parameter estimation, sequential designs, posterior optimization, model comparison

### Advanced Bayesian Features (NEW)

- **Highest Density Intervals (HDI)** - More accurate 95% credible intervals than percentiles
- **Convergence Diagnostics** - Effective Sample Size (ESS), R-hat, autocorrelation analysis
- **Posterior Visualization** - Comprehensive plots with prior overlay, trace plots, ACF diagnostics
- **Model Comparison** - Automatic BIC/AIC/Bayes factor comparison with best model selection
- **Prior Specification** - One-click presets (Weakly Informative / Uninformative)

### Data Preprocessing & Quality

- **Missing Data Imputation** - Mean, median, mode, forward fill, backward fill, linear interpolation
- **Outlier Detection** - IQR method, Z-score method, visual diagnostics
- **Data Transformation** - Log, square root, Box-Cox, standardization, normalization
- **Excel-like Table Interface** - Edit data inline with copy/paste support

### Protocol Generation

- **Randomization Methods** - Complete (CRD), Block (RBD), Restricted randomization
- **Seed-based Reproducibility** - Identical results with same seed for audit trails
- **Blinding System** - Single, double, triple-blind with separate confidential key
- **PDF Export** - Publication-ready protocols with all experimental details
- **Comprehensive Sections** - Objective, materials, procedure, safety, quality controls

### Visualizations

- Interactive 3D response surfaces with rotation and zoom
- Contour plots with optimization paths and constraints
- Diagnostic plots (residuals, Q-Q, Cook's distance, leverage)
- Main effects and interaction plots
- Cube plots for factorial designs
- Half-normal plots for effect screening
- Posterior density plots with prior overlay (Bayesian)
- MCMC trace plots with running mean (Bayesian)
- Autocorrelation plots with significance bounds (Bayesian)

### Export Capabilities

- **PDF** - Publication-ready reports with embedded figures and statistical tables
- **Excel** - Multi-sheet workbooks with formatted tables
- **CSV/TSV** - Standard data formats for further analysis
- **JMP/Minitab** - Industry-standard DOE formats
- **PNG/SVG** - High-resolution figures (300-1200 DPI, vector graphics)

---

## Quick Start

**See [QUICKSTART.md](QUICKSTART.md) for detailed instructions**

### Docker (Recommended - Full Features)

```bash
git clone https://github.com/SimiamiDrew101/MasterStat.git
cd MasterStat
./start.sh
```

Open http://localhost:5173 in your browser.

### Local Development

```bash
# Backend (Python 3.11+)
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend (Node.js 16+)
cd frontend
npm install
npm run dev
```

**Access points:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

---

## Installation

### Prerequisites

**For Docker (recommended):**
- [Docker Desktop](https://www.docker.com/products/docker-desktop)

**For local development:**
- Python 3.11+ ([Download](https://www.python.org/downloads/))
- Node.js v16+ ([Download](https://nodejs.org/))

### Docker Setup

Docker provides the complete application with both frontend and backend:

```bash
# Clone repository
git clone https://github.com/SimiamiDrew101/MasterStat.git
cd MasterStat

# Start application
./start.sh

# Or use docker compose directly
docker compose up --build

# Stop application
# Press Ctrl+C, or run:
docker compose down
```

**Access points:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

**Troubleshooting:**

```bash
# Port conflicts
lsof -ti:5173
lsof -ti:8000

# Clean rebuild
docker compose down -v
docker compose build --no-cache
docker compose up
```

### Local Development Setup

**Backend:**

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev
```

**Production build:**

```bash
npm run build
npm run preview
```

---

## Documentation

### Example Workflows

**1. Factorial Design & Analysis**

Navigate to **Experiment Wizard** → Select goal (Screening/Optimization) → Enter factors → Generate design → Export to Excel/CSV

**2. Response Surface Optimization**

Go to **Response Surface** → Upload data → Fit quadratic model → View 3D surface and contours → Perform optimization → Export report

**3. ANOVA with Post-hoc Tests**

Open **ANOVA** → Choose design (One-way/Two-way) → Input data → Run analysis → View F-test results → Perform post-hoc comparisons (Tukey HSD)

**4. Bayesian DOE with Convergence Diagnostics**

Go to **Bayesian DOE** → Generate factorial design → Click "Weakly Informative" preset → Run analysis → Review posterior distributions, trace plots, ESS diagnostics → Compare models

**5. Experimental Protocol Generation**

Navigate to **Protocol Generator** → Enter study details → Configure randomization → Set up blinding → Export PDF protocol + confidential blinding key

**6. Data Preprocessing**

Open **Data Preprocessing** → Import data → Detect outliers (IQR/Z-score) → Handle missing values (imputation) → Transform data (log, Box-Cox) → Export clean data

### Educational Features

Built-in DOE glossary with 50+ terms including:
- Aliasing and confounding patterns
- Blocking strategies
- Central Composite Design structures
- Effect hierarchy principles
- Interaction interpretation
- Power and sample size
- Resolution in fractional factorials
- Bayesian credible intervals
- Effective Sample Size (ESS)

Hover over any statistical term for context-sensitive help with examples.

---

## Technology Stack

**Frontend:**
- React 18 with Vite 5.4
- Plotly.js for interactive visualizations
- TailwindCSS for styling
- Axios for API communication
- jsPDF for PDF generation
- xlsx for Excel workbooks

**Backend:**
- FastAPI (Python 3.11)
- scipy, statsmodels for statistical computing
- pandas, numpy for data manipulation
- ReportLab for PDF protocol generation

**Statistical Methods:**
- MCMC (Metropolis-Hastings) for Bayesian inference
- Multiple imputation for missing data
- Box-Cox transformation for normality
- Robust outlier detection (IQR, Z-score)

---

## Architecture

**Component Organization:**

```
frontend/src/
├── pages/              # 14 analysis modules
├── components/         # 70+ reusable UI components
├── utils/              # Statistical utilities, validation
└── services/           # API client

backend/app/
├── api/               # 13 FastAPI endpoint modules
├── models/            # Statistical models
└── utils/             # Report generation, helpers
```

**Project Statistics:**
- 14 statistical modules
- 70+ React components
- 40+ visualization types
- 10+ export formats
- ~25,000 lines of code
- Publication-quality output
- Modern, well-maintained dependencies

---

## Contributing

Contributions are welcome! Here's how to help:

**Ways to Contribute:**
- Report bugs via GitHub Issues
- Suggest features or improvements
- Improve documentation
- Submit pull requests
- Help test new features
- Share your use cases

**Development Workflow:**

```bash
# Fork repository on GitHub
git clone https://github.com/YOUR_USERNAME/MasterStat.git
cd MasterStat

# Install dependencies
cd frontend && npm install
cd ../backend && pip install -r requirements.txt

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
# Frontend: npm run dev
# Backend: uvicorn app.main:app --reload

# Commit and push
git commit -m "Add: Brief description"
git push origin feature/your-feature-name

# Open Pull Request on GitHub
```

**Code Guidelines:**
- Use functional React components with hooks
- Follow TailwindCSS utility-first approach
- Write descriptive variable names
- Add comments for complex statistical logic
- Include JSDoc for component props
- Maintain dark theme consistency
- Test with sample data

---

## Support

MasterStat is free and open-source software. Support the project:

**Financial Support:**

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/MasterStat)

Your contributions help us develop new features, improve visualizations, and keep the project free for everyone.

**Other Ways to Help:**
- Star this repository ⭐
- Share with colleagues and students
- Write tutorials or blog posts
- Cite MasterStat in publications
- Report bugs and suggest improvements
- Contribute code or documentation

---

## License

**Creative Commons Attribution 4.0 International (CC BY 4.0)**

You are free to:
- **Share** - copy and redistribute the material in any medium or format
- **Adapt** - remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- **Attribution** - You must give appropriate credit to MasterStat, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

See [LICENSE](LICENSE) file for full details or visit https://creativecommons.org/licenses/by/4.0/

**How to Cite:**

```
MasterStat: Professional-grade statistical analysis and Design of Experiments platform
https://github.com/SimiamiDrew101/MasterStat
```

---

## Acknowledgments

Built with excellent open-source technologies:
- **Frontend:** React, Vite, Plotly.js, TailwindCSS, Lucide Icons
- **Backend:** FastAPI, SciPy, statsmodels, pandas, NumPy, ReportLab
- **Development:** Claude Code, GitHub

**Special Thanks:**
- The open-source community for amazing tools and libraries
- All contributors and supporters of MasterStat
- Research and engineering professionals who provide feedback

---

## Contact

- **Issues & Bugs:** [GitHub Issues](https://github.com/SimiamiDrew101/MasterStat/issues)
- **Feature Requests:** [GitHub Discussions](https://github.com/SimiamiDrew101/MasterStat/discussions)
- **Support:** [Ko-fi](https://ko-fi.com/MasterStat)

---

<div align="center">

**Made with ❤️ for the research and engineering community**

[⬆ Back to Top](#masterstat)

</div>
