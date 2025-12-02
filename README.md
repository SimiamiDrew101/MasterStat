# MasterStat

**Professional-grade statistical analysis and Design of Experiments (DOE) platform**

Free, open-source web application for researchers, engineers, and data scientists. Built with React, FastAPI, and modern visualization libraries.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ko-fi](https://img.shields.io/badge/Support-Ko--fi-FF5E5B?logo=ko-fi&logoColor=white)](https://ko-fi.com/MasterStat)

[Features](#features) • [Quick Start](#quick-start) • [Installation](#installation) • [Documentation](#documentation) • [Contributing](#contributing)

---

## Features

### Statistical Analysis Modules

- **ANOVA** - One-way, two-way, repeated measures with post-hoc tests (Tukey, Bonferroni, Scheffé)
- **Hypothesis Testing** - t-tests, F-tests, Z-tests, Mann-Whitney, Kruskal-Wallis
- **Power Analysis** - Sample size calculation, effect size estimation
- **Factorial Designs** - 2^k, 3^k, fractional factorial, Plackett-Burman, screening
- **Response Surface Methodology** - Central Composite Design, Box-Behnken, optimization
- **Mixed Models** - Split-plot, nested, repeated measures, random effects
- **Block Designs** - RCBD, Latin squares, incomplete blocks
- **Mixture Design** - Simplex-centroid, simplex-lattice, constrained regions
- **Robust Design** - Taguchi methods, parameter design, signal-to-noise ratios
- **Bayesian DOE** - Sequential experimentation, posterior optimization

### Visualizations

- Interactive 3D response surfaces with rotation and zoom
- Contour plots with optimization paths
- Diagnostic plots (residuals, Q-Q, Cook's distance, leverage)
- Main effects and interaction plots
- Cube plots for factorial designs
- Half-normal plots for effect screening

### Export Capabilities

- **PDF** - Publication-ready reports with embedded figures
- **Excel** - Multi-sheet workbooks with formatted tables
- **CSV/TSV** - Standard data formats
- **JMP/Minitab** - Industry-standard DOE formats
- **PNG** - High-resolution figures (300-1200 DPI)

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

### Local Development (Frontend Only)

```bash
git clone https://github.com/SimiamiDrew101/MasterStat.git
cd MasterStat/frontend
npm install
npm run dev
```

**Note:** Backend statistical features require Docker.

---

## Installation

### Prerequisites

**For Docker (recommended):**
- [Docker Desktop](https://www.docker.com/products/docker-desktop)

**For local development:**
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

For frontend-only development (no backend features):

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

### Educational Features

Built-in DOE glossary with 50+ terms including:
- Aliasing and confounding patterns
- Blocking strategies
- Central Composite Design structures
- Effect hierarchy principles
- Interaction interpretation
- Power and sample size
- Resolution in fractional factorials

Hover over any statistical term for context-sensitive help with examples.

---

## Technology Stack

**Frontend:**
- React 18 with Vite 5.4
- Plotly.js for interactive visualizations
- TailwindCSS for styling
- Axios for API communication

**Backend:**
- FastAPI (Python 3.11)
- scipy, statsmodels for statistical computing
- pandas, numpy for data manipulation

**Export:**
- jsPDF for PDF generation
- xlsx for Excel workbooks
- html2canvas for figure capture

---

## Architecture

**Component Organization:**

```
frontend/src/
├── pages/          # 12 analysis modules
├── components/     # Reusable UI components
└── services/       # API client and utilities

backend/app/
├── api/           # FastAPI endpoints
└── models/        # Statistical models
```

**63 React components** organized into:
- 28 visualization components
- 20 analysis components
- 15 UI/utility components

**Project Statistics:**
- 12 statistical modules
- 35+ visualization types
- 7+ export formats
- ~20,000 lines of code
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

**Development Workflow:**

```bash
# Fork repository on GitHub
git clone https://github.com/YOUR_USERNAME/MasterStat.git
cd MasterStat/frontend

# Install dependencies
npm install

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
npm run dev

# Commit and push
git commit -m "Add: Brief description"
git push origin feature/your-feature-name

# Open Pull Request on GitHub
```

**Code Guidelines:**
- Use functional React components with hooks
- Follow TailwindCSS utility-first approach
- Write descriptive variable names
- Add comments for complex logic
- Include JSDoc for component props

---

## Support

MasterStat is free and open-source software. Support the project:

**Financial Support:**

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/MasterStat)

Your contributions help us develop new features, improve visualizations, and keep the project free for everyone.

**Other Ways to Help:**
- Star this repository
- Share with colleagues and students
- Write tutorials or blog posts
- Cite MasterStat in publications
- Report bugs and suggest improvements
- Contribute code or documentation

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this software commercially or privately. Attribution appreciated but not required.

---

## Acknowledgments

Built with excellent open-source technologies:
- React, Vite, Plotly.js, TailwindCSS, Lucide Icons
- FastAPI, SciPy, statsmodels, pandas

Inspired by professional DOE software including JMP, Minitab, and Design-Expert.

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
