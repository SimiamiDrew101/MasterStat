# MasterStat

<div align="center">

![MasterStat Banner](https://img.shields.io/badge/MasterStat-Statistical%20Analysis-blue?style=for-the-badge)
[![Ko-fi](https://img.shields.io/badge/Support%20on-Ko--fi-FF5E5B?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/MasterStat)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=white)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev/)

**A comprehensive, free, and open-source statistical analysis platform for Design of Experiments (DOE)**

*Professional-grade statistical tools with a modern, intuitive interface*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation) • [Support](#-support)

</div>

---

## 🎯 Overview

MasterStat is a modern web-based statistical analysis platform designed for researchers, engineers, data scientists, and anyone working with experimental design. With **12 comprehensive statistical modules**, **63 React components**, and **35+ visualization types**, MasterStat brings professional-grade Design of Experiments capabilities to your browser.

### Why MasterStat?

- 🆓 **100% Free & Open Source** - No licensing fees, no subscriptions
- 🎨 **Beautiful Visualizations** - Interactive 3D surfaces, contour plots, and diagnostic charts
- 📊 **Publication Ready** - Export high-quality figures and comprehensive reports
- 🎓 **Educational** - Built-in DOE glossary with 50+ terms and interactive tooltips
- 🚀 **Modern Tech Stack** - Built with React 18, Vite, Plotly.js, and TailwindCSS
- 🌐 **Web-Based** - No installation required, works on any device with a browser

---

## ✨ Features

### 🧪 Statistical Analysis Modules

<table>
<tr>
<td width="50%">

#### Core Analysis
- **📊 ANOVA** - One-way, two-way, post-hoc tests (Tukey, Bonferroni, Scheffé)
- **📈 Hypothesis Testing** - t-tests, F-tests, Z-tests, Mann-Whitney, Kruskal-Wallis
- **🎯 Power Analysis** - Sample size calculation, effect size estimation
- **📉 Regression** - Linear, quadratic, polynomial models

</td>
<td width="50%">

#### Advanced DOE
- **⛰️ Response Surface Methodology** - CCD, Box-Behnken, steepest ascent
- **🔲 Factorial Designs** - 2^k, 3^k, fractional factorial, Plackett-Burman
- **🔀 Mixed Models** - Split-plot, nested, repeated measures, growth curves
- **🛡️ Robust Design** - Taguchi methods, noise factors, parameter design

</td>
</tr>
<tr>
<td width="50%">

#### Specialized Designs
- **🔲 Block Designs** - RCBD, Latin squares, incomplete blocks
- **⚗️ Mixture Design** - Simplex-centroid, simplex-lattice, constraints
- **🔮 Bayesian DOE** - Sequential designs, optimal uncertainty quantification
- **✨ Experiment Wizard** - AI-powered design recommendations

</td>
<td width="50%">

#### Visualization & Export
- **3D Response Surfaces** - Rotatable, zoomable, interactive
- **Contour Plots** - With optimization paths and stationary points
- **Diagnostic Plots** - Residuals, Q-Q, Cook's distance, leverage
- **Multi-Format Export** - PDF, Excel, CSV, JMP, Minitab

</td>
</tr>
</table>

---

## 🎨 Visualizations

MasterStat includes **35+ professional visualization types**:

| Category | Visualizations |
|----------|---------------|
| **3D Plots** | Response surfaces, cube plots (2³/2⁴), scatter plots |
| **2D Plots** | Contour plots, interaction plots, main effects, sliced surfaces |
| **Diagnostics** | Residual vs fitted, Q-Q plots, Cook's distance, leverage plots |
| **Statistical** | Box plots, violin plots, forest plots, Pareto charts, half-normal plots |
| **Specialized** | Growth curves, variance decomposition, correlation heatmaps, power curves |

### Visualization Features
- ✅ **Interactive** - Zoom, pan, rotate 3D plots
- ✅ **Customizable** - Adjust colors, fonts, labels
- ✅ **Export Ready** - High-resolution PNG, SVG export
- ✅ **Dark Theme** - Modern, professional appearance
- ✅ **Responsive** - Works on desktop, tablet, and mobile

---

## 🚀 Installation

MasterStat can be run in two ways: **Docker** (recommended, includes full backend) or **Local Development** (frontend only).

### Method 1: Docker (Recommended)

Run the complete application with both frontend and backend using Docker.

#### Prerequisites
- **Docker Desktop** ([Download](https://www.docker.com/products/docker-desktop))

#### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/SimiamiDrew101/MasterStat.git
cd MasterStat

# 2. Start the application
./start.sh

# Or use docker-compose directly:
docker-compose up --build

# 3. Open in browser
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

#### Stop the Application

```bash
# Press Ctrl+C in terminal, or run:
docker-compose down
```

---

### Method 2: Local Development (Frontend Only)

Run just the React frontend without Docker. Backend features will not be available.

#### Prerequisites
- **Node.js** v16 or higher ([Download](https://nodejs.org/))
- **npm** (comes with Node.js)

#### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/SimiamiDrew101/MasterStat.git
cd MasterStat/frontend

# 2. Install dependencies
npm install

# 3. Start development server
npm run dev

# 4. Open in browser
# Navigate to http://localhost:5173
```

#### Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview

# The optimized files will be in the dist/ directory
```

---

### Which Method Should I Use?

| Feature | Docker | Local Development |
|---------|--------|-------------------|
| **Full backend** | ✅ Yes | ❌ No |
| **Statistical computations** | ✅ Yes | ❌ No |
| **Frontend UI** | ✅ Yes | ✅ Yes |
| **Setup complexity** | Medium | Easy |
| **Prerequisites** | Docker Desktop | Node.js |
| **Best for** | Full analysis, production | UI development, quick preview |

**Recommendation:** Use Docker for the complete experience with all statistical features.

### Docker Deployment

MasterStat includes full Docker support with optimized configurations for Apple Silicon (M1/M2/M3).

#### Prerequisites
- Docker Desktop for Mac ([Download](https://www.docker.com/products/docker-desktop))
- Docker Compose (included with Docker Desktop)

#### Quick Start with Docker

```bash
# 1. Clone the repository
git clone https://github.com/SimiamiDrew101/MasterStat.git
cd MasterStat

# 2. Build and start containers
docker-compose up --build

# 3. Access the application
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

#### Docker Services

The `docker-compose.yml` configures two services:

- **Backend** (Python/FastAPI)
  - Port: 8000
  - Platform: linux/arm64 (M1/M2/M3 optimized)
  - Hot-reload enabled for development

- **Frontend** (React/Vite)
  - Port: 5173
  - Platform: linux/arm64 (M1/M2/M3 optimized)
  - Hot-reload enabled for development

#### Docker Commands

```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# Stop services
docker-compose down

# Rebuild containers
docker-compose up --build

# View logs
docker-compose logs -f

# Stop and remove volumes
docker-compose down -v
```

#### Development Mode

Both containers support hot-reloading:
- **Backend**: Changes in `backend/app/` automatically reload FastAPI
- **Frontend**: Changes in `frontend/src/` trigger instant Vite updates

#### Troubleshooting Docker

**Port conflicts:**
```bash
# Check what's using ports 8000 or 5173
lsof -ti:8000
lsof -ti:5173

# Stop containers and remove
docker-compose down
```

**Clean rebuild:**
```bash
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

---

## 📖 Usage

### Quick Start Examples

#### Example 1: Design a Factorial Experiment

1. Navigate to **Experiment Wizard** from the home page
2. Select your **goal** (Screening, Modeling, or Optimization)
3. Enter **number of factors** and **factor names**
4. Specify **constraints** (budget, time, resources)
5. Review **AI-powered recommendations**
6. Generate your **design matrix**
7. Export to **Excel, PDF, or CSV**

#### Example 2: Analyze Response Surface Data

1. Go to **Response Surface** module
2. Upload your **experimental data** or use built-in examples
3. Fit a **second-order model** (quadratic with interactions)
4. View **3D surface** and **contour plots**
5. Perform **optimization** (steepest ascent, canonical analysis)
6. Validate with **diagnostic plots**
7. Export **publication-ready figures**

#### Example 3: Run ANOVA with Post-hoc Tests

1. Open **ANOVA** module
2. Choose **One-way** or **Two-way** ANOVA
3. Input your **group data**
4. Run the analysis
5. View **F-test results** and **effect sizes**
6. Perform **post-hoc comparisons** (Tukey HSD)
7. Check **assumptions** (normality, homogeneity)
8. Generate **comprehensive report**

---

## 🛠️ Technology Stack

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 18.x | UI framework |
| **Vite** | 5.4 | Build tool & dev server |
| **Plotly.js** | 2.x | Interactive 3D visualizations |
| **TailwindCSS** | 3.x | Utility-first styling |
| **Lucide React** | latest | Beautiful icons |
| **Axios** | 1.x | HTTP client |
| **React Router** | 6.x | Navigation |

### Export Libraries

| Library | Purpose |
|---------|---------|
| **jsPDF** | PDF report generation |
| **jsPDF-autotable** | Tables in PDFs |
| **xlsx** | Excel workbook export |
| **html2canvas** | Screenshot capture |

### Backend (Python)

| Library | Purpose |
|---------|---------|
| **FastAPI** | Modern web framework |
| **scipy** | Scientific computing |
| **statsmodels** | Statistical modeling |
| **pandas** | Data manipulation |
| **numpy** | Numerical computing |

---

## 📚 Documentation

### Component Architecture

MasterStat is built with **63 React components** organized into:

**Visualization Components (28):**
- `ResponseSurface3D` - Interactive 3D response surfaces
- `ContourPlot` - 2D contour plots with overlays
- `CubePlot` - Factorial design cube visualizations
- `ResidualAnalysis` - Comprehensive diagnostic suite
- `HalfNormalPlot` - Lenth's method for effect screening
- And 23 more specialized visualization components

**Analysis Components (20):**
- `MultiResponseManager` - Multi-objective optimization
- `PredictionProfiler` - Interactive factor profiling
- `DesignRecommendationStep` - AI-powered design suggestions
- `CrossValidationResults` - Model validation metrics
- And 16 more analysis components

**UI Components (15):**
- `InteractiveTooltip` - Educational DOE tooltips
- `FactorInteractionSelector` - Factor interaction selection
- `SmartValidation` - Input validation with feedback
- And 12 more UI components

### Export Formats

MasterStat supports comprehensive export capabilities:

#### PDF Reports
- **Full experiment documentation** with embedded figures
- **Automatic figure numbering** and captions
- **Professional layout** matching journal standards
- **Includes:** Design matrix, ANOVA tables, model equations, diagnostic plots

#### Excel Workbooks
- **Multi-sheet exports** (Design, Summary, Instructions)
- **Formatted tables** with conditional formatting
- **Ready for data entry** with pre-configured formulas

#### Industry-Standard Formats
- **JMP Format** - Tab-delimited .jmp.txt files
- **Minitab Format** - CSV with StdOrder/RunOrder columns
- **CSV/TSV** - Standard data interchange formats
- **JSON** - Full metadata and configuration

#### Figures
- **PNG** - Screen resolution or high-DPI (300, 600, 1200 DPI)
- **SVG** - Vector graphics for publications (coming soon)
- **Clipboard** - Copy any figure directly to clipboard

---

## 🎓 Educational Features

### Built-in DOE Glossary

MasterStat includes an **interactive glossary** with **50+ DOE terms**:

- **Aliasing** - Understanding confounding in fractional factorials
- **Blocking** - Controlling for nuisance variables
- **Central Composite Design** - RSM design structure
- **Confounding** - When effects cannot be separated
- **Desirability Function** - Multi-response optimization
- **Interaction** - When factors combine synergistically
- **Power** - Probability of detecting true effects
- **Resolution** - Quality measure for fractional factorials
- And 40+ more terms with practical advice and examples

### Interactive Tooltips

Every analysis step includes **context-sensitive help**:
- Hover over any statistical term to see definition
- Click for detailed explanation with examples
- View related concepts and cross-references
- Access practical advice for your specific situation

---

## 🔬 Advanced Features

### Multi-Response Optimization

- **Simultaneous optimization** of multiple responses
- **Desirability functions** (maximize, minimize, target)
- **Overlay contour plots** for visual trade-off analysis
- **Constrained optimization** with box and linear constraints

### Sequential Experimentation

- **Screening → Characterization → Optimization** workflow
- **Steepest ascent/descent** path calculation
- **Foldover designs** for de-aliasing effects
- **Augmented designs** to increase resolution

### Model Diagnostics

Complete suite of diagnostic tools:
- **Residual plots** - Check assumptions
- **Normal Q-Q plots** - Assess normality
- **Cook's distance** - Identify influential points
- **Leverage plots** - Detect unusual factor combinations
- **VIF analysis** - Check for multicollinearity

### Cross-Validation

- **K-fold cross-validation** for model validation
- **Predicted R²** calculation
- **PRESS statistic** for prediction quality
- **Validation set** support

---

## 📊 Example Workflows

### Workflow 1: Screening Experiment

```
1. Start with Experiment Wizard
2. Goal: Screening (identify active factors)
3. 8 factors, limited budget
4. Wizard recommends: Plackett-Burman design (12 runs)
5. Generate design matrix
6. Run experiments, collect data
7. Analyze with Half-Normal plot (Lenth's method)
8. Identify 3 significant factors
9. Proceed to characterization
```

### Workflow 2: Response Surface Optimization

```
1. Start with 2-3 important factors (from screening)
2. Goal: Optimization
3. Wizard recommends: Central Composite Design (Face-centered)
4. Generate design (17 runs for 3 factors)
5. Collect response data
6. Fit second-order model
7. Check diagnostics (R², residuals, lack-of-fit)
8. Visualize 3D surface and contour plots
9. Perform canonical analysis
10. Find optimal settings (steepest ascent)
11. Confirm with verification runs
```

### Workflow 3: Robust Design (Taguchi)

```
1. Navigate to Robust Design module
2. Define control factors (what you can control)
3. Define noise factors (sources of variation)
4. Select inner/outer array design
5. Calculate signal-to-noise ratios
6. Identify factor settings that minimize variation
7. Validate with confirmation experiments
```

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute

- 🐛 **Report Bugs** - Open an issue with detailed description
- 💡 **Suggest Features** - Share your ideas for improvements
- 📝 **Improve Documentation** - Help make MasterStat easier to use
- 🔧 **Submit Pull Requests** - Fix bugs or add features
- 🎨 **Design** - Improve UI/UX
- 🧪 **Testing** - Help test new features

### Development Setup

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/MasterStat.git
cd MasterStat/frontend

# Install dependencies
npm install

# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# Test thoroughly

# Commit with clear message
git commit -m "Add: Brief description of your changes"

# Push to your fork
git push origin feature/your-feature-name

# Open a Pull Request on GitHub
```

### Code Style

- Use **functional React components** with hooks
- Follow **TailwindCSS** utility-first approach
- Write **clear, descriptive** variable names
- Add **comments** for complex logic
- Include **JSDoc** for component props

---

## 💖 Support

MasterStat is **free and open-source** software. If you find it useful for your research, teaching, or work, please consider supporting the project:

### 💰 Financial Support

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/MasterStat)

**Every contribution helps us:**
- Develop new features and modules
- Improve visualizations and UI/UX
- Add more statistical methods
- Maintain and update dependencies
- Keep the project free for everyone

### ⭐ Other Ways to Support

- **Star this repository** on GitHub
- **Share** MasterStat with colleagues and students
- **Write tutorials** or blog posts about using MasterStat
- **Cite** MasterStat in your publications
- **Report bugs** and suggest improvements
- **Contribute code** or documentation

---

## 📄 License

MasterStat is open-source software released under the **MIT License**.

You are free to:
- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Use privately
- ✅ Include in proprietary software

**Attribution appreciated but not required.**

---

## 🙏 Acknowledgments

MasterStat is made possible by:

### Open Source Technologies
- **React Team** - For the amazing React framework
- **Vite Team** - For blazing-fast build tooling
- **Plotly Team** - For powerful visualization library
- **TailwindCSS Team** - For the utility-first CSS framework
- **Lucide Icons** - For beautiful, consistent icons

### Statistical Computing
- **SciPy** - Scientific computing in Python
- **statsmodels** - Statistical modeling and testing
- **pandas** - Data manipulation and analysis

### Inspiration
- **JMP** - Professional statistical software excellence
- **Minitab** - Quality improvement tools
- **Design-Expert** - DOE software standards
- **Academic Research** - Decades of DOE methodology

---

## 📞 Contact & Community

### Get in Touch

- **GitHub Issues** - Bug reports and feature requests
- **Discussions** - General questions and ideas
- **Ko-fi** - Support and suggestions

### Roadmap

We're continuously improving MasterStat. **Upcoming features:**

**Short-term (Next 3 months):**
- ✅ Vector graphics export (SVG, EPS)
- ✅ Light theme for publications
- ✅ Enhanced PDF reports with embedded figures
- ✅ Correlation heatmaps with clustering

**Medium-term (3-6 months):**
- 🔄 Desirability function 3D surfaces
- 🔄 Optimization path animation
- 🔄 Composite figure builder (multi-panel layouts)
- 🔄 Mathematical notation support (KaTeX)

**Long-term (6-12 months):**
- 🔮 Cloud hosting and sharing
- 🔮 Collaborative experiments
- 🔮 API for programmatic access
- 🔮 Mobile app (iOS/Android)

---

## 📊 Project Stats

- **Components:** 63 React components
- **Modules:** 12 statistical analysis modules
- **Visualizations:** 35+ plot types
- **Export Formats:** 7+ formats (PDF, Excel, CSV, JMP, Minitab, JSON, PNG)
- **DOE Glossary:** 50+ terms with explanations
- **Lines of Code:** ~20,000+ (frontend + backend)
- **Dependencies:** Modern, well-maintained packages
- **Browser Support:** Chrome 90+, Firefox 88+, Safari 14+, Edge 90+

---

## 🌟 Star History

If MasterStat helped you with your research or work, please consider giving it a star! ⭐

---

<div align="center">

**Made with ❤️ for the research and engineering community**

[⬆ Back to Top](#masterstat)

</div>
