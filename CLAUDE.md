# MasterStat - Project Context for Claude Code

**Last Updated:** 2026-02-01
**Status:** Tier 3 In Progress | Features 1-6 COMPLETE (Reliability, GLM, Custom Design, Enhanced SPC/MSA, Predictive Modeling, Complete Mixture Designs)

---

## Quick Start for New Sessions

```bash
# 1. Start backend
cd /Users/nj/Desktop/MasterStat/backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 2. Start frontend (new terminal)
cd /Users/nj/Desktop/MasterStat/frontend
npm run dev

# 3. Access app
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000/docs
```

---

## Project Overview

**MasterStat** is a professional statistical analysis desktop application providing:
- Design of Experiments (DOE): CCD, Box-Behnken, DSD, Plackett-Burman, Factorial
- Statistical Analysis: ANOVA, RSM, Mixed Models, Nonlinear Regression
- Multi-Response Optimization with desirability functions
- Session persistence with IndexedDB
- Cross-platform desktop builds (macOS, Windows, Linux)

**Tech Stack:**
| Layer | Technologies |
|-------|-------------|
| Backend | Python 3.11+ / FastAPI 0.109 / Uvicorn |
| Frontend | React 18.2 / Vite 5.0 / TailwindCSS 3.4 |
| Desktop | Electron 28.1 (cross-platform) |
| Visualization | Plotly.js 3.3 / Recharts 2.10 |
| Statistics | SciPy / statsmodels / NumPy / pandas / pyDOE2 |
| Persistence | IndexedDB via Dexie.js 4.2 |

---

## Current Implementation Status

### Tier 1: COMPLETE
All core statistical analysis features working.

### Tier 2: 100% COMPLETE (as of 2026-01-28)

| Feature | Status | Key Files |
|---------|--------|-----------|
| **Feature 1: Experiment Wizard** | COMPLETE | `rsm.py` (DSD, PB, confounding), `DesignPreviewVisualization.jsx`, `PowerCurvePlot.jsx`, `ConfoundingDiagram.jsx` |
| **Feature 2: Model Validation** | COMPLETE | `model_validation.py`, `ModelValidation.jsx`, validation endpoints in all analysis modules |
| **Feature 3: Multi-Response Optimization** | COMPLETE | `rsm.py` (desirability methods, overlay contours), `OverlayContourPlot.jsx`, RSM.jsx enhancements |
| **Feature 4: Session Management** | COMPLETE | `sessionManager.js`, `SessionContext.jsx`, `SessionHistory.jsx`, App.jsx integration |

### Tier 3: IN PROGRESS (as of 2026-01-31)

| Feature | Status | Key Files |
|---------|--------|-----------|
| **Feature 1: Reliability/Survival Analysis** | COMPLETE | `reliability.py`, `ReliabilityAnalysis.jsx`, `SurvivalCurvePlot.jsx`, `WeibullPlot.jsx`, `HazardRatioForest.jsx`, `LifeDistributionResults.jsx` |
| **Feature 2: Generalized Linear Models (GLM)** | COMPLETE | `glm.py`, `GLM.jsx` |
| **Feature 3: Custom Design Platform** | COMPLETE | `custom_design.py`, `CustomDesign.jsx` |
| **Feature 4: Enhanced SPC/MSA** | COMPLETE | `quality_control.py` (CUSUM, EWMA, CI), `msa.py`, `QualityControl.jsx` (MSA tab) |
| **Feature 5: Predictive Modeling Suite** | COMPLETE | `predictive_modeling.py`, `PredictiveModeling.jsx` |
| **Feature 6: Complete Mixture Designs** | COMPLETE | `mixture.py`, `MixtureDesign.jsx` (extreme vertices, ternary plots, trace plots, mixture+process) |

### Electron Builds: COMPLETE (2026-01-28)

| Platform | Packages | Location |
|----------|----------|----------|
| macOS | DMG + ZIP (arm64 & x64) | `dist-electron/MasterStat-1.0.0*.dmg` |
| Windows | NSIS installer + Portable | `dist-electron/MasterStat*.exe` |
| Linux | AppImage + deb (arm64) | `dist-electron/MasterStat*.AppImage`, `*.deb` |

---

## Directory Structure

```
MasterStat/
├── backend/                      # Python/FastAPI statistical engine
│   ├── app/
│   │   ├── api/                 # 22 API modules (rsm.py, anova.py, reliability.py, glm.py, custom_design.py, msa.py, predictive_modeling.py, mixture.py, etc.)
│   │   ├── utils/               # model_validation.py, report_generator.py
│   │   └── main.py              # FastAPI entry point
│   └── requirements.txt
├── frontend/                     # React/Vite web interface
│   ├── src/
│   │   ├── pages/               # 22 page components
│   │   ├── components/          # 94+ reusable components
│   │   ├── contexts/            # SessionContext.jsx
│   │   └── utils/               # sessionManager.js, smartValidation.js, etc.
│   └── package.json
├── electron/                     # Electron desktop wrapper
│   ├── main.js                  # Main process (starts backend, creates window)
│   └── preload.js               # Security layer
├── build/                        # Build resources (icons, entitlements)
├── dist-electron/                # Built installers for all platforms
├── CLAUDE.md                     # This file
├── TIER2_STATUS.md              # Detailed implementation notes
└── package.json                  # Root Electron/build config
```

---

## Key Files Reference

### Backend - Most Important Files

| File | Lines | Purpose |
|------|-------|---------|
| `backend/app/api/rsm.py` | 4,600+ | RSM, CCD, Box-Behnken, DSD, desirability optimization, overlay contours |
| `backend/app/api/reliability.py` | 1,175 | Life distributions, Kaplan-Meier, Cox PH, ALT, test planning |
| `backend/app/api/anova.py` | 1,700+ | ANOVA analysis with validation |
| `backend/app/api/factorial.py` | 2,200+ | Factorial designs with validation |
| `backend/app/api/mixed_models.py` | 2,500+ | Split-plot, nested, random effects |
| `backend/app/utils/model_validation.py` | 456 | PRESS, k-fold CV, adequacy scoring |

### Frontend - Most Important Files

| File | Purpose |
|------|---------|
| `frontend/src/pages/RSM.jsx` | Main RSM page (largest: ~3000 lines) |
| `frontend/src/pages/ReliabilityAnalysis.jsx` | Reliability/Survival analysis (1,528 lines) |
| `frontend/src/components/OverlayContourPlot.jsx` | Multi-response contour visualization |
| `frontend/src/components/SurvivalCurvePlot.jsx` | Kaplan-Meier survival curve visualization |
| `frontend/src/components/ModelValidation.jsx` | Validation results display |
| `frontend/src/components/SessionHistory.jsx` | Session browser UI |
| `frontend/src/contexts/SessionContext.jsx` | Global session state management |
| `frontend/src/utils/sessionManager.js` | IndexedDB operations via Dexie.js |
| `frontend/src/App.jsx` | App entry, routing, SessionProvider wrapper |

---

## API Endpoints - Key Features

### Design Generation
```
POST /api/rsm/ccd/generate          # Central Composite Design
POST /api/rsm/box-behnken/generate  # Box-Behnken Design
POST /api/rsm/dsd/generate          # Definitive Screening Design
POST /api/rsm/plackett-burman/generate  # Plackett-Burman screening
POST /api/factorial/generate        # Factorial designs
```

### Analysis
```
POST /api/anova/analyze             # ANOVA analysis
POST /api/rsm/fit-model             # RSM model fitting
POST /api/mixed-models/analyze      # Mixed model analysis
```

### Multi-Response Optimization (Feature 3)
```
POST /api/rsm/desirability-optimization  # Composite desirability (3 methods)
POST /api/rsm/multi-response-contour     # Overlay contour data
```

### Model Validation (Feature 2)
```
POST /api/anova/validate-model
POST /api/factorial/validate-model
POST /api/mixed-models/validate-model
POST /api/nonlinear-regression/validate-model
POST /api/rsm/confounding-analysis
```

### Reliability/Survival Analysis (Tier 3 Feature 1)
```
POST /api/reliability/life-distribution    # Fit Weibull, lognormal, exponential, loglogistic
POST /api/reliability/kaplan-meier         # Survival curves with log-rank test
POST /api/reliability/cox-ph               # Cox Proportional Hazards regression
POST /api/reliability/alt                  # Accelerated Life Testing
POST /api/reliability/test-planning        # Sample size for reliability tests
```

---

## Code Patterns

### Backend - FastAPI Endpoint
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

class RequestModel(BaseModel):
    field: str = Field(..., description="Description")

@router.post("/endpoint-name")
async def endpoint_function(request: RequestModel):
    try:
        # Implementation
        return {"result": data, "message": "Success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Frontend - React Component with API Call
```jsx
import React, { useState } from 'react';
import axios from 'axios';

const Component = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_URL}/api/endpoint`, data);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-slate-800 text-gray-100 p-6 rounded-lg">
      {/* Dark mode styling */}
    </div>
  );
};
```

### Session Management Pattern
```jsx
import { useSession } from '../contexts/SessionContext';

const AnalysisPage = () => {
  const { saveCurrentSession, currentSession } = useSession();

  const handleSave = async () => {
    await saveCurrentSession('Session Name', {
      analysis_type: 'RSM',
      data: { factors, responses, originalData },
      results: { modelFit, optimization }
    });
  };
};
```

### Tailwind Dark Mode Classes
```jsx
// Container: bg-slate-800 text-gray-100 p-6 rounded-lg
// Card: bg-slate-700/50 border border-slate-600 p-4
// Input: bg-slate-900 text-gray-100 border border-slate-600 rounded px-3 py-2
// Button: bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg
// Success: bg-green-600, Warning: bg-yellow-600, Error: bg-red-600
```

### Plotly Dark Theme
```javascript
const layout = {
  paper_bgcolor: '#1e293b',
  plot_bgcolor: '#0f172a',
  font: { color: '#e2e8f0' }
};
```

---

## Development Commands

### Daily Development
```bash
# Backend
cd /Users/nj/Desktop/MasterStat/backend
python -m uvicorn app.main:app --reload --port 8000

# Frontend
cd /Users/nj/Desktop/MasterStat/frontend
npm run dev

# Test endpoint
curl -X POST http://localhost:8000/api/rsm/dsd/generate \
  -H "Content-Type: application/json" \
  -d '{"n_factors": 3, "factor_names": ["A", "B", "C"]}'
```

### Building Electron Apps
```bash
cd /Users/nj/Desktop/MasterStat

# Build all platforms
npm run dist

# Platform-specific
npm run build:mac      # macOS DMG + ZIP
npm run build:win      # Windows NSIS + Portable
npm run build:linux    # Linux AppImage + deb
```

### Git Workflow
```bash
git status
git add .
git commit -m "feat: Description"
git push origin main
```

---

## Ports & URLs

| Service | URL |
|---------|-----|
| Backend API | http://localhost:8000 |
| Frontend Dev | http://localhost:5173 |
| Swagger Docs | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |

---

## Codebase Statistics

| Area | Files | Lines |
|------|-------|-------|
| Backend (Python) | 20 | ~20,600 |
| Frontend (React) | 124 | ~55,600 |
| Electron | 2 | ~240 |
| **Total** | **146** | **~76,400** |

### Backend API Modules (17 total)
| Module | Lines | Purpose |
|--------|-------|---------|
| `rsm.py` | 4,605 | RSM, CCD, Box-Behnken, DSD, desirability, overlay contours |
| `reliability.py` | 1,175 | Life distributions, Kaplan-Meier, Cox PH, ALT, test planning |
| `mixed_models.py` | 2,529 | Split-plot, nested, random effects, BLUPs |
| `factorial.py` | 2,219 | 2^k, 2^(k-p), confounding, aliasing |
| `power_analysis.py` | 1,721 | Sample size, power curves, effect size |
| `anova.py` | 1,698 | 1-way, 2-way, ANCOVA, post-hoc tests |
| `block_designs.py` | 1,584 | RCBD, Latin squares, BIBD |
| `hypothesis_testing.py` | 692 | t-tests, F-tests, Z-tests |
| `nonlinear_regression.py` | 651 | Curve fitting, validation |
| `quality_control.py` | 639 | SPC, control charts |
| `imputation.py` | 562 | MICE, KNN, missing data |
| `bayesian_doe.py` | 542 | Bayesian inference, sequential |
| `preprocessing.py` | 497 | Outlier detection, transformation |
| `protocol.py` | 444 | Randomization, blinding, PDF |
| `import_data.py` | 306 | CSV, Excel, JSON parsing |
| `optimal_designs.py` | 225 | D/I/A-optimal designs |
| `prediction_profiler.py` | 217 | Sensitivity analysis |

### Frontend Pages (19 total)
| Page | Lines | Purpose |
|------|-------|---------|
| `RSM.jsx` | 2,863 | Response Surface Methodology |
| `FactorialDesigns.jsx` | 2,262 | Full/fractional factorial |
| `ExperimentPlanning.jsx` | 2,148 | Power, sample size |
| `MixedModels.jsx` | 1,745 | Mixed models analysis |
| `ANOVA.jsx` | 1,636 | ANOVA analysis |
| `ReliabilityAnalysis.jsx` | 1,528 | Reliability/Survival analysis |
| `ExperimentWizardPage.jsx` | 1,307 | Guided design creation |
| `BayesianDOE.jsx` | 1,195 | Bayesian analysis |
| `BlockDesigns.jsx` | 1,166 | Block design analysis |
| Plus 10 more pages... | | |

### Frontend Components (89 total)
- **Visualization (15):** 3D surfaces, contour plots, interaction plots
- **Analysis/Diagnostics (20):** Model validation, residuals, BLUPs
- **Design/Planning (12):** Wizards, power curves, confounding diagrams
- **Data Processing (8):** File upload, transformations, imputation
- **Multi-Response (5):** Overlay contours, desirability functions
- **Reliability (4):** SurvivalCurvePlot, WeibullPlot, HazardRatioForest, LifeDistributionResults
- **Profiling (4):** Sensitivity curves, prediction profiler
- **UI/Utilities (21):** Result cards, session history, tooltips

---

## Tier 3 Development Plan (Professional Feature Expansion)

**Goal:** Expand MasterStat with advanced statistical capabilities for professional statisticians.

**Dropped from original plan:**
- ~~VR/AR support~~ (impractical for statistical software)
- ~~Cloud sync/collaboration~~ (complex, lower priority)

### Priority Matrix

| Feature | User Impact | Importance | Complexity | Priority |
|---------|-------------|------------|------------|----------|
| Reliability/Survival | ★★★★★ | Critical | Medium | **1st** |
| GLM | ★★★★★ | Critical | Low-Med | **2nd** |
| Custom Design | ★★★★★ | Critical | High | **3rd** |
| SPC/MSA | ★★★★☆ | Important | Medium | **4th** |
| Predictive Modeling | ★★★★☆ | Important | Medium | **5th** |
| Mixture Designs | ★★★★☆ | Important | Medium | **6th** |
| Graph Builder | ★★★★☆ | Important | High | **7th** |
| Dynamic Linking | ★★★☆☆ | Nice-have | Medium | **8th** |
| Space-Filling | ★★★☆☆ | Nice-have | Low | **9th** |
| Time Series | ★★★☆☆ | Nice-have | Medium | **10th** |
| Reporting System | ★★★☆☆ | Nice-have | Medium | **11th** |
| Pub-Quality Export | ★★☆☆☆ | Nice-have | Low | **12th** |

---

### Tier 3A: Critical Statistical Gaps (Highest Priority)

#### Feature 1: Reliability & Survival Analysis ✅ COMPLETE
**Status:** Implemented 2026-01-31
**Why:** Major gap. Essential for manufacturing, pharmaceutical, engineering users.

| Component | Description |
|-----------|-------------|
| Life Distribution Fitting | Weibull, lognormal, exponential, generalized gamma |
| Probability Plots | Weibull plots with confidence bounds |
| Kaplan-Meier Survival | Non-parametric survival curves, log-rank tests |
| Cox Proportional Hazards | Semi-parametric regression for censored data |
| Accelerated Life Testing | Arrhenius, power law models for stress testing |
| Reliability Test Planning | Sample size for demonstration tests |

**Implementation:**
- Backend: New `backend/app/api/reliability.py` (~1,500 lines)
- Frontend: New `frontend/src/pages/ReliabilityAnalysis.jsx` + 6-8 components
- Dependencies: `lifelines` or `reliability` Python package

**API Endpoints:**
```
POST /api/reliability/life-distribution    # Fit life distributions
POST /api/reliability/kaplan-meier         # Survival analysis
POST /api/reliability/cox-ph               # Cox regression
POST /api/reliability/alt                  # Accelerated life testing
POST /api/reliability/test-planning        # Reliability test design
```

---

#### Feature 2: Generalized Linear Models (GLM)
**Why:** MasterStat assumes normal response. Real data often doesn't.

| Distribution | Link Function | Use Case |
|--------------|---------------|----------|
| Poisson | Log | Count data (defects, events) |
| Binomial | Logit | Binary outcomes (pass/fail) |
| Negative Binomial | Log | Overdispersed counts |
| Gamma | Log/Inverse | Positive continuous (time, cost) |
| Beta | Logit | Proportions/rates (0-1 bounded) |

**Implementation:**
- Backend: New `backend/app/api/glm.py` (~800 lines)
- Frontend: New `frontend/src/pages/GLM.jsx` with distribution selector
- Components: Deviance residuals, dispersion diagnostics, link function plots

**API Endpoints:**
```
POST /api/glm/fit                          # Fit GLM model
POST /api/glm/predict                      # Predictions with CI
POST /api/glm/diagnostics                  # Deviance, Pearson residuals
POST /api/glm/compare                      # Model comparison (AIC, BIC)
```

---

#### Feature 3: Custom Design Platform
**Why:** Essential DOE capability. Enables optimal designs with constraints.

| Component | Description |
|-----------|-------------|
| D-Optimal Algorithm | Maximize \|X'X\| for parameter estimation |
| I-Optimal Algorithm | Minimize average prediction variance |
| Constraint Handling | Linear/nonlinear constraints on factors |
| Disallowed Combinations | Exclude infeasible factor settings |
| Hard-to-Change Factors | Automatic split-plot structure |
| Design Evaluation | Compare efficiency, power, aliasing |
| Augment Designs | Add runs optimally to existing experiments |

**Implementation:**
- Backend: Enhance `optimal_designs.py` + `coordinate_exchange.py` (~1,200 lines added)
- Frontend: New `frontend/src/pages/CustomDesign.jsx` wizard-style interface
- Components: ConstraintBuilder, DesignEvaluator, AugmentDesignPanel

**API Endpoints:**
```
POST /api/custom-design/generate           # Generate optimal design
POST /api/custom-design/evaluate           # Evaluate design properties
POST /api/custom-design/augment            # Add runs to existing design
POST /api/custom-design/compare            # Compare multiple designs
```

---

### Tier 3B: Professional Quality Tools (High Priority)

#### Feature 4: Enhanced SPC & Measurement Systems Analysis
**Why:** Essential for Six Sigma, manufacturing quality, FDA compliance.

| Component | Description |
|-----------|-------------|
| Control Chart Builder | Drag-and-drop chart creation |
| Advanced Charts | CUSUM, EWMA, moving range, multivariate |
| Western Electric Rules | Automatic out-of-control detection |
| Process Capability | Cp, Cpk, Pp, Ppk, Cpm with confidence intervals |
| Gauge R&R | Crossed and nested MSA studies |
| Attribute Agreement | Kappa, Kendall's W for categorical data |

**Implementation:**
- Backend: Enhance `quality_control.py` + new `msa.py` (~1,000 lines)
- Frontend: Enhance `QualityControl.jsx` + new `GaugeRR.jsx`
- Components: ControlChartBuilder, CapabilityReport, MSAStudy

**API Endpoints:**
```
POST /api/spc/control-chart                # Generate control chart
POST /api/spc/capability                   # Process capability analysis
POST /api/msa/gauge-rr                     # Gauge R&R study
POST /api/msa/attribute-agreement          # Attribute agreement analysis
```

---

#### Feature 5: Predictive Modeling Suite
**Why:** Competitive necessity. Model comparison is expected by modern users.

| Method | Purpose |
|--------|---------|
| Decision Trees (CART) | Interpretable segmentation |
| Random Forest | Variable importance, robust prediction |
| Gradient Boosting | High-accuracy prediction |
| Regularized Regression | Lasso, Ridge, Elastic Net for variable selection |
| Model Screening | Run multiple methods, compare automatically |

**Implementation:**
- Backend: New `backend/app/api/predictive_modeling.py` (~1,500 lines)
- Frontend: New `frontend/src/pages/PredictiveModeling.jsx`
- Components: ModelComparison, VariableImportance, TreeVisualizer
- Dependencies: scikit-learn (already installed)

**API Endpoints:**
```
POST /api/ml/decision-tree                 # Fit decision tree
POST /api/ml/random-forest                 # Fit random forest
POST /api/ml/gradient-boosting             # Fit gradient boosting
POST /api/ml/regularized-regression        # Lasso/Ridge/Elastic Net
POST /api/ml/model-screening               # Compare all methods
POST /api/ml/variable-importance           # Feature importance
```

---

#### Feature 6: Complete Mixture Designs
**Why:** Required for formulation work (pharma, food, chemicals, materials).

| Component | Description |
|-----------|-------------|
| Simplex Designs | Centroid, lattice with augmentation |
| Extreme Vertices | For constrained mixture regions |
| Mixture + Process | Combined mixture-process experiments |
| Ternary/Quaternary Plots | Triangular contour visualization |
| Cox/Scheffé Models | Proper mixture polynomial models |
| Trace Plots | Effect of changing one component |

**Implementation:**
- Backend: Enhance existing or new `mixture.py` (~800 lines)
- Frontend: Significantly enhance `MixtureDesign.jsx`
- Components: TernaryPlot, MixtureContour, TracePlot, ConstraintRegion

**API Endpoints:**
```
POST /api/mixture/simplex-design           # Generate simplex design
POST /api/mixture/extreme-vertices         # Extreme vertices design
POST /api/mixture/fit-model                # Fit Scheffé model
POST /api/mixture/contour                  # Ternary contour data
POST /api/mixture/trace                    # Component trace plots
```

---

### Tier 3C: Advanced Visualization System (Medium Priority)

#### Feature 7: Graph Builder
**Why:** Essential UX feature. Drag-and-drop democratizes visualization.

| Component | Description |
|-----------|-------------|
| Drag-and-Drop Zones | X, Y, Group, Color, Size, Label dropzones |
| Chart Types | Bar, line, scatter, box, histogram, heatmap, contour |
| Dynamic Statistics | Mean, median, CI overlays on demand |
| Faceting | Small multiples by categorical variable |
| Save Presets | Reusable chart configurations |

**Implementation:**
- Frontend: New `frontend/src/pages/GraphBuilder.jsx` (~1,500 lines)
- Components: DropZone, ChartTypeSelector, StatisticsOverlay, FacetPanel
- State: Complex drag-and-drop state management

---

#### Feature 8: Dynamic Linking & Interactivity
**Why:** Exploration requires connected views.

| Component | Description |
|-----------|-------------|
| Brushing | Select points in one plot, highlight everywhere |
| Linked Data Tables | Click row → highlight in all plots |
| Profiler Enhancements | Monte Carlo simulation, optimization traces |
| Animated Sliders | Watch surfaces change as factors vary |
| Drill-Down | Click aggregate → see underlying detail |

**Implementation:**
- Frontend: New shared selection context across components
- Components: Enhance all existing plot components with selection sync
- State: Global selection store (React Context or Zustand)

---

#### Feature 9: Publication-Quality Export
**Why:** Researchers need journal-ready figures.

| Component | Description |
|-----------|-------------|
| Vector Export | SVG, EPS, PDF at any resolution |
| Theme Presets | Journal styles (Nature, IEEE, APA, custom) |
| Editable Annotations | Titles, labels, legends with font control |
| Batch Export | Export all plots from an analysis |
| Office Integration | PowerPoint/Word with editable objects |

**Implementation:**
- Backend: Enhance `report_generator.py`
- Frontend: New ExportDialog with theme/format selector
- Dependencies: Consider `python-pptx`, `python-docx`

---

### Tier 3D: Extended Capabilities (Lower Priority)

#### Feature 10: Space-Filling Designs
**Why:** Growing demand for computer experiments and simulation.

| Design Type | Use Case |
|-------------|----------|
| Latin Hypercube | Uniform coverage of design space |
| Sphere Packing | Maximize minimum distance between points |
| Maximum Entropy | Optimal for Gaussian Process models |
| Sobol/Halton Sequences | Quasi-random for high dimensions |

**Implementation:**
- Backend: Add to `optimal_designs.py` (~400 lines)
- Frontend: Add options to `OptimalDesigns.jsx`
- Dependencies: `scipy.stats.qmc` (already available)

---

#### Feature 11: Time Series Analysis
**Why:** Process monitoring, forecasting, trend analysis.

| Component | Description |
|-----------|-------------|
| ARIMA Modeling | Autoregressive integrated moving average |
| Seasonal Decomposition | Trend, seasonal, residual components |
| Autocorrelation | ACF, PACF plots for model identification |
| Forecasting | Prediction with confidence intervals |
| Spectral Analysis | Frequency domain analysis |

**Implementation:**
- Backend: New `backend/app/api/time_series.py` (~700 lines)
- Frontend: New `frontend/src/pages/TimeSeries.jsx`
- Dependencies: statsmodels (already installed)

---

#### Feature 12: Professional Reporting System
**Why:** Enterprise adoption requires automation and sharing.

| Component | Description |
|-----------|-------------|
| Template Builder | Drag sections, customize layout |
| Interactive HTML | Web-based shareable reports |
| Batch Generation | Generate reports for multiple analyses |
| Branding | Logo, colors, fonts customization |
| Scheduled Reports | Automate recurring analyses |

**Implementation:**
- Backend: Significantly enhance `report_generator.py`
- Frontend: New `frontend/src/pages/ReportBuilder.jsx`
- Dependencies: Jinja2 templates, WeasyPrint or similar

---

### Implementation Estimates

| Phase | Features | Backend | Frontend | Total |
|-------|----------|---------|----------|-------|
| **3A** | Reliability, GLM, Custom Design | ~3,500 lines | ~3,000 lines | ~6,500 |
| **3B** | SPC/MSA, Predictive, Mixtures | ~3,300 lines | ~2,500 lines | ~5,800 |
| **3C** | Graph Builder, Linking, Export | ~500 lines | ~3,500 lines | ~4,000 |
| **3D** | Space-Filling, Time Series, Reports | ~1,500 lines | ~2,000 lines | ~3,500 |
| **Total** | All Tier 3 | ~8,800 lines | ~11,000 lines | **~19,800** |

---

### Feature Summary

**Currently Implemented:**
- DOE: CCD, Box-Behnken, DSD, Plackett-Burman, Factorial ✓
- RSM with multi-response optimization ✓
- ANOVA (1-way, 2-way, ANCOVA) ✓
- Mixed Models (split-plot, nested) ✓
- Model Validation (PRESS, k-fold CV) ✓
- Basic SPC and control charts ✓
- Session persistence ✓
- **Reliability/Survival Analysis (Weibull, Kaplan-Meier, Cox PH, ALT) ✓** *(NEW)*

**Critical priorities (Tier 3A):**
- ~~Reliability/Survival Analysis~~ ✓ COMPLETE
- Generalized Linear Models (Poisson, logistic, gamma)
- Custom Design Platform (D/I-optimal with constraints)

**Important priorities (Tier 3B):**
- Gauge R&R / MSA
- Machine Learning (trees, forests, boosting)
- Complete mixture designs with ternary plots

**Future enhancements (Tier 3C/D):**
- Graph Builder (drag-and-drop)
- Dynamic plot linking
- Space-filling designs
- Time series analysis
- Advanced reporting/automation

---

### Known Limitations (Current)
- Mixed models validation uses marginal fixed effects (not full conditional residuals)
- Multi-response contour limited to 2 factors
- IndexedDB has browser quota limits (~50MB+, sufficient for most use)
- Linux builds are arm64 only (x64 would need separate build environment)

---

## Troubleshooting

### Backend won't start
```bash
# Check if port 8000 is in use
lsof -i :8000
# Kill process if needed
kill -9 <PID>
```

### Frontend build errors
```bash
cd frontend
rm -rf node_modules
npm install
npm run build
```

### Electron app shows blank screen
- Ensure backend is running on port 8000
- Check backend health: `curl http://localhost:8000/health`
- Backend serves frontend static files from `frontend/dist`

### Session data not persisting
- IndexedDB is browser-specific
- Check browser dev tools > Application > IndexedDB > MasterStatDB
- Dexie.js handles all persistence automatically

---

## Summary for Claude Code

**This is a complete, working statistical analysis application with Tier 3 planned.**

### Current State (as of 2026-01-31)
1. **Tier 2 is 100% complete** - All 4 features implemented and tested
2. **Tier 3 Feature 1 (Reliability/Survival) is COMPLETE** - 11 features remaining
3. **Electron builds are current** (Jan 28, 2026) for all platforms
4. **Codebase:** ~76,400 lines across 146 files

### Tier 3 Quick Reference
| Priority | Feature | Status |
|----------|---------|--------|
| 1st | Reliability/Survival Analysis | **COMPLETE** |
| 2nd | Generalized Linear Models (GLM) | NOT STARTED |
| 3rd | Custom Design Platform | NOT STARTED |
| 4th | Enhanced SPC/MSA | NOT STARTED |
| 5th | Predictive Modeling Suite | NOT STARTED |
| 6th | Complete Mixture Designs | NOT STARTED |
| 7th | Graph Builder | NOT STARTED |
| 8th | Dynamic Linking | NOT STARTED |
| 9th | Space-Filling Designs | NOT STARTED |
| 10th | Time Series Analysis | NOT STARTED |
| 11th | Professional Reporting | NOT STARTED |
| 12th | Publication-Quality Export | NOT STARTED |

### To Continue Tier 3 Development
1. **Feature 1 (Reliability) is COMPLETE** - See files: `reliability.py`, `ReliabilityAnalysis.jsx`
2. **Next: Feature 2 (GLM)** - See detailed spec in "Tier 3A: Critical Statistical Gaps" section
3. Create backend module: `backend/app/api/glm.py`
4. Create frontend page: `frontend/src/pages/GLM.jsx`
5. statsmodels GLM support already available (no new dependencies needed)

### Development Checklist for New Features
```bash
# 1. Backend first
cd backend/app/api
# Create new module following existing patterns (see rsm.py as reference)

# 2. Register routes in main.py
# Add: from app.api import new_module
# Add: app.include_router(new_module.router, prefix="/api/new-module")

# 3. Frontend page
cd frontend/src/pages
# Create new page component following existing patterns (see RSM.jsx)

# 4. Add route in App.jsx
# Add: import NewPage from './pages/NewPage'
# Add: <Route path="/new-page" element={<NewPage />} />

# 5. Add navigation in App.jsx sidebar menuItems array

# 6. **IMPORTANT: Add frontpage card in Home.jsx**
# Add new entry to the `features` array with:
#   - icon: appropriate lucide-react icon
#   - title: feature name
#   - description: brief description
#   - path: route path
#   - color: gradient colors (e.g., 'from-blue-400 to-blue-600')

# 7. Test
npm run build  # Frontend
python -m uvicorn app.main:app --reload  # Backend

# 8. Update CLAUDE.md
# - Update codebase statistics (file counts, line counts)
# - Update Tier 3 Quick Reference status
# - Add new endpoints to API Endpoints section
# - Add new files to Key Files Reference

# 9. Build Electron (after feature complete)
npm run dist
```

**CRITICAL:** Every new feature MUST have a corresponding card on the Home page (`frontend/src/pages/Home.jsx`). This ensures users can discover and access all features from the main dashboard.

### Code Pattern Reminders
- **Backend:** FastAPI + Pydantic models + try/except with HTTPException
- **Frontend:** React hooks + axios + Tailwind dark mode classes
- **Plots:** Plotly.js with dark theme (`paper_bgcolor: '#1e293b'`)
- **State:** React useState/useEffect, SessionContext for persistence

### Key Dependencies for Tier 3
```bash
# Backend (in requirements.txt)
lifelines>=0.27.0          # Survival analysis (Feature 1) - INSTALLED
# scikit-learn already installed (Feature 5)
# scipy.stats.qmc available (Feature 10)
# statsmodels already installed (Feature 2 GLM, Feature 11)

# Frontend (in package.json)
# Most visualization needs covered by existing Plotly/Recharts
```

### Files to Reference
- `TIER2_STATUS.md` - Detailed Tier 2 implementation notes
- `backend/app/api/rsm.py` - Best backend pattern reference (4,600 lines)
- `frontend/src/pages/RSM.jsx` - Best frontend pattern reference (2,863 lines)
- `frontend/src/utils/sessionManager.js` - Session persistence pattern
