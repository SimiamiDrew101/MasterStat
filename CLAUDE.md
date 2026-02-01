# MasterStat - Project Context for Claude Code

**Last Updated:** 2026-02-01
**Status:** Tier 3 Complete | JMP Pro 16 Feature Parity Achieved

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

**MasterStat** is a professional statistical analysis desktop application (JMP alternative) providing:
- Design of Experiments (DOE): CCD, Box-Behnken, DSD, Plackett-Burman, Factorial, Custom Optimal
- Statistical Analysis: ANOVA, RSM, Mixed Models, GLM, Nonlinear Regression
- Reliability/Survival Analysis: Kaplan-Meier, Cox PH, Weibull, ALT
- Predictive Modeling: Decision Trees, Random Forest, Gradient Boosting, Regularized Regression
- Quality Control: SPC, MSA, Gauge R&R, Control Charts
- Multi-Response Optimization with desirability functions
- Interactive Visualization: Graph Builder, Linked Data Explorer
- Session persistence with IndexedDB
- Cross-platform desktop builds (macOS, Windows, Linux)

**Tech Stack:**
| Layer | Technologies |
|-------|-------------|
| Backend | Python 3.11+ / FastAPI 0.109 / Uvicorn |
| Frontend | React 18.2 / Vite 5.0 / TailwindCSS 3.4 |
| Desktop | Electron 28.1 (cross-platform) |
| Visualization | Plotly.js 3.3 / Recharts 2.10 |
| Statistics | SciPy / statsmodels / NumPy / pandas / pyDOE2 / lifelines / scikit-learn |
| Persistence | IndexedDB via Dexie.js 4.2 |

---

## Implementation Status

### Tier 1: COMPLETE
All core statistical analysis features working.

### Tier 2: COMPLETE (2026-01-28)

| Feature | Status | Key Files |
|---------|--------|-----------|
| **Experiment Wizard** | COMPLETE | `rsm.py`, `DesignPreviewVisualization.jsx`, `PowerCurvePlot.jsx`, `ConfoundingDiagram.jsx` |
| **Model Validation** | COMPLETE | `model_validation.py`, `ModelValidation.jsx` |
| **Multi-Response Optimization** | COMPLETE | `rsm.py`, `OverlayContourPlot.jsx` |
| **Session Management** | COMPLETE | `sessionManager.js`, `SessionContext.jsx`, `SessionHistory.jsx` |

### Tier 3: COMPLETE (2026-02-01)

| Feature | Status | Backend | Frontend |
|---------|--------|---------|----------|
| **1. Reliability/Survival Analysis** | COMPLETE | `reliability.py` (1,175 lines) | `ReliabilityAnalysis.jsx` (1,528 lines) |
| **2. Generalized Linear Models** | COMPLETE | `glm.py` (981 lines) | `GLM.jsx` (1,250 lines) |
| **3. Custom Design Platform** | COMPLETE | `custom_design.py` (980 lines) | `CustomDesign.jsx` (900 lines) |
| **4. Enhanced SPC/MSA** | COMPLETE | `quality_control.py` (945 lines), `msa.py` (793 lines) | `QualityControl.jsx` (2,020 lines) |
| **5. Predictive Modeling Suite** | COMPLETE | `predictive_modeling.py` (931 lines) | `PredictiveModeling.jsx` (1,265 lines) |
| **6. Complete Mixture Designs** | COMPLETE | `mixture.py` (866 lines) | `MixtureDesign.jsx` (1,345 lines) |
| **7. Graph Builder** | COMPLETE | - | `GraphBuilder.jsx` (900 lines) |
| **8. Dynamic Linking & Interactivity** | COMPLETE | - | `LinkedDataExplorer.jsx` (737 lines), `SelectionContext.jsx` |

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
│   │   ├── api/                 # 24 API modules
│   │   ├── utils/               # model_validation.py, report_generator.py
│   │   └── main.py              # FastAPI entry point
│   └── requirements.txt
├── frontend/                     # React/Vite web interface
│   ├── src/
│   │   ├── pages/               # 24 page components
│   │   ├── components/          # 90+ reusable components
│   │   ├── contexts/            # SessionContext.jsx, SelectionContext.jsx
│   │   └── utils/               # sessionManager.js, smartValidation.js, etc.
│   └── package.json
├── electron/                     # Electron desktop wrapper
│   ├── main.js                  # Main process (starts backend, creates window)
│   └── preload.js               # Security layer
├── build/                        # Build resources (icons, entitlements)
├── dist-electron/                # Built installers for all platforms
├── CLAUDE.md                     # This file
└── package.json                  # Root Electron/build config
```

---

## Codebase Statistics

| Area | Files | Lines |
|------|-------|-------|
| Backend (Python) | 24 modules | ~25,400 |
| Frontend Pages (React) | 24 | ~28,400 |
| Frontend Components | 90+ | ~29,100 |
| Electron | 2 | ~240 |
| **Total** | **~140** | **~93,000** |

### Backend API Modules (24 total)

| Module | Lines | Purpose |
|--------|-------|---------|
| `rsm.py` | 4,605 | RSM, CCD, Box-Behnken, DSD, desirability, overlay contours |
| `mixed_models.py` | 2,529 | Split-plot, nested, random effects, BLUPs |
| `factorial.py` | 2,219 | 2^k, 2^(k-p), confounding, aliasing |
| `power_analysis.py` | 1,721 | Sample size, power curves, effect size |
| `anova.py` | 1,698 | 1-way, 2-way, ANCOVA, post-hoc tests |
| `block_designs.py` | 1,584 | RCBD, Latin squares, BIBD |
| `reliability.py` | 1,175 | Kaplan-Meier, Cox PH, Weibull, ALT |
| `glm.py` | 981 | Poisson, Binomial, Gamma, Negative Binomial |
| `custom_design.py` | 980 | D/I/A-optimal designs with constraints |
| `quality_control.py` | 945 | SPC, control charts, process capability |
| `predictive_modeling.py` | 931 | Trees, Random Forest, Boosting, Lasso/Ridge |
| `mixture.py` | 866 | Simplex, extreme vertices, ternary, Scheffé |
| `msa.py` | 793 | Gauge R&R, attribute agreement |
| `hypothesis_testing.py` | 692 | t-tests, F-tests, Z-tests |
| `nonlinear_regression.py` | 651 | Curve fitting, validation |
| `bayesian_doe.py` | 542 | Bayesian inference, sequential |
| `preprocessing.py` | 497 | Outlier detection, transformation |
| `imputation.py` | 562 | MICE, KNN, missing data |
| `protocol.py` | 444 | Randomization, blinding, PDF |
| `import_data.py` | 306 | CSV, Excel, JSON parsing |
| `optimal_designs.py` | 225 | D/I/A-optimal designs |
| `prediction_profiler.py` | 217 | Sensitivity analysis |
| `stats.py` | ~100 | Basic statistics utilities |

### Frontend Pages (24 total)

| Page | Lines | Purpose |
|------|-------|---------|
| `RSM.jsx` | 2,863 | Response Surface Methodology |
| `FactorialDesigns.jsx` | 2,262 | Full/fractional factorial |
| `ExperimentPlanning.jsx` | 2,148 | Power, sample size |
| `QualityControl.jsx` | 2,020 | SPC, control charts, MSA |
| `MixedModels.jsx` | 1,745 | Mixed models analysis |
| `ANOVA.jsx` | 1,636 | ANOVA analysis |
| `ReliabilityAnalysis.jsx` | 1,528 | Survival, Weibull, Cox |
| `MixtureDesign.jsx` | 1,345 | Simplex, ternary plots |
| `ExperimentWizardPage.jsx` | 1,307 | Guided design creation |
| `PredictiveModeling.jsx` | 1,265 | ML methods comparison |
| `GLM.jsx` | 1,250 | Generalized linear models |
| `BayesianDOE.jsx` | 1,195 | Bayesian analysis |
| `BlockDesigns.jsx` | 1,166 | Block design analysis |
| `CustomDesign.jsx` | 900 | Optimal design builder |
| `GraphBuilder.jsx` | 900 | Drag-and-drop visualization |
| `LinkedDataExplorer.jsx` | 737 | Linked plots with brushing |
| Plus 8 more pages... | | |

---

## API Endpoints Reference

### Design Generation
```
POST /api/rsm/ccd/generate          # Central Composite Design
POST /api/rsm/box-behnken/generate  # Box-Behnken Design
POST /api/rsm/dsd/generate          # Definitive Screening Design
POST /api/rsm/plackett-burman/generate  # Plackett-Burman screening
POST /api/factorial/generate        # Factorial designs
POST /api/custom-design/generate    # D/I/A-optimal designs
POST /api/mixture/simplex-lattice   # Simplex lattice mixture
POST /api/mixture/extreme-vertices/generate  # Extreme vertices
```

### Analysis
```
POST /api/anova/analyze             # ANOVA analysis
POST /api/rsm/fit-model             # RSM model fitting
POST /api/mixed-models/analyze      # Mixed model analysis
POST /api/glm/fit                   # GLM fitting
POST /api/glm/predict               # GLM predictions
POST /api/predictive-modeling/decision-tree
POST /api/predictive-modeling/random-forest
POST /api/predictive-modeling/gradient-boosting
POST /api/predictive-modeling/regularized-regression
POST /api/predictive-modeling/model-comparison
```

### Reliability/Survival Analysis
```
POST /api/reliability/life-distribution   # Weibull, lognormal, exponential
POST /api/reliability/kaplan-meier        # Survival curves + log-rank test
POST /api/reliability/cox-ph              # Cox Proportional Hazards
POST /api/reliability/alt                 # Accelerated Life Testing
POST /api/reliability/test-planning       # Reliability test sample size
```

### Quality Control & MSA
```
POST /api/quality-control/control-chart   # X-bar, R, S, P, C charts
POST /api/quality-control/capability      # Cp, Cpk, Pp, Ppk
POST /api/quality-control/cusum           # CUSUM charts
POST /api/quality-control/ewma            # EWMA charts
POST /api/msa/gauge-rr                    # Gauge R&R studies
POST /api/msa/attribute-agreement         # Kappa analysis
```

### Mixture Designs
```
POST /api/mixture/simplex-lattice         # Simplex lattice
POST /api/mixture/simplex-centroid        # Simplex centroid
POST /api/mixture/extreme-vertices/generate
POST /api/mixture/mixture-process/generate
POST /api/mixture/ternary-contour
POST /api/mixture/trace-plot
```

### Multi-Response Optimization
```
POST /api/rsm/desirability-optimization  # Composite desirability
POST /api/rsm/multi-response-contour     # Overlay contour data
```

### Model Validation
```
POST /api/anova/validate-model
POST /api/factorial/validate-model
POST /api/mixed-models/validate-model
POST /api/nonlinear-regression/validate-model
POST /api/rsm/confounding-analysis
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

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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

### Selection Context Pattern (Linked Views)
```jsx
import { useSelection } from '../contexts/SelectionContext';

const LinkedPlot = () => {
  const { selectedIndices, toggleIndex, selectByBrush, clearSelection } = useSelection();

  // Selection syncs across all linked components
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
  font: { color: '#e2e8f0' },
  xaxis: { gridcolor: '#475569' },
  yaxis: { gridcolor: '#475569' }
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

## Feature Summary

### JMP Pro 16 Parity Features (All Implemented)

| Category | Features |
|----------|----------|
| **DOE** | CCD, Box-Behnken, DSD, Plackett-Burman, Full/Fractional Factorial, Custom D/I/A-Optimal, Mixture Designs |
| **Analysis** | ANOVA, RSM, Mixed Models, GLM, Nonlinear Regression, Bayesian DOE |
| **Reliability** | Kaplan-Meier, Cox PH, Weibull/Lognormal/Exponential, ALT, Test Planning |
| **Quality** | X-bar/R/S/P/C Charts, CUSUM, EWMA, Cp/Cpk/Pp/Ppk, Gauge R&R |
| **ML/Predictive** | Decision Trees, Random Forest, Gradient Boosting, Lasso/Ridge/ElasticNet |
| **Visualization** | 3D Surfaces, Contours, Overlay Plots, Graph Builder, Linked Explorer, Ternary Plots |
| **Optimization** | Desirability Functions, Multi-Response, Overlay Contours |
| **Validation** | PRESS, k-fold CV, Adequacy Scoring, Diagnostic Tests |

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

## Dependencies

### Backend (requirements.txt)
```
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic>=2.5.0
numpy>=1.26.0
scipy>=1.12.0
pandas>=2.1.0
statsmodels>=0.14.0
scikit-learn>=1.4.0
pyDOE2>=1.3.0
lifelines>=0.27.0
reportlab>=4.0.0
openpyxl>=3.1.0
python-multipart>=0.0.6
```

### Frontend (package.json)
```json
{
  "react": "^18.2.0",
  "react-router-dom": "^6.21.0",
  "axios": "^1.6.0",
  "plotly.js": "^2.27.0",
  "react-plotly.js": "^2.6.0",
  "recharts": "^2.10.0",
  "dexie": "^4.0.1",
  "lucide-react": "^0.303.0",
  "tailwindcss": "^3.4.0"
}
```

---

## Summary for Claude Code

**This is a complete, production-ready statistical analysis application with JMP Pro 16 feature parity.**

### Current State (as of 2026-02-01)
1. **All tiers complete** - Tier 1, 2, and 3 fully implemented
2. **93,000+ lines of code** across 140+ files
3. **24 backend API modules** with comprehensive statistical methods
4. **24 frontend pages** with modern React/TailwindCSS
5. **Electron builds** for macOS, Windows, and Linux

### Key Capabilities
- Full DOE suite (CCD, Box-Behnken, DSD, Factorial, Custom Optimal, Mixture)
- Comprehensive analysis (ANOVA, RSM, GLM, Mixed Models, Reliability, Predictive)
- Advanced visualization (Graph Builder, Linked Explorer, 3D surfaces)
- Multi-response optimization with desirability functions
- Session persistence and export

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
# Create new page component (see RSM.jsx as reference)

# 4. Add route in App.jsx
# Add: import NewPage from './pages/NewPage'
# Add: <Route path="/new-page" element={<NewPage />} />

# 5. Add navigation in App.jsx sidebar and Home.jsx grid

# 6. Test
npm run dev  # Frontend
python -m uvicorn app.main:app --reload  # Backend

# 7. Build Electron (after feature complete)
npm run dist
```
