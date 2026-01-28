# MasterStat - Project Context

## Overview

**MasterStat** is a professional statistical analysis desktop application designed as a JMP alternative. It provides comprehensive Design of Experiments (DOE), statistical analysis, and optimization capabilities through an intuitive interface.

**Tech Stack:**
- **Backend:** Python 3.11+ / FastAPI 0.109.0 / Uvicorn
- **Frontend:** React 18.2 / Vite 5.0 / TailwindCSS 3.4
- **Desktop:** Electron 28.1.0 (cross-platform)
- **Visualization:** Plotly.js 3.3, Recharts 2.10
- **Statistics:** SciPy, statsmodels, NumPy, pandas, scikit-learn, pyDOE2

---

## Directory Structure

```
MasterStat/
├── backend/                    # Python/FastAPI statistical engine
│   ├── app/
│   │   ├── api/               # API route modules (14 modules)
│   │   ├── models/            # Data models
│   │   ├── utils/             # Utilities (validation, reports)
│   │   └── main.py            # FastAPI app entry
│   └── requirements.txt
├── frontend/                   # React/Vite web interface
│   ├── src/
│   │   ├── pages/             # 14+ page components
│   │   ├── components/        # 87+ reusable components
│   │   └── utils/             # 15+ utility functions
│   ├── package.json
│   └── vite.config.js
├── electron/                   # Electron desktop wrapper
│   ├── main.js                # Main process
│   └── preload.js             # Security layer
├── build/                      # Build resources (icons)
├── dist/                       # Production builds
├── TIER2_STATUS.md            # Detailed implementation status
└── package.json               # Root npm config
```

---

## Backend API Reference

### API Modules (`/backend/app/api/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `rsm.py` | 4,605 | Response Surface Methodology, CCD, Box-Behnken, desirability optimization |
| `mixed_models.py` | 2,529 | Split-plot, nested designs, random effects ANOVA |
| `factorial.py` | 2,219 | Full/fractional factorial, Plackett-Burman, DSD |
| `power_analysis.py` | 1,721 | Sample size calculation, power curves |
| `anova.py` | 1,698 | One-way/two-way ANOVA, post-hoc tests (Tukey, Dunnett) |
| `block_designs.py` | 1,584 | RCBD, Latin squares, incomplete blocks |
| `quality_control.py` | 639 | Control charts (X-bar, R, S, P, C), process capability |
| `hypothesis_testing.py` | 692 | t-tests, F-tests, Z-tests, chi-square, non-parametric |
| `nonlinear_regression.py` | 651 | Curve fitting, growth models, convergence analysis |
| `imputation.py` | 562 | Missing data: mean, median, interpolation, MICE |
| `bayesian_doe.py` | 542 | MCMC, convergence diagnostics, posterior optimization |
| `preprocessing.py` | 497 | Data cleaning, transformation, outlier detection |
| `protocol.py` | 444 | Randomization, blinding, PDF protocol export |
| `import_data.py` | 306 | CSV/Excel file parsing |

### Utility Modules (`/backend/app/utils/`)

| Module | Purpose |
|--------|---------|
| `model_validation.py` | PRESS statistic, k-fold CV, adequacy assessment (Tier 2) |
| `report_generator.py` | PDF report generation (ReportLab) |

### Key Endpoints by Feature

**Design of Experiments:**
- `POST /api/rsm/ccd/generate` - Central Composite Design
- `POST /api/rsm/box-behnken/generate` - Box-Behnken Design
- `POST /api/rsm/dsd/generate` - Definitive Screening Design
- `POST /api/rsm/plackett-burman/generate` - Plackett-Burman screening
- `POST /api/factorial/generate` - Factorial designs

**Analysis:**
- `POST /api/anova/analyze` - ANOVA analysis
- `POST /api/rsm/fit-model` - RSM model fitting
- `POST /api/mixed-models/analyze` - Mixed model analysis
- `POST /api/rsm/desirability-optimization` - Multi-response optimization

**Validation (Tier 2):**
- `POST /api/anova/validate-model` - ANOVA validation
- `POST /api/factorial/validate-model` - Factorial validation
- `POST /api/mixed-models/validate-model` - Mixed model validation
- `POST /api/nonlinear-regression/validate-model` - Nonlinear validation
- `POST /api/rsm/confounding-analysis` - Alias structure analysis
- `POST /api/rsm/multi-response-contour` - Overlay contour plots

---

## Frontend Reference

### Pages (`/frontend/src/pages/`)

| Page | Purpose |
|------|---------|
| `Home.jsx` | Landing page with feature cards |
| `ExperimentWizardPage.jsx` | Guided experiment design workflow |
| `ExperimentPlanning.jsx` | Sample size and power analysis |
| `DataPreprocessing.jsx` | Data cleaning, transformation, imputation |
| `HypothesisTesting.jsx` | Statistical testing interface |
| `ANOVA.jsx` | ANOVA analysis |
| `FactorialDesigns.jsx` | Factorial experiment design |
| `BlockDesigns.jsx` | Block design generation |
| `MixedModels.jsx` | Mixed model analysis |
| `RSM.jsx` | Response Surface Methodology (largest: 121KB) |
| `MixtureDesign.jsx` | Mixture experiment design |
| `RobustDesign.jsx` | Taguchi methods |
| `BayesianDOE.jsx` | Bayesian experimental design |
| `PredictionProfiler.jsx` | Model prediction profiling |
| `OptimalDesigns.jsx` | Optimal design generation |
| `NonlinearRegression.jsx` | Curve fitting |
| `QualityControl.jsx` | Control charts and capability |
| `ProtocolGeneratorPage.jsx` | Protocol PDF generation |

### Key Components (`/frontend/src/components/`)

**Visualization:**
- `ResponseSurface3D.jsx` - 3D surface plots
- `ContourPlot.jsx` - 2D contour plots
- `MainEffectsPlot.jsx`, `InteractionPlot.jsx` - Effect plots
- `CubePlot.jsx`, `HalfNormalPlot.jsx` - Design visualization
- `DiagnosticPlots.jsx`, `ResidualPlots.jsx` - Model diagnostics
- `CorrelationHeatmap.jsx`, `ScatterMatrix.jsx` - Exploratory plots

**Tier 2 Components (NEW):**
- `DesignPreviewVisualization.jsx` - Interactive design preview (318 lines)
- `PowerCurvePlot.jsx` - Statistical power curves (228 lines)
- `ConfoundingDiagram.jsx` - Alias structure display (292 lines)
- `ModelValidation.jsx` - Model adequacy assessment (685 lines)

**Data Management:**
- `FileUploadZone.jsx` - File import
- `ExcelTable.jsx` - Excel-like data editing
- `ColumnPreprocessor.jsx` - Column transformation
- `OutlierDetection.jsx` - Outlier detection UI
- `MissingDataPanel.jsx` - Imputation options

**Analysis:**
- `ExperimentWizard.jsx` - Step-by-step design guidance
- `ProtocolGenerator.jsx` - PDF protocol generation
- `MultiResponseManager.jsx` - Multi-response optimization

### Utilities (`/frontend/src/utils/`)

| Utility | Purpose |
|---------|---------|
| `smartValidation.js` | Input validation with business logic |
| `doeGlossary.js` | 50+ statistical terms with definitions |
| `designExport.js` | Export designs to various formats |
| `fileParser.js` | CSV/Excel parsing |
| `clipboardParser.js` | Clipboard data import |
| `plotlyConfig.js` | Plotly dark theme configuration |
| `randomization.js` | Randomization algorithms |

---

## Development Commands

### Backend
```bash
cd /Users/nj/Desktop/MasterStat/backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend
```bash
cd /Users/nj/Desktop/MasterStat/frontend
npm install          # Install dependencies
npm run dev          # Development server (localhost:5173)
npm run build        # Production build
```

### Electron (Desktop App)
```bash
cd /Users/nj/Desktop/MasterStat
npm run electron     # Run desktop app
npm run dist         # Build distributables
```

### Cross-Platform Builds
```bash
npm run build:mac    # macOS (DMG + ZIP)
npm run build:win    # Windows (NSIS installer)
npm run build:linux  # Linux (AppImage + deb + rpm)
```

### Testing Endpoints
```bash
curl -X POST http://localhost:8000/api/rsm/dsd/generate \
  -H "Content-Type: application/json" \
  -d '{"n_factors": 3, "factor_names": ["A", "B", "C"], "center_points": 3}'
```

---

## Implementation Status

### Tier 1: COMPLETE
Core statistical analysis features including RSM, ANOVA, Factorial, Mixed Models, Quality Control, Prediction Profiler, Optimal Designs, Nonlinear Regression.

### Tier 2: 60% COMPLETE (Target: 80-90% JMP Parity)

| Feature | Status | Details |
|---------|--------|---------|
| **Feature 1: Experiment Wizard** | 100% COMPLETE | DSD, Plackett-Burman, confounding analysis, design preview, power curves |
| **Feature 2: Model Validation** | 100% COMPLETE | PRESS statistic, k-fold CV, adequacy score, diagnostic tests |
| **Feature 3: Multi-Response Optimization** | 50% COMPLETE | Backend done (compositing methods, overlay contours), frontend pending |
| **Feature 4: Session Management** | 0% NOT STARTED | IndexedDB persistence, save/load sessions, export/import |

**See `TIER2_STATUS.md` for detailed implementation notes, code locations, and next steps.**

---

## Code Patterns & Conventions

### Backend Patterns

**FastAPI Endpoint:**
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

**Model Validation Pattern:**
```python
from app.utils.model_validation import full_model_validation
validation = full_model_validation(model, df, response_var, k_folds=5)
```

### Frontend Patterns

**React Component:**
```jsx
import React, { useState, useEffect } from 'react';
import { Icon } from 'lucide-react';

const ComponentName = ({ prop1, onCallback }) => {
  const [state, setState] = useState(null);

  useEffect(() => {
    // Side effects
  }, [dependencies]);

  return (
    <div className="bg-slate-800 text-slate-100 p-6 rounded-lg">
      {/* JSX */}
    </div>
  );
};

export default ComponentName;
```

**API Call Pattern:**
```jsx
const fetchData = async () => {
  setLoading(true);
  try {
    const response = await fetch('http://localhost:8000/api/endpoint', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestData)
    });
    const data = await response.json();
    setResult(data);
  } catch (err) {
    setError(err.message);
  } finally {
    setLoading(false);
  }
};
```

**Tailwind Dark Mode Classes:**
```jsx
// Container
className="bg-slate-800 text-slate-100 p-6 rounded-lg"
// Card
className="bg-slate-700 border border-slate-600 p-4"
// Input
className="bg-slate-900 text-slate-100 border border-slate-600 rounded px-3 py-2"
// Button
className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded"
// Status: green (success), yellow (warning), red (error), blue (info)
```

**Plotly Dark Theme:**
```javascript
const layout = {
  paper_bgcolor: '#1e293b',
  plot_bgcolor: '#0f172a',
  font: { color: '#e2e8f0' }
};
```

---

## Key File Locations

### Critical Backend Files
- `/backend/app/api/rsm.py` - RSM, optimization, multi-response (4,605 lines)
- `/backend/app/api/anova.py` - ANOVA analysis (1,698 lines)
- `/backend/app/api/factorial.py` - Factorial designs (2,219 lines)
- `/backend/app/utils/model_validation.py` - Validation utilities (456 lines)
- `/backend/app/main.py` - FastAPI app entry point

### Critical Frontend Files
- `/frontend/src/pages/RSM.jsx` - RSM page (largest, 121KB)
- `/frontend/src/pages/ANOVA.jsx` - ANOVA page
- `/frontend/src/pages/FactorialDesigns.jsx` - Factorial page
- `/frontend/src/components/ModelValidation.jsx` - Validation UI (685 lines)
- `/frontend/src/App.jsx` - App entry and routing

### Configuration
- `/frontend/package.json` - Frontend dependencies
- `/backend/requirements.txt` - Backend dependencies
- `/package.json` - Root Electron config
- `/frontend/tailwind.config.js` - Tailwind configuration

### Status Documents
- `/TIER2_STATUS.md` - Detailed Tier 2 implementation status
- `/CLAUDE.md` - This file (project context)

---

## Ports & URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Backend | `http://localhost:8000` | FastAPI statistical engine |
| Frontend (dev) | `http://localhost:5173` | Vite dev server |
| API Docs | `http://localhost:8000/docs` | Swagger UI |
| Health Check | `http://localhost:8000/health` | Backend status |

---

## Quick Start for New Sessions

1. **Read this file** for full project context
2. **Check `TIER2_STATUS.md`** for current implementation progress
3. **Start backend:** `cd backend && python -m uvicorn app.main:app --reload`
4. **Start frontend:** `cd frontend && npm run dev`
5. **Continue with pending features** (Feature 3 frontend, Feature 4 complete)
