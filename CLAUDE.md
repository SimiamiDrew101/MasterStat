# MasterStat - Project Context for Claude Code

**Last Updated:** 2026-01-28
**Status:** Tier 2 Complete - All 4 Features Implemented

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
│   │   ├── api/                 # 14 API modules (rsm.py, anova.py, etc.)
│   │   ├── utils/               # model_validation.py, report_generator.py
│   │   └── main.py              # FastAPI entry point
│   └── requirements.txt
├── frontend/                     # React/Vite web interface
│   ├── src/
│   │   ├── pages/               # 18 page components
│   │   ├── components/          # 90+ reusable components
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
| `backend/app/api/anova.py` | 1,700+ | ANOVA analysis with validation |
| `backend/app/api/factorial.py` | 2,200+ | Factorial designs with validation |
| `backend/app/api/mixed_models.py` | 2,500+ | Split-plot, nested, random effects |
| `backend/app/utils/model_validation.py` | 456 | PRESS, k-fold CV, adequacy scoring |

### Frontend - Most Important Files

| File | Purpose |
|------|---------|
| `frontend/src/pages/RSM.jsx` | Main RSM page (largest: ~3000 lines) |
| `frontend/src/components/OverlayContourPlot.jsx` | Multi-response contour visualization |
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

## Potential Next Steps (Future Development)

### Tier 3 Features (Not Started)
1. **Advanced Visualization** - 3D contour animations, VR/AR support
2. **Collaboration** - Multi-user sessions, cloud sync
3. **Machine Learning Integration** - AutoML for model selection
4. **Report Templates** - Customizable PDF/Word reports
5. **API Extensions** - REST API for external tool integration

### Technical Improvements
1. **Performance** - Code splitting, lazy loading large components
2. **Testing** - Unit tests, integration tests, E2E tests
3. **CI/CD** - Automated builds, GitHub Actions
4. **Documentation** - User manual, API documentation

### Known Limitations
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

**This is a complete, working statistical analysis application.**

Key things to know:
1. **Tier 2 is 100% complete** - All 4 features implemented and tested
2. **Electron builds are current** (Jan 28, 2026) for all platforms
3. **Session management works** via IndexedDB/Dexie.js
4. **Multi-response optimization** has 3 compositing methods + overlay contours
5. **Model validation** includes PRESS, k-fold CV, adequacy scoring

For any new feature work:
- Check `TIER2_STATUS.md` for detailed implementation notes
- Follow existing code patterns (FastAPI + React + Tailwind dark mode)
- Test with `npm run build` before committing
- Build Electron with `npm run dist` for releases
