"""
MasterStat Standalone Backend
Serves built React frontend and API endpoints
Designed for bundled desktop application (PyInstaller)
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import sys
import os

# Import existing routers
from app.api import (
    hypothesis_testing,
    anova,
    factorial,
    block_designs,
    mixed_models,
    rsm,
    power_analysis,
    bayesian_doe,
    import_data,
    stats,
    preprocessing,
    imputation,
    protocol
)

app = FastAPI(
    title="MasterStat",
    description="Professional-grade statistical analysis and Design of Experiments platform",
    version="1.0.0"
)

# Get path to static files (works with PyInstaller)
def get_static_path():
    """Get path to built React frontend"""
    try:
        # PyInstaller bundles files in _MEIPASS temp folder
        base_path = Path(sys._MEIPASS)
    except AttributeError:
        # Running in development mode
        base_path = Path(__file__).parent.parent.parent

    return base_path / "frontend" / "dist"

# Include API routes
app.include_router(import_data.router, prefix="/api/import", tags=["Data Import"])
app.include_router(stats.router, prefix="/api/stats", tags=["Statistics"])
app.include_router(power_analysis.router, prefix="/api/power", tags=["Power Analysis"])
app.include_router(hypothesis_testing.router, prefix="/api/hypothesis", tags=["Hypothesis Testing"])
app.include_router(anova.router, prefix="/api/anova", tags=["ANOVA"])
app.include_router(factorial.router, prefix="/api/factorial", tags=["Factorial Designs"])
app.include_router(block_designs.router, prefix="/api/block-designs", tags=["Block Designs"])
app.include_router(mixed_models.router, prefix="/api/mixed", tags=["Mixed Models"])
app.include_router(rsm.router, prefix="/api/rsm", tags=["Response Surface Methodology"])
app.include_router(bayesian_doe.router, prefix="/api/bayesian-doe", tags=["Bayesian DOE"])
app.include_router(preprocessing.router, prefix="/api/preprocessing", tags=["Data Preprocessing"])
app.include_router(imputation.router, prefix="/api/imputation", tags=["Missing Data Imputation"])
app.include_router(protocol.router, prefix="/api/protocol", tags=["Experimental Protocols"])

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "mode": "standalone"}

# Serve React static files
static_path = get_static_path()

if static_path.exists():
    # Mount assets directory for CSS, JS, fonts, etc.
    assets_path = static_path / "assets"
    if assets_path.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        """
        Serve React app for all non-API routes
        This enables client-side routing
        """
        # Try to serve the requested file
        file_path = static_path / full_path

        if file_path.is_file():
            return FileResponse(file_path)

        # Fallback to index.html for client-side routing
        index_path = static_path / "index.html"
        if index_path.exists():
            return FileResponse(index_path)

        return {"error": "Frontend not found"}

else:
    @app.get("/")
    async def root_error():
        return {
            "error": "Frontend build not found",
            "message": "The React frontend must be built before packaging.",
            "instruction": "Run 'npm run build' in the frontend directory",
            "static_path_searched": str(static_path)
        }
