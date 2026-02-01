from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from app.api import hypothesis_testing, anova, factorial, block_designs, mixed_models, rsm, power_analysis, bayesian_doe, import_data, stats, preprocessing, imputation, protocol, prediction_profiler, optimal_designs, nonlinear_regression, quality_control, reliability, glm, custom_design, msa, predictive_modeling, mixture

app = FastAPI(
    title="MasterStat - Statistical Analysis Tool",
    description="Comprehensive statistical analysis covering experimental design and ANOVA",
    version="1.0.0"
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:8000",  # Electron app
        "http://localhost:8000"    # Alternative localhost
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
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
app.include_router(prediction_profiler.router, prefix="/api/prediction-profiler", tags=["Prediction Profiler"])
app.include_router(optimal_designs.router, prefix="/api/optimal-designs", tags=["Optimal Designs"])
app.include_router(nonlinear_regression.router, prefix="/api/nonlinear-regression", tags=["Nonlinear Regression"])
app.include_router(quality_control.router, prefix="/api/quality-control", tags=["Quality Control"])
app.include_router(reliability.router, prefix="/api/reliability", tags=["Reliability Analysis"])
app.include_router(glm.router, tags=["Generalized Linear Models"])
app.include_router(custom_design.router, tags=["Custom Design"])
app.include_router(msa.router, tags=["Measurement Systems Analysis"])
app.include_router(predictive_modeling.router, tags=["Predictive Modeling"])
app.include_router(mixture.router, tags=["Mixture Designs"])

@app.get("/health")
@app.head("/health")
async def health_check():
    return {"status": "healthy"}

# Serve static frontend files (for Electron and standalone deployment)
frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    # Mount static assets (CSS, JS, images)
    assets_path = frontend_dist / "assets"
    if assets_path.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

    # Serve index.html for root and all non-API routes (SPA routing)
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Serve index.html for root and all routes except API/docs/health
        index_path = frontend_dist / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"error": "Frontend not found"}
