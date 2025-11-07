from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import hypothesis_testing, anova, factorial, block_designs, mixed_models, rsm

app = FastAPI(
    title="MasterStat - Statistical Analysis Tool",
    description="Comprehensive statistical analysis covering experimental design and ANOVA",
    version="1.0.0"
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(hypothesis_testing.router, prefix="/api/hypothesis", tags=["Hypothesis Testing"])
app.include_router(anova.router, prefix="/api/anova", tags=["ANOVA"])
app.include_router(factorial.router, prefix="/api/factorial", tags=["Factorial Designs"])
app.include_router(block_designs.router, prefix="/api/blocks", tags=["Block Designs"])
app.include_router(mixed_models.router, prefix="/api/mixed", tags=["Mixed Models"])
app.include_router(rsm.router, prefix="/api/rsm", tags=["Response Surface Methodology"])

@app.get("/")
async def root():
    return {
        "message": "MasterStat API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
