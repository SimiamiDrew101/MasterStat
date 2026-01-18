#!/usr/bin/env python3
"""
MasterStat Desktop Launcher
Serves the built React frontend and FastAPI backend as a desktop application
Runs entirely locally with no internet access required
"""
import sys
import os
import webbrowser
import threading
import time
from pathlib import Path

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def start_backend():
    """Start FastAPI backend server"""
    import uvicorn
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    from pathlib import Path

    # Add backend directory to Python path
    backend_path = get_resource_path('backend')
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    # Import all API routers
    from app.api import (
        hypothesis_testing, anova, factorial, block_designs,
        mixed_models, rsm, power_analysis, bayesian_doe,
        import_data, stats, preprocessing, imputation, protocol
    )

    # Create FastAPI app
    app = FastAPI(
        title="MasterStat",
        description="Professional-grade statistical analysis and Design of Experiments platform",
        version="1.0.0"
    )

    # Include all API routers
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
        return {"status": "healthy", "mode": "standalone"}

    # Get path to frontend dist folder
    static_path = Path(get_resource_path('frontend')) / 'dist'

    if static_path.exists():
        # Mount assets directory
        assets_path = static_path / "assets"
        if assets_path.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

        @app.get("/{full_path:path}")
        async def serve_react(full_path: str):
            """Serve React app for all non-API routes"""
            file_path = static_path / full_path
            if file_path.is_file():
                return FileResponse(file_path)
            # Fallback to index.html
            index_path = static_path / "index.html"
            if index_path.exists():
                return FileResponse(index_path)
            return {"error": "Frontend not found"}

    # Run on localhost only (security) - no external access
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

def open_browser():
    """Open browser after backend starts"""
    print("Waiting for backend to start...")

    # Wait for server to be ready
    import urllib.request
    import urllib.error

    for i in range(30):  # Try for 30 seconds
        try:
            urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1)
            print("Backend ready!")
            break
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(1)

    # Open browser
    print("Opening MasterStat in your default browser...")
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    print("=" * 60)
    print("  MasterStat - Statistical Analysis & Design of Experiments")
    print("  Version 1.0.0")
    print("  Running locally (no internet required)")
    print("=" * 60)
    print()
    print("Backend API: http://127.0.0.1:8000")
    print("API Documentation: http://127.0.0.1:8000/docs")
    print()

    # Start backend in background thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()

    # Open browser in separate thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    # Keep main thread alive
    try:
        print("Press Ctrl+C to quit")
        backend_thread.join()
    except KeyboardInterrupt:
        print("\n\nShutting down MasterStat...")
        print("Thank you for using MasterStat!")
        sys.exit(0)
