# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for MasterStat
Creates a standalone macOS application bundle
"""

block_cipher = None

# Collect all backend Python files
backend_datas = [
    ('backend/app', 'backend/app'),
]

# Include built frontend (must run 'npm run build' first)
frontend_datas = [
    ('frontend/dist', 'frontend/dist'),
]

# Combine data files
datas = backend_datas + frontend_datas

# Hidden imports - packages PyInstaller might miss during auto-detection
hiddenimports = [
    # Uvicorn internals
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',

    # SciPy compiled extensions
    'scipy.special._ufuncs_cxx',
    'scipy.linalg.cython_blas',
    'scipy.linalg.cython_lapack',
    'scipy.integrate',
    'scipy.integrate._odepack',
    'scipy.integrate._quadpack',
    'scipy.integrate._dop',
    'scipy.integrate._lsoda',
    'scipy.sparse.csgraph._validation',

    # Scikit-learn internals
    'sklearn.utils._cython_blas',
    'sklearn.neighbors._typedefs',
    'sklearn.neighbors._quad_tree',
    'sklearn.tree._utils',

    # Statsmodels internals
    'statsmodels.stats',
    'statsmodels.stats.power',
    'statsmodels.stats.api',
    'statsmodels.stats.multitest',
    'statsmodels.stats.anova',
    'statsmodels.stats.diagnostic',
    'statsmodels.stats.libqsturng',
    'statsmodels.stats.contrast',
    'statsmodels.stats.outliers_influence',
    'statsmodels.stats.stattools',
    'statsmodels.tsa.statespace._filters',
    'statsmodels.tsa.statespace._smoothers',
    'statsmodels.tsa.statespace._initialization',
    'statsmodels.tsa.statespace._representation',
    'statsmodels.tsa.statespace._kalman_filter',
    'statsmodels.tsa.statespace._kalman_smoother',
    'statsmodels.formula',
    'statsmodels.formula.api',
    'statsmodels.regression',
    'statsmodels.regression.linear_model',
    'statsmodels.api',

    # ReportLab for PDF generation
    'reportlab.pdfbase._fontdata',
    'reportlab.pdfbase._cidfontdata',

    # Pandas/NumPy
    'pandas._libs.tslibs.timedeltas',
    'pandas._libs.tslibs.nattype',
    'pandas._libs.tslibs.np_datetime',
    'pandas._libs.skiplist',
]

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'matplotlib',  # Not used
        'IPython',
        'jupyter',
        'notebook',
        'tkinter',
        'PyQt5',
        'PySide2',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MasterStat',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MasterStat',
)

# macOS app bundle
app = BUNDLE(
    coll,
    name='MasterStat.app',
    bundle_identifier='com.masterstat.app',
    info_plist={
        'CFBundleName': 'MasterStat',
        'CFBundleDisplayName': 'MasterStat',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': 'True',
        'LSMinimumSystemVersion': '10.13.0',
        'CFBundleDocumentTypes': [
            {
                'CFBundleTypeName': 'CSV Document',
                'CFBundleTypeRole': 'Viewer',
                'LSHandlerRank': 'Alternate',
                'LSItemContentTypes': ['public.comma-separated-values-text'],
            }
        ],
        'NSHumanReadableCopyright': 'Copyright © 2024 MasterStat Contributors. Licensed under CC BY 4.0.',
    },
)
