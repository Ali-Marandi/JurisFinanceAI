# -*- mode: python ; coding: utf-8 -*-

"""
PyInstaller spec file for JurisFinanceAI
Builds a professional Windows .exe application.
"""

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'openai',
        'httpx',
        'httpcore2',
        'pdfplumber',
        'docx',
        'matplotlib',
        'numpy',
        'scipy',
        'scipy.optimize',
        'scipy.stats',
        'scipy.linalg',
        'scipy.sparse',
        'cryptography',
        'PIL',
        'src.quant',
        'src.quant.portfolio',
        'src.quant.derivatives',
        'src.quant.risk_models',
        'src.quant.time_series',
        'src.quant.fuzzy_systems',
        'src.quant.network',
        'src.quant.behavioral',
        'src.quant.monte_carlo',
        'src.quant.interest_rates',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'pandas',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='JurisFinanceAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='JurisFinanceAI',
)
