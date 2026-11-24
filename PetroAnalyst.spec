# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = []
hiddenimports += collect_submodules("app")
hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("petroleum_kb")


a = Analysis(
    ["desktop_entry.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("app/static", "app/static"),
        (
            "app/capabilities/providers/hydrocarbon_type_calculator",
            "app/capabilities/providers/hydrocarbon_type_calculator",
        ),
        ("data/experimental_db", "data/experimental_db"),
        ("petroleum_kb/kb_core/data", "petroleum_kb/kb_core/data"),
        ("petroleum_kb/eval/datasets", "petroleum_kb/eval/datasets"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "transformers",
        "sentence_transformers",
        "tensorflow",
        "sklearn",
        "scipy",
        "matplotlib",
        "plotly",
        "pytesseract",
        "fitz",
        "spacy",
        "thinc",
        "IPython",
        "jedi",
        "pygments",
        "pyarrow",
        "sqlalchemy",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="PetroAnalyst",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
