@echo off
echo STORM SYSTEMS DEPENDENCY INSTALLER (PYTHON MODULE MODE)
echo =======================================================
echo Installing critical research modules...
python -m pip install arxiv requests networkx scipy beautifulsoup4 numpy pandas tabula-py sentence-transformers faiss-cpu semanticscholar
echo.
echo All dependencies installed.
pause
