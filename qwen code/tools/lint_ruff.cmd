@echo off
REM Aider lint wrapper — always use workspace venv ruff
set "SCRIPTS=%~dp0..\..\.venv\Scripts"
if defined QWEN_VENV_SCRIPTS set "SCRIPTS=%QWEN_VENV_SCRIPTS%"
"%SCRIPTS%\ruff.exe" check --fix --unsafe-fixes %*
