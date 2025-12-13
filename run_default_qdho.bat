@echo off

@REM echo Running: python -m examples.qdho.main --mode train --is_pretrained=True
@REM python -m examples.qdho.main --mode train --is_pretrained=True
@REM if %errorlevel% neq 0 (
@REM     echo Command failed: python -m examples.qdho.main --mode train --is_pretrained=True
@REM     exit /b %errorlevel%
@REM )

echo Running: python -m examples.qdho.main --mode train --is_pretrained=False
python -m examples.qdho.main --mode train --is_pretrained=False
if %errorlevel% neq 0 (
    echo Command failed: python -m examples.qdho.main --mode train --is_pretrained=False
    exit /b %errorlevel%
)

echo Running: python -m examples.qdho.main --mode eval
python -m examples.qdho.main --mode eval
if %errorlevel% neq 0 (
    echo Command failed: python -m examples.qdho.main --mode eval
    exit /b %errorlevel%
)

echo All commands executed successfully.