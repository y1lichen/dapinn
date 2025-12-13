@echo off

echo Running: python -m examples.burgers.main --mode train --is_pretrained=True
python -m examples.burgers.main --mode train --is_pretrained=True
if %errorlevel% neq 0 (
    echo Command failed: python -m examples.burgers.main --mode train --is_pretrained=True
    exit /b %errorlevel%
)

@REM echo Running: python -m examples.burgers.main --mode train --is_pretrained=False
@REM python -m examples.burgers.main --mode train --is_pretrained=False
@REM if %errorlevel% neq 0 (
@REM     echo Command failed: python -m examples.burgers.main --mode train --is_pretrained=False
@REM     exit /b %errorlevel%
@REM )

@REM echo Running: python -m examples.burgers.main --mode eval
@REM python -m examples.burgers.main --mode eval
@REM if %errorlevel% neq 0 (
@REM     echo Command failed: python -m examples.burgers.main --mode eval
@REM     exit /b %errorlevel%
@REM )

echo All commands executed successfully.