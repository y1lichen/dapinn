@echo off
setlocal enabledelayedexpansion

REM Define the list of noise levels
set noise_levels=0.01 0.02 0.03 0.05 0.10 0.15

REM Loop through each noise level
for %%n in (%noise_levels%) do (
    echo ============================
    echo Running for noise=%%n
    echo ============================

    echo Running: python -m examples.dho.main --mode train --is_pretrained=False --noise=%%n
    python -m examples.dho.main --mode train --is_pretrained=False --noise=%%n
    if !errorlevel! neq 0 (
        echo Command failed: train for noise=%%n
        exit /b !errorlevel!
    )

    echo Running: python -m examples.dho.main --mode eval --is_pretrained=False --noise=%%n
    python -m examples.dho.main --mode eval --is_pretrained=False --noise=%%n
    if !errorlevel! neq 0 (
        echo Command failed: eval for noise=%%n
        exit /b !errorlevel!
    )

    REM Create output directory
    set outdir=noise\dho\%%n
    if not exist "!outdir!" (
        mkdir "!outdir!"
    )

    REM Copy default.py
    copy /Y examples\dho\configs\default.py "!outdir!\default.py"

    REM Copy results/dho folder recursively
    xcopy /E /I /Y results\dho "!outdir!\dho"

    echo Finished processing noise=%%n
)

echo All noise levels processed successfully.
endlocal
