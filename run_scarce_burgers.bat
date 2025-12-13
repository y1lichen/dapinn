@echo off

setlocal enabledelayedexpansion

REM Define the list of sample sizes
set sample_sizes=15 50 100 500 1000 5000 10000 30000

REM Loop through each sample size
for %%s in (%sample_sizes%) do (
    echo ============================
    echo Running for sample_size=%%s
    echo ============================

    echo Running: python -m examples.burgers.main --mode train --is_pretrained=False --finetune_sample_size=%%s
    python -m examples.burgers.main --mode train --is_pretrained=False --finetune_sample_size=%%s
    if !errorlevel! neq 0 (
        echo Command failed: train for sample_size=%%s
        exit /b !errorlevel!
    )

    echo Running: python -m examples.burgers.main --mode eval --is_pretrained=False --finetune_sample_size=%%s
    python -m examples.burgers.main --mode eval --is_pretrained=False --finetune_sample_size=%%s
    if !errorlevel! neq 0 (
        echo Command failed: eval for sample_size=%%s
        exit /b !errorlevel!
    )

    REM Create output directory
    set outdir=scarce\burgers\%%s
    if not exist !outdir! (
        mkdir !outdir!
    )

    REM Copy default.py
    copy /Y examples\burgers\configs\default.py !outdir!\default.py

    REM Copy results/burgers folder recursively
    xcopy /E /I /Y results\burgers !outdir!\burgers

    echo Finished processing sample_size=%%s
)

echo All sample sizes processed successfully.
endlocal
