#!/bin/bash

# 使用 # 代替 @REM 作為註釋
# echo Running: python3 -m examples.dho.main --mode train --is_pretrained=True
# python3 -m examples.dho.main --mode train --is_pretrained=True
# if [ $? -ne 0 ]; then
#     echo "Command failed: python3 -m examples.dho.main --mode train --is_pretrained=True"
#     exit 1
# fi

echo "Running: python3 -m examples.dho.main --mode train --is_pretrained=False"
python3 -m examples.dho.main --mode train --is_pretrained=False

# $? 會抓取上一個指令的結束狀態碼 (Exit Code)
if [ $? -ne 0 ]; then
    echo "Command failed: python3 -m examples.dho.main --mode train --is_pretrained=False"
    exit 1
fi

echo "Running: python3 -m examples.dho.main --mode eval"
python3 -m examples.dho.main --mode eval
if [ $? -ne 0 ]; then
    echo "Command failed: python3 -m examples.dho.main --mode eval"
    exit 1
fi

echo "All commands executed successfully."