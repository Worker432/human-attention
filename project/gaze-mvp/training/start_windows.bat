@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set TORCH_HOME=%CD%\models\torch
if not exist ".venv\Scripts\python.exe" (
  echo Create the Python environment first. See training\WINDOWS.md
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -c "import torch; assert torch.cuda.is_available(), 'NVIDIA CUDA is not available'; print(torch.cuda.get_device_name(0))"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -u -m training.train_hat --data dataset\coco-freeview --output models\hat-windows-k6 --device cuda --batch 4 --epochs 5 --check-only
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -u -m training.train_hat --data dataset\coco-freeview --output models\hat-windows-k6 --device cuda --batch 4 --epochs 5
if errorlevel 1 goto failed
echo Training finished or stopped with saving.
pause
exit /b 0
:failed
echo Training did not finish normally. See the error above.
pause
exit /b 1
