# Обучение с нуля в Windows, NVIDIA CUDA

На Маке скопируйте обновлённую папку `training` и `dataset/coco-freeview`.
На ноутбуке расположите их как `C:\human-attention\training` и
`C:\human-attention\dataset\coco-freeview`. В данных нужны `images` с фотографиями
и `raw/COCOFreeView_fixations_trainval.json`. Данные читаются с SSD, не с флешки.
Старые веса для нового запуска не нужны.

Установите 64-битный Python 3.12 с python.org и актуальный драйвер NVIDIA.
В PowerShell, из папки `C:\human-attention`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install numpy pillow
.\training\start_windows.bat
```

Активация окружения и изменение ExecutionPolicy не требуются. Библиотеки CUDA
поставляются с PyTorch; отдельный CUDA Toolkit для этого переноса не нужен.
Установка: https://pytorch.org/get-started/locally/

BAT проверяет доступность GPU и реальный шаг с валидацией, затем начинает обучение.
При первом запуске нужны интернет для весов ImageNet и несколько ГБ свободного места.
Результаты: `models/hat-windows-k6/latest.pt`, `metrics.jsonl`, `status.json`.
При повторном запуске продолжается сохранённое обучение. Ctrl+C нажмите один раз
и дождитесь сообщения о сохранении; не закрывайте окно до этого. Windows может
после Ctrl+C спросить о завершении BAT — после сохранения можно ответить Y.
Не допускайте сна ноутбука во время обучения.

На этой машине Windows/CUDA не проверялись; обязательная проверка перед полным
обучением выполняется непосредственно на целевом ноутбуке. При CUDA out of memory
для свежего эксперимента можно снизить batch до 2 в обеих командах BAT, сохранив
результаты в новой папке. Для существующих весов batch должен совпадать.
