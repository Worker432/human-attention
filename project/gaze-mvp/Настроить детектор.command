#!/bin/zsh
set -e
cd -- "${0:A:h}"
.venv/bin/python -m venv .venv-detector
.venv-detector/bin/python -m pip install -r requirements-detector.txt
echo "Детектор готов. В панели записей нажмите «Обработать новые · YOLO»."
