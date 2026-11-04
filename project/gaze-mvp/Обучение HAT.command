#!/bin/bash
set -o pipefail
cd "$(dirname "$0")" || exit 1
mkdir -p models/hat-freeview-k6-seed42
export TORCH_HOME="$PWD/models/torch"
export PYTHONUNBUFFERED=1
caffeinate -i .venv-detector/bin/python -u -m training.run_training 2>&1 | tee -a models/hat-freeview-k6-seed42/console.log
result=${PIPESTATUS[0]}
echo ""
echo "Процесс остановлен (код $result). Журнал: models/hat-freeview-k6-seed42/console.log"
echo "Для продолжения снова откройте Обучение HAT.command."
read -r -p "Нажмите Enter, чтобы закрыть окно…"
exit "$result"
