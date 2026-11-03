#!/bin/zsh
set -e
cd -- "${0:A:h}"
if [[ ! -x .venv-detector/bin/python ]]; then
    .venv/bin/python -m venv .venv-detector
fi
.venv-detector/bin/python -m pip install setuptools==80.9.0 wheel
.venv-detector/bin/python -m pip install --no-build-isolation -r requirements-prediction.txt
echo "Готово. Откройте «Прогноз внимания.command»."
