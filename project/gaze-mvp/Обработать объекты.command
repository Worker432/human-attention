#!/bin/zsh
cd -- "${0:A:h}"
exec .venv-detector/bin/python -u analyze_gaze.py
