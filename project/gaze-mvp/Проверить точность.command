#!/bin/zsh
cd -- "${0:A:h}"
exec .venv/bin/python -u validate_gaze.py
