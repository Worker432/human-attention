#!/bin/zsh
cd -- "${0:A:h}"
exec .venv/bin/python collection_store.py collected/p001/dataset.sqlite3 "collected/export-$(date +%Y%m%d-%H%M%S)"
