#!/bin/bash
D=$1; R=/private/tmp/claude-501/-Users-arrams-Documents-personal-work-quant-quant-regime-research-notebooks/a3adba8f-b536-4614-be27-5ad5ba6bdbd4/scratchpad/replay
W=$R/w_$D; rm -rf $W; mkdir -p $W; cp $R/base/* $W/ 2>/dev/null
cd $W; PY=/Users/arrams/Documents/personal_work/quant/quant-regime-research/.venv/bin/python
export DASH_ASOF=$D
$PY -W ignore build_gold_dashboard.py > gold.log 2>&1 || { echo "$D gold FAIL"; exit 1; }
if [[ "$D" > "2026-07-12" ]]; then $PY -W ignore build_metals_dashboard.py > metals.log 2>&1 || { echo "$D metals FAIL"; exit 1; }; export WITH_METALS=1; fi
$PY -W ignore extract_state.py > extract.log 2>&1 || { echo "$D extract FAIL"; exit 1; }
echo "$D ok"
