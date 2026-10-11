#!/bin/sh
# Rebuild Paper 17: markdown -> HTML -> PDF (run from notebooks/)
python3 paper17_build.py && pandoc weather_impacts_paper_draft.md -o weather_impacts_paper_draft.html --standalone --css=predictability_paper/paper_style.css --metadata pagetitle="Why Everyday Weather Does Not Move Markets, and Where Rare Extremes Might" --embed-resources && weasyprint weather_impacts_paper_draft.html weather_impacts_paper_draft.pdf --base-url .
