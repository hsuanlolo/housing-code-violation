#!/usr/bin/env bash
# Build distributable formats of the working paper from paper/manuscript.md.
# Uses --metadata pagetitle (not title) so pandoc does not print a second title
# block above the manuscript's own "# " heading.
set -euo pipefail
cd "$(dirname "$0")/.."

MD=paper/manuscript.md
OUT=${1:-paper/Lo_2026_Before_Signing_the_Lease}
TITLE="Before Signing the Lease: Predictive Value and Spatial Reliability of Housing Maintenance Records in New York City"
ARGS=(--from=markdown+pipe_tables+tex_math_dollars --resource-path="paper:.:out/figures"
      --metadata pagetitle="$TITLE")

pandoc "$MD" "${ARGS[@]}" --to=html5 --standalone --mathjax -o paper/manuscript.html
pandoc "$MD" "${ARGS[@]}" --to=docx -o "$OUT.docx"
echo "built paper/manuscript.html and $OUT.docx"
