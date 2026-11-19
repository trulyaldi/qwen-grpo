#!/usr/bin/env bash
# Prints reports/Soup_Final_Research_Report.html to A4 PDF with the Windows Chrome in headless mode (WSL host).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
chrome="${CHROME:-/mnt/c/Program Files/Google/Chrome/Application/chrome.exe}"
tmp_win="$(cmd.exe /c echo %TEMP% 2>/dev/null | tr -d '\r')"
tmp_wsl="$(wslpath -u "$tmp_win")/soup_report_build"
rm -rf "$tmp_wsl" && mkdir -p "$tmp_wsl/figures"
cp "$here/Soup_Final_Research_Report.html" "$tmp_wsl/"
cp "$here/figures/fig_memory_scores.svg" "$tmp_wsl/figures/"
src="$(wslpath -w "$tmp_wsl/Soup_Final_Research_Report.html")"
out="$(wslpath -w "$tmp_wsl/out.pdf")"
"$chrome" --headless=new --disable-gpu --no-pdf-header-footer --run-all-compositor-stages-before-draw \
  --virtual-time-budget=5000 --print-to-pdf="$out" "file:///${src//\\//}" >/dev/null 2>&1
cp "$tmp_wsl/out.pdf" "$here/Soup_Final_Research_Report.pdf"
pdfinfo "$here/Soup_Final_Research_Report.pdf" | grep -E "^Pages|^Page size"
