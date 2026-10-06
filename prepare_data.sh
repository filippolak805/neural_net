#!/usr/bin/env bash
# Extract data/*.csv.gz back into plain CSVs (keeps the .gz archives).
# Run once after cloning, or let run.sh call it automatically.
shopt -s nullglob
archives=(data/*.csv.gz)
(( ${#archives[@]} )) || exit 0
gzip -dkf "${archives[@]}"
