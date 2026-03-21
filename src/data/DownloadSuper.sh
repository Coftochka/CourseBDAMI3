#!/bin/bash

SKIP_DOWNLOAD=false

if [ "$1" = "--skip-download" ]; then
    SKIP_DOWNLOAD=true
fi

echo "write down the ticker:   [--skip-download  flag is possible]"
read ticker

cur_dir=$(dirname "$(readlink -f "$0")")
cd "$cur_dir"

mkdir -p "$cur_dir/dataset/${ticker}/" "$cur_dir/dataset/${ticker}/frac_data"

# Запуск download только если флаг не установлен
if [ "$SKIP_DOWNLOAD" = false ]; then
    python3 "${cur_dir}/download_super_candels.py" "$ticker"
fi

python3 "${cur_dir}/merge_datasets.py" "$ticker"
head -n 50 "$cur_dir/dataset/${ticker}/${ticker}_SUPER_FULL.csv" > "$cur_dir/dataset/${ticker}/${ticker}_SUPER_HEAD50.csv"
