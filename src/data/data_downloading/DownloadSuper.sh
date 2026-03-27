
#echo "write down the ticker:   [--skip-download  flag is possible]"
#read ticker

#cur_dir=$(dirname "$(readlink -f "$0")")
#cd "$cur_dir"

#mkdir -p "$cur_dir/dataset/${ticker}/" "$cur_dir/dataset/${ticker}/frac_data"

#python3 "${cur_dir}/download_super_candels.py" "$ticker"
#python3 "${cur_dir}/merge_datasets.py" "$ticker"

#head -n 50 "$cur_dir/dataset/${ticker}/${ticker}_SUPER_FULL.csv" > "$cur_dir/dataset/${ticker}/${ticker}_SUPER_HEAD50.csv"

#==========================================



#!/bin/bash

cur_dir=$(dirname "$(readlink -f "$0")")
cd "$cur_dir"


tickers_file="${1:-tickers.txt}"

if [ ! -f "$tickers_file" ]; then
    echo "Файл с тикерами $tickers_file не найден!"
    exit 1
fi


echo "Читаем тикеры из $tickers_file"


while IFS= read -r ticker  || [ -n "$ticker" ] 
do    
    echo "Processing ticker: $ticker"
    
    mkdir -p "$cur_dir/dataset/${ticker}/" "$cur_dir/dataset/${ticker}/frac_data"
    
    python3 "${cur_dir}/download_super_candels.py" "$ticker"
    python3 "${cur_dir}/merge_datasets.py" "$ticker"
    
    head -n 50 "$cur_dir/dataset/${ticker}/${ticker}_SUPER_FULL.csv" > "$cur_dir/dataset/${ticker}/${ticker}_SUPER_HEAD50.csv"

done < $tickers_file
