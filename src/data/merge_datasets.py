import os 
import pandas as pd 
import sys 

def MergeDatasets(ticker_symb):
    path = f"./dataset/{ticker_symb}/frac_data/"
    file_list = [path+f for f in os.listdir(path) if (f.endswith('.csv') and (f.startswith(ticker_symb)))]
    
    df_list = []
    for file in file_list:
        df = pd.read_csv(file, header=0)
        df['timestamp'] = pd.to_datetime(df['tradedate'] + ' ' + df['tradetime'])
     #   print(*df.columns)
        df = df.drop(columns=["tradetime", "tradedate"])
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='first')

    combined_df.sort_values('timestamp', ignore_index=True)
    print(combined_df)
    combined_df.to_csv(f"./dataset/{ticker_symb}/{ticker_symb}_SUPER_FULL.csv", index=False)

#MergeDatasets("YDEX")

if __name__ == "__main__":
    if len(sys.argv) < 2: 
        print("missing ticker argument"); 
    MergeDatasets(sys.argv[1])