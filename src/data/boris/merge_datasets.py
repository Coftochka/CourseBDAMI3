import os 
import pandas as pd 



def MergeDatasets(ticker_symb):
    path = f"./dataset/{ticker_symb}/"
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
    combined_df.to_csv(f"{ticker_symb}_FULL.csv", index=False)



MergeDatasets("SBER")