import pandas as pd

def nodes_preprocessing(df, save_csv=False, output_path=None):
    print("Preprocessing the nodes dataframe...")
    df_ = df.copy().sort_values(by=['st_id', 'arr_time'], ascending=True)\
        .reset_index(drop=True)\
        .drop(columns=['mileage', 'st_no', 'dep_time'])
    df_.rename(columns={'date': 'day'}, inplace=True)
    # Remove the 4 first character of the elements in day column:
    df_['day'] = df_['day'].str[4:].astype(int)
    if save_csv and (output_path is not None):
        df_.to_csv(output_path, index=False)
        print("Nodes dataframe saved in: " + output_path)
    print("Nodes dataframe preprocessed.")
    return df_