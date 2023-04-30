import datetime
import pandas as pd
from tqdm import tqdm

def find_mileages_differences(df_):
    """
    This function finds the differences between the mileages of the edges. It is a temporary function, used to find
    the edges with different mileages.
    :param df_: The dataframe of the railway dataset.
    """
    edges_subs = {}
    dp_st_id = 0
    arr_st_id = 0
    edge_mileages = []
    double_edge_detector = False
    print("Starting edges analysis.")
    for index, row in tqdm(df_.iterrows(), total=len(df_)):
        if row['dep_st_id'] != dp_st_id or row['arr_st_id'] != arr_st_id:
            if double_edge_detector:
                print("ERROR: EDGE WITH DIFFERENT MILEAGES DETECTED: { DEP_ST_ID=" + str(dp_st_id) + " , ARR_ST_ID=" + str(arr_st_id) + " , EDGE_MILEAGES(" + str(len(edge_mileages)) + ")=" + str(edge_mileages) + " }")
                double_edge_detector = False
                # Add the edge to the dictionary
                element = {
                    'dep_st_id': dp_st_id,
                    'arr_st_id': arr_st_id,
                    'mileages': edge_mileages[0]
                }
                edges_subs[index] = element
            dp_st_id = row['dep_st_id']
            arr_st_id = row['arr_st_id']
            edge_mileages = []
            edge_mileages.append(row['mileage'])
        if row['arr_st_id'] == arr_st_id and row['dep_st_id'] == dp_st_id:
            # if row['mileage'] != mileage and row['mileage'] != 0:
            if (not row['mileage'] in edge_mileages) and row['mileage'] != 0:
                double_edge_detector = True
                edge_mileages.append(row['mileage'])


def edges_preprocessing(df, save_csv=False, output_path=None):
    """
    This function preprocesses the edges dataframe, converting the trains stops to edges travels.
    :param df: The dataframe of the railway dataset.
    :param save_csv: The boolean flag that indicates whether to save the dataframe to a csv file or not.
    :param output_path: The output path of the csv file.
    :return: The dataframe of the edges travels.
    """
    print("Preprocessing the edges dataframe...")
    # STEP 1: EDGES CONSTRUCTION: CONVERT TRAINS STOPS TO EDGES TRAVELS:
    edgesPassages = {}
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if index != len(df) - 1 and row['train'] == df.iloc[index + 1]['train']:
            travel_time = (datetime.datetime.strptime(df.iloc[index + 1]['arr_time'], '%H:%M:%S') - datetime.datetime.strptime(row['dep_time'], '%H:%M:%S')).seconds // 60
            # Convert total mileage to mileage between two stations:
            if df.iloc[index + 1]['mileage'] != ' ' and row['st_no'] == 1:
                mileage = int(df.iloc[index + 1]['mileage'])
            elif df.iloc[index + 1]['mileage'] != ' ' and row['mileage'] != ' ':
                mileage = int(df.iloc[index + 1]['mileage']) - int(row['mileage'])
            else:
                mileage = 0
            # Create the edge travel as an element of a dictionary:
            element = {
                'train_id': row['train'],
                'dep_st_id': row['st_id'],
                'day': int(row['date']),
                'dep_date': row['dep_time'],
                'travel_time': travel_time,
                'arr_st_id': df.iloc[index + 1]['st_id'],
                'mileage': mileage if mileage >= 0 else 0,
            }
            # Add the edge to the dictionary
            edgesPassages[index] = element
    # Convert the dictionary to a dataframe:
    df_ = pd.DataFrame(edgesPassages).T\
        .sort_values(by=['dep_st_id', 'arr_st_id'], ascending=True)\
        .reset_index(drop=True)

    # STEP 2: MILEAGE FILTERING/CONSENSUS:
    # The idea is to find the mileage of the edges whose mileage is 0 and set it to the value of the another instance of
    # a combination of dp_st_id and arr_st_id whose travel time is the closest to their average travel time:
    # First, we need a dataframe with the average travel time for each combination of dep_st_id, arr_st_id and mileage:
    # Group by unique combinations of dep_st_id, arr_st_id and mileage (and keep the average travel time):
    df_values = df_.groupby(['dep_st_id', 'arr_st_id', 'mileage']).apply(lambda x: pd.Series({
        'avg_travel_time': x['travel_time'].mean()
    })).reset_index()
    # Drop the rows whose mileage is 0:
    df_values = df_values[df_values['mileage'] != 0]
    # Sort the dataframe by dep_st_id, arr_st_id and mileage:
    df_values = df_values.sort_values(by=['dep_st_id', 'arr_st_id', 'mileage'], ascending=True)
    # In the df_ dataframe, if the mileage is 0, set the mileage to the value of the another instance of a combination
    # of dp_st_id and arr_st_id whose travel time is the closest to their average travel time:
    for i in tqdm(range(len(df_)), total=len(df_)):
        if df_.iloc[i]['mileage'] == 0:
            dp_st_id = df_.iloc[i]['dep_st_id']
            ar_st_id = df_.iloc[i]['arr_st_id']
            avg_travel_time = df_.iloc[i]['travel_time']
            # Find the closest travel time:
            travel_times = df_values[(df_values['dep_st_id'] == dp_st_id) & (df_values['arr_st_id'] == ar_st_id)][
                'avg_travel_time']
            if len(travel_times) == 0:  # If there is no other instance of a combination of dp_st_id and arr_st_id
                mileage = 0             # Set the mileage to 0
            else:
                if len(travel_times) == 1:  # If there is only one other instance of a combination of dp_st_id and arr_st_id
                    closest_travel_time = travel_times.iloc[0]  # Set the mileage to the value of the only instance
                    mileage = df_values[
                        (df_values['dep_st_id'] == dp_st_id) & (df_values['arr_st_id'] == ar_st_id) & (
                                    df_values['avg_travel_time'] == closest_travel_time)]['mileage'].iloc[0]
                    df_.loc[i, 'travel_time'] = round(closest_travel_time)
                    df_.loc[i, 'mileage'] = mileage
                else:   # If there are more than one other instance of a combination of dp_st_id and arr_st_id
                    closest_travel_time = travel_times[
                        abs(travel_times - avg_travel_time) == abs(travel_times - avg_travel_time).min()]
                    mileage = df_values[
                        (df_values['dep_st_id'] == dp_st_id) & (df_values['arr_st_id'] == ar_st_id) & (
                                    df_values['avg_travel_time'] == closest_travel_time.iloc[0])]['mileage'].iloc[0]
                    df_.loc[i, 'travel_time'] = round(closest_travel_time.iloc[0])
                    df_.loc[i, 'mileage'] = mileage
    df_ = df_.sort_values(by=['dep_st_id', 'arr_st_id', 'mileage'], ascending=True)\
        .reset_index(drop=True)
    if save_csv and output_path is not None:
        df_.to_csv(output_path, index=False)
        print("Edges dataframe saved in: " + output_path)
    print("Edges dataframe preprocessed.")
    return df_