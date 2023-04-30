import datetime
import datetime
from tqdm import tqdm
import math

# 1. REMOVE DUPLICATED ROWS:
def remove_duplicates(df):
    """
    This pipeline step removes duplicated rows from the dataframe.
    :param df: The dataframe to have its duplicated rows removed.
    :return: The dataframe without duplicated rows.
    """
    df = df.drop_duplicates()
    print("Removed " + str(len(df) - len(df.drop_duplicates())) + " duplicated row(s).")
    df = df.reset_index(drop=True)
    return df

# 2. REMOVE EDGES THAT CONNECT THE SAME STATION:
def remove_edges_that_connect_the_same_station(df):
    """
    This pipeline step removes edges that connect the same station.
    :param df: The dataframe to have its edges that connect the same station removed.
    :return: The dataframe without edges that connect the same station.
    """
    indexes = []
    for index, row in df.iterrows():
        if index != 0 and row['train'] == df.iloc[index - 1]['train'] and row['st_id'] == df.iloc[index - 1]['st_id']:
            indexes.append(index)
    print("Found and removed " + str(len(indexes)) + " edges that connect the same station: { ", end='')
    for index in indexes:
        print("{Train=" + str(df.iloc[index]['train']) + ", Stop n°=" + str(df.iloc[index]['st_no']) + "}", end=' ')
    print("}")
    df = df.drop(indexes)
    df = df.reset_index(drop=True)
    return df

# 3. UNNORMALIZE THE DEPARTURE AND ARRIVAL TIMES:
def unnormalize_times(df):
    """
    This pipeline step unnormalizes the departure and arrival times.
    :param df: The dataframe to have its departure and arrival times unnormalized.
    :return: The dataframe with unnormalized departure and arrival times.
    """
    count = 0
    for index, row in df.iterrows():
        try:
            datetime.datetime.strptime(row['dep_time'], '%H:%M:%S')
        except ValueError:
            temp = row['dep_time']
            temp = datetime.datetime.fromtimestamp(float(temp) * 86400).strftime('%H:%M:%S')
            row['dep_time'] = temp
            df.at[index, 'dep_time'] = temp
            count += 1
        try:
            datetime.datetime.strptime(row['arr_time'], '%H:%M:%S')
        except ValueError:
            temp = row['arr_time']
            temp = datetime.datetime.fromtimestamp(float(temp) * 86400).strftime('%H:%M:%S')
            row['arr_time'] = temp
            df.at[index, 'arr_time'] = temp
            count += 1
    print("Unnormalized " + str(count) + " times.")
    df = df.reset_index(drop=True)
    return df

# 4. REMOVE TRAINS WITH CORRUPTED TIMES:
def remove_trains_with_corrupted_times(df):
    """
    This pipeline step removes trains with corrupted times.
    :param df: The dataframe to have its trains with corrupted times removed.
    :return: The dataframe without trains with corrupted times.
    """
    train_id = 0
    isTrainCorrupted = False
    corrupted_train_ids = []
    for index, row in df.iterrows():
        if train_id != row['train']:
            train_id = row['train']
            isTrainCorrupted = False
        if isTrainCorrupted == False:
            if datetime.datetime.strptime(row['dep_time'], '%H:%M:%S') < datetime.datetime.strptime(row['arr_time'], '%H:%M:%S'):
                if int(row['date'][4:]) == int(df.at[index - 1, 'date'][4:]):
                    isTrainCorrupted = True
                    corrupted_train_ids.append([train_id, row['st_no']])
    for train_id in corrupted_train_ids:
        df = df[df.train != train_id[0]]
    print('Found and removed ' + str(len(corrupted_train_ids)) + ' train(s) with corrupted times: {', end='')
    for element in corrupted_train_ids:
        print('{',element[0], ',', element[1], '}, ', end='')
    print('}')
    df = df.reset_index(drop=True)
    return df

# 5. REPLACE NULL MILEAGES WITH ZEROES:
def replace_null_mileages_with_zeroes(df):
    """
    This pipeline step replaces null mileages with zeroes.
    :param df: The dataframe to have its null mileages replaced with zeroes.
    :return: The dataframe with null mileages replaced with zeroes.
    """
    df['mileage'] = df['mileage'].fillna(0)
    df['mileage'] = df['mileage'].replace(' ', 0)
    print("Replaced null mileage with zeroes.")
    df = df.reset_index(drop=True)
    return df

# 6. REPLACE NULL STAY TIMES WITH ZEROES:
def replace_null_stay_times_with_zeroes(df):
    """
    This pipeline step replaces null stay times with zeroes.
    :param df: The dataframe to have its null stay times replaced with zeroes.
    :return: The dataframe with null stay times replaced with zeroes.
    """
    df['stay_time'] = df['stay_time']\
        .fillna(0)\
        .replace(' ', 0)\
        .replace('-', 0)
    print("Replaced null stay times with zeroes.")
    df = df.reset_index(drop=True)
    return df

# 7. REPLACE DAY N by THE N VALUE ONLY:
def replace_day_values(df):
    """
    This pipeline step replaces day values by their numeric values.
    :param df: The dataframe to have its day values replaced by their numeric values.
    :return: The dataframe with day values replaced by their numeric values.
    """
    df['date'] = df['date'].str[4:].astype(int)
    print("Replaced day values by their numeric values.")
    df = df.reset_index(drop=True)
    return df

# X. DEDUCE MILEAGES USING TRAINS SPEED METADATA:
def deduce_mileage_using_speed_and_travel_times(df):
    """
    This pipeline step deduces the mileage using the trains speed metadata.
    :param df: The dataframe to have its mileage deduced using the trains speed metadata.
    :return: The dataframe with mileage deduced using the trains speed metadata.
    """
    kmph_speed_dict = {
    'G': 350,
    'C': 350,
    'D': 260,
    'Z': 160,
    'T': 150,
    'K': 120,
    'others': 120
    }
    kmph_to_mph_factor = 0.621371
    mph_spped_dict = {}
    for key in kmph_speed_dict:
        mph_spped_dict[key] = kmph_speed_dict[key] * kmph_to_mph_factor

    counter = 0
    for index, row in df.iterrows():
        if index != 0 and row['train'] == df.iloc[index - 1]['train'] and row['mileage'] == 0:
            train_first_letter = row['train'][0]
            if train_first_letter in kmph_speed_dict:
                speed = mph_spped_dict[train_first_letter]
            else:
                speed = mph_spped_dict['others']
            travel_time = (datetime.datetime.strptime(row['arr_time'], '%H:%M:%S') - datetime.datetime.strptime(df.iloc[index - 1]['dep_time'], '%H:%M:%S')).seconds // 60
            mileage = speed * travel_time / 60
            df.at[index, 'mileage'] = int(mileage + int(df.at[index - 1, 'mileage']))
            counter += 1
    print("Deduced " + str(counter) + " mileages using trains speed metadata.")
    df = df.reset_index(drop=True)
    return df

# CHINESE RAILWAY PREPROCESSING PIPELINE:
def chinese_railway_preprocessing_pipeline(df, save_csv=False, output_path=None):
    """
    This function is the preprocessing pipeline for the Chinese Railway dataset.
    :param df: The dataframe to be preprocessed.
    :param save_csv: The boolean value that indicates whether to save the preprocessed dataframe as a csv file.
    :param output_path: The path to save the preprocessed dataframe as a csv file.
    :return:
    """
    # SETTINGS=
    original_size = len(df)

    # START:
    print("Preprocessing pipeline started.")

    # 1. REMOVE DUPLICATE ROWS:
    df = remove_duplicates(df)

    # 2. REMOVE EDGES THAT CONNECT THE SAME STATION:
    df = unnormalize_times(df)

    # 3. UNNORMALIZE THE DEPARTURE AND ARRIVAL TIMES:
    df = remove_trains_with_corrupted_times(df)

    # 4. REMOVE TRAINS WITH CORRUPTED TIMES:
    df = remove_edges_that_connect_the_same_station(df) # TODO: Discuss with teachers.

    # 5. REPLACE NULL MILEAGES WITH ZEROES:
    df = replace_null_mileages_with_zeroes(df)

    # 6. REPLACE NULL STAY TIMES WITH ZEROES:
    df = replace_null_stay_times_with_zeroes(df)

    # 7. REPLACE DAY N by THE N VALUE ONLY:
    df = replace_day_values(df)

    # Convert st_id to string:
    df = df.astype({'st_id': str})
    df = df.reset_index(drop=True)

    # X. DEDUCE MILEAGES USING TRAINS SPEED METADATA:
    # df = deduce_mileage_using_speed_and_travel_times(df)  # TODO: This is not working properly. Discuss with teachers.

    # END. RESET INDEXES:
    df = df.reset_index(drop=True)
    print("Preprocessing pipeline finished.")
    print("Remaining rows: " + str(len(df)) + " out of " + str(original_size) + " initially.")
    print("Successfully resetted indexes.")

    # BONUS. SAVE THE PREPROCESSED DATAFRAME TO A CSV FILE:
    if save_csv == True:
        df.to_csv(output_path, index=False)
        print("Saved dataframe to " + output_path)
    return df

# INDIAN RAILWAY PREPROCESSING PIPELINE:
def indian_railway_preprocessing_pipeline(df, save_csv=False, output_path=None):
    """
    This function is the preprocessing pipeline for the Indian Railway dataset.
    :param df: The dataframe to be preprocessed.
    :param save_csv: The boolean value that indicates whether to save the preprocessed dataframe as a csv file.
    :param output_path: The path to save the preprocessed dataframe as a csv file.
    :return:
    """
    # Remove unnecessary columns:
    df = df.drop(columns=['state', 'name', 'zone', 'address', 'train_name', 'station_name'])
    df = df.rename(columns={'train_number': 'train', 'id': 'st_no', 'station_code': 'st_id', 'departure': 'dep_time', 'arrival': 'arr_time', 'day': 'date', 'X': 'lon', 'Y': 'lat'})
    df = df[['train', 'st_no', 'st_id', 'date', 'arr_time', 'dep_time', 'lon', 'lat']]
    df.sort_values(by=['train', 'st_no'], inplace=True)
    df = df.reset_index(drop=True)
    print("Removed unnecessary columns.")

    # Remove duplicate rows:
    original_size = len(df)
    df = df.drop_duplicates(subset=['train', 'st_id', 'date', 'arr_time', 'dep_time', 'lon', 'lat'], keep='first')
    df = df.reset_index(drop=True)
    print("Removed " + str(original_size - len(df)) + " duplicated rows.")

    # Iterate through the df, and for each train, replace null 'day' values by the previous non-null value (only for each train):
    print("Replacing null 'day' values by the previous non-null value (only for each train)...")
    train_id = df.iloc[0]['train']
    train_ids_to_remove = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if train_id != row['train']:
            if no_time_values:
                train_ids_to_remove.append(df.iloc[index - 1]['train'])
            train_id = row['train']
            no_time_values = True
        if not row['date'] == "None":
            no_time_values = False
        if row['arr_time'] == "None" and row['dep_time'] == "None":
            df.at[index, 'date'] = df.iloc[index - 1]['date']
    for train_id in train_ids_to_remove:
        df = df[df.train_number != train_id]
    df['date'] = df['date'].astype(int)
    df = df.reset_index(drop=True)

    # Iterate through the df, and for each train, replace starting/ending arr/dep times with their corresponding values:
    print("Replacing starting/ending arr/dep times with their corresponding values...")
    train_id = df.iloc[0]['train']
    df.at[0, 'arr_time'] = df.iloc[0]['dep_time']
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if train_id != row['train']:
            train_id = row['train']
            df.at[index, 'arr_time'] = row['dep_time']
            df.at[index - 1, 'dep_time'] = df.iloc[index - 1]['arr_time']
    df.at[len(df) - 1, 'dep_time'] = df.iloc[len(df) - 1]['arr_time']
    df = df.reset_index(drop=True)

    # Remove trains whose both the arrival and departure times are null:
    print("Removing trains whose both the arrival and departure times are null...")
    corrupted_trains_ids = []
    train_id = df.iloc[0]['train']
    is_corrupted = False
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if train_id != row['train']:
            train_id = row['train']
            if is_corrupted:
                corrupted_trains_ids.append(df.iloc[index - 1]['train'])
            is_corrupted = False
        if row['arr_time'] == "None" and row['dep_time'] == "None":
            is_corrupted = True
    for train_id in corrupted_trains_ids:
        df = df[df.train != train_id]
    df = df.reset_index(drop=True)

    # Remove rows with null values for lat and lon:
    df = df[df.lat.notnull() & df.lon.notnull()]
    df = df.reset_index(drop=True)
    print("Removed rows with null values for lat and lon (lat == null and lon == null).")

    # Calculate the stay time for each row (times in format HH:MM:SS):
    print("Calculating the stay time for each row (times in format HH:MM:SS) using the arrival and departure times...")
    df['stay_time'] = None
    for index, row in tqdm(df.iterrows(), total=len(df)):
        df.at[index, 'stay_time'] = (datetime.datetime.strptime(row['dep_time'], '%H:%M:%S') - datetime.datetime.strptime(row['arr_time'], '%H:%M:%S')).seconds // 60
    df['stay_time'] = df['stay_time'].apply(lambda x: 0 if x < 0 else x)
    df = df.reset_index(drop=True)

    # Mileage calculation (using relative lat and lon values):
    print("Calculating mileage (using relative lat and lon values)...")
    df['mileage'] = None
    train_id = 0
    train_mileage_so_far = 0
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if train_id != row['train']:
            train_id = row['train']
            train_mileage_so_far = 0
        if index == 0:
            df.at[index, 'mileage'] = 0
        else:
            df.at[index, 'mileage'] = \
                3956 * 2 * math.asin(
                math.sqrt(math.sin((math.radians(row['lat']) - math.radians(df.iloc[index - 1]['lat']))/2)**2
                          + math.cos(math.radians(df.iloc[index - 1]['lat'])) * math.cos(math.radians(row['lat']))
                          * math.sin((math.radians(row['lon']) - math.radians(df.iloc[index - 1]['lon']))/2)**2)
            ) + train_mileage_so_far
            train_mileage_so_far = df.iloc[index]['mileage']
    df = df.reset_index(drop=True)

    # Convert st_id to string:
    df = df.astype({'st_id': str})
    df = df.reset_index(drop=True)

    # Save the result to 'schedules_with_station_names.csv' file:
    if save_csv:
        df.to_csv(output_path, index=False)
        print("Saved the result to: " + output_path)
    return df