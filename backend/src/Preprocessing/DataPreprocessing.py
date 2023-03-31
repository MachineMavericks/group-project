import pandas as pd
import datetime

# 1. REMOVE DUPLICATED ROWS:
def remove_duplicates(df):
    df = df.drop_duplicates()
    print("Removed " + str(len(df) - len(df.drop_duplicates())) + " duplicated row(s).")
    df = df.reset_index(drop=True)
    return df

# 2. REMOVE EDGES THAT CONNECT THE SAME STATION:
def remove_edges_that_connect_the_same_station(df):
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
    df['mileage'] = df['mileage'].fillna(0)
    df['mileage'] = df['mileage'].replace(' ', 0)
    print("Replaced null mileage with zeroes.")
    df = df.reset_index(drop=True)
    return df

# 6. REPLACE NULL STAY TIMES WITH ZEROES:
def replace_null_stay_times_with_zeroes(df):
    df['stay_time'] = df['stay_time']\
        .fillna(0)\
        .replace(' ', 0)\
        .replace('-', 0)
    print("Replaced null stay times with zeroes.")
    df = df.reset_index(drop=True)
    return df

# X. DEDUCE MILEAGES USING TRAINS SPEED METADATA:
def deduce_mileage_using_speed_and_travel_times(df):
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


def preprocessing_pipeline(df, save_csv=False, path=None):

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

    # X. DEDUCE MILEAGES USING TRAINS SPEED METADATA:
    # df = deduce_mileage_using_speed_and_travel_times(df)  # TODO: This is not working properly. Discuss with teachers.

    # END. RESET INDEXES:
    df = df.reset_index(drop=True)
    print("Preprocessing pipeline finished.")
    print("Remaining rows: " + str(len(df)) + " out of " + str(original_size) + " initially.")
    print("Successfully resetted indexes.")

    # BONUS. SAVE THE PREPROCESSED DATAFRAME TO A CSV FILE:
    if save_csv == True:
        df.to_csv(path, index=False)
        print("Saved dataframe to " + path)
    return df


def main():
    pass
if __name__ == '__main__':
    main()