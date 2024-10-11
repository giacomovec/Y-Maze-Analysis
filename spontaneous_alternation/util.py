import cv2
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm_notebook as tqdm
import os


### UTILITY FUNCTIONS FOR OTHER SCRIPTS


warnings.simplefilter(action="ignore", category=FutureWarning)


def load_data(data_path):
    df = pd.read_hdf(data_path)
    df.columns = df.columns.droplevel()
    new_cols = [("{0}_{1}".format(*tup)) for tup in df.columns]
    df.columns = new_cols
    df.sort_index(axis="columns")
    columns_to_process = [
        "nose",
        "ear_left",
        "ear_right",
        "tail_base",
        "center_bottom",
        "center_left",
        "center_right",
    ]

    df = remove_points_below_likelihood_threshold(df, columns_to_process)
    df = df.interpolate()

    df["y_center_x"] = (
        df["center_left_x"] + df["center_bottom_x"] + df["center_right_x"]
    ) / 3
    df["y_center_y"] = (
        df["center_left_y"] + df["center_bottom_y"] + df["center_right_y"]
    ) / 3

    df["centroid_x"] = (df["nose_x"] + df["tail_base_x"]) / 2
    df["centroid_y"] = (df["nose_y"] + df["tail_base_y"]) / 2

    df["p_dist"] = euclid(df)

    df = df.interpolate()

    return df


def arm_tracking(df):
    

    df["arm"] = ""
    for index, row in df.iterrows():
        if row["nose_y"] < (row["center_right_y"] + row["center_left_y"])/2 and row["tail_base_y"] < (row["center_right_y"] + row["center_left_y"])/2:
            df.loc[index, "arm"] = "A"
        elif row["nose_x"] > (row["center_right_x"] + row["center_bottom_x"])/2 and row["tail_base_x"] > (row["center_right_x"] + row["center_bottom_x"])/2:
            df.loc[index, "arm"] = "B"
        elif row["nose_x"] < (row["center_left_x"] + row["center_bottom_x"])/2 and row["tail_base_x"] < (row["center_left_x"] + row["center_bottom_x"])/2:
            df.loc[index, "arm"] = "C"
        else:
            df.loc[index, "arm"] = "center"
    return df

def euclid(df):
    return np.sqrt(
        (df["centroid_x"] - df["centroid_x"].shift(1)) ** 2
        + (df["centroid_y"] - df["centroid_y"].shift(1)) ** 2
    )


def periods(df):
    periods = []
    value = None
    length = 0

    for i, row in df.iterrows():
        if row["arm"] == value:
            length += 1
        else:
            if value is not None:
                periods.append({"arm": value, "length_period": length})
            value = row["arm"]
            length = 1

    if value is not None:
        periods.append({"arm": value, "length_period": length})

    return pd.DataFrame(periods).reset_index(drop=True)


def remove_points_below_likelihood_threshold(df, columns, likelihood_threshold=0.80):
    for column in columns:
        if column + "_likelihood" in df.columns:
            likelihood_column = column + "_likelihood"
            df.loc[
                df[likelihood_column] < likelihood_threshold,
                [column + "_x", column + "_y"],
            ] = np.nan
    return df


def create_periods_df(df):
    periods_df = periods(df)
    periods_df = periods_df[periods_df['length_period'] > 50]  

    periods_df = periods_df[periods_df["arm"] != "center"]

    periods_df = periods_df.reset_index(drop=True)
    periods_df["sa"] = ""
    for index, row in periods_df.iterrows():
        if index > 1:
            if row["arm"] != periods_df.iloc[index - 1]["arm"]:
                if (
                    periods_df.loc[index - 1, "arm"] != periods_df.loc[index - 2,"arm"]
                    and row["arm"] != periods_df.loc[index - 2,"arm"]
                ):
                    periods_df.loc[index, ["sa"]] = "correct"
                elif periods_df.loc[index - 1,"arm"] == periods_df.loc[index - 2,"arm"]:
                    periods_df.loc[index, ["sa"]] = "neutral"
                else:
                    periods_df.loc[index, ["sa"]] = "incorrect"

            else:
                periods_df.loc[index, "sa"]= "incorrect"
    
    return(periods_df)


def create_dataset(directory):
    dataset = []
    print("Creating Dataset...")
    
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".h5"):
            df = load_data(f"{directory}/{filename}")
            #df1 = normalize_dataframe(df)

            df1 = arm_tracking(df)
            clean_name = filename.split("DLC")[0]
            df1.to_csv(f"{directory}/{clean_name}.csv", index=False)
            print(clean_name)
            group = clean_name.split("_")[0]
            id = group + clean_name.split("_")[1]
            day = clean_name.split("_")[2]
            avg_pdist = df1["p_dist"].mean()

            dataset.append([group, id, day, avg_pdist, clean_name])



    # Create a DataFrame from the list
    df = pd.DataFrame(dataset, columns=["group", "id", "day", "avg_pdist", "file_name"])
    df["group"] = df["group"].str.replace("g", "")
    df["day"] = df["day"].str.replace("d", "")
    df["condition"] = df["id"].str[:-1]
    df["condition"] = df["condition"].str[2:]
    return df


def add_metrics(df, directory):
    print("Calculating Metrics...")

    df["n_a"] = 0
    df["n_b"] = 0
    df["n_c"] = 0
    df["n_center"] = 0
    df["correct"] = 0
    df["incorrect"] = 0
    df["neutral"] = 0

    df["sa_rate"] = 0
    df['len_track'] = 0

    for i, row in tqdm(df.iterrows()):
        temp_df = pd.read_csv(f"{directory}/{row['file_name']}.csv")
        df.loc[i, 'len_track'] = len(temp_df)
        try:
            df.loc[i, 'n_a'] = temp_df["arm"].value_counts()["A"]
        except:
            print(f"{row['file_name']}: no A")
        try:
            df.loc[i, 'n_b'] = temp_df["arm"].value_counts()["B"]
        except:
            print(f"{row['file_name']}: no B")

        df.loc[i, 'n_c'] = temp_df["arm"].value_counts()["C"]
        try:
            df.loc[i, 'n_center'] = temp_df["arm"].value_counts()["center"]
        except:
            print(f"{row['file_name']}: no Center")
        periods_temp = create_periods_df(temp_df)

        try:
            df.loc[i, 'correct'] = periods_temp['sa'].value_counts()["correct"]
        except:
            print(f"{row['file_name']}: no Correct")
        try:
            df.loc[i, 'incorrect'] = periods_temp['sa'].value_counts()["incorrect"]
        except:
            print(f"{row['file_name']}: no Incorrect")
        try:
            df.loc[i, 'neutral'] = periods_temp['sa'].value_counts()["neutral"]
        except:
            print(f"{row['file_name']}: no Neutral")

                
        df.loc[i, 'sa_rate'] = df.loc[i, 'correct']/(df.loc[i, 'correct']+df.loc[i, 'incorrect']) #df.loc[i, 'correct'] / df.loc[i,"len_track"]

    return df
    
def combine_rows(df):

    rows_to_add = []
    for i, row in df.iterrows():
        if row["file_name"].split("_")[-1] == "1":
            new_group = df.loc[i, 'group']
            new_id = df.loc[i, 'id']
            new_day = df.loc[i, 'day']
            new_avgpdist = np.mean([df.loc[i, 'avg_pdist'], df.loc[i+1, 'avg_pdist']])
            new_filename = df.loc[i, 'file_name'][0:-2]
            new_condition = df.loc[i, 'condition']
            new_na = df.loc[i, 'n_a'] + df.loc[i+1, 'n_a']
            new_nb = df.loc[i, 'n_b'] + df.loc[i+1, 'n_b']
            new_nc = df.loc[i, 'n_c'] + df.loc[i+1, 'n_c']
            new_ncent = df.loc[i, 'n_center'] + df.loc[i+1, 'n_center']
            new_correct = df.loc[i, 'correct'] + df.loc[i+1, 'correct']
            new_incorrect = df.loc[i, 'incorrect'] + df.loc[i+1, 'incorrect']
            new_neutral = df.loc[i, 'neutral'] + df.loc[i+1, 'neutral']
            new_sarate = np.mean([df.loc[i, "sa_rate"], df.loc[i+1, "sa_rate"]])  
            new_lentrack = df.loc[i, "len_track"] + df.loc[i+1, "len_track"]
            rows_to_add.append([new_group, new_id, new_day, new_avgpdist, new_filename, new_condition, new_na, new_nb, new_nc,new_ncent,new_correct,new_incorrect,new_neutral, new_sarate,new_lentrack])


    rows_to_add = pd.DataFrame(rows_to_add, columns=["group", "id","day", "avg_pdist", "file_name", "condition", "n_a", "n_b", "n_c", "n_center","correct","incorrect","neutral", "sa_rate","len_track"])
    df = pd.concat([df, rows_to_add])
    to_drop = []
    for i, row in df.iterrows():
        if row["file_name"].split("_")[-1] == "1" or row["file_name"].split("_")[-1] == "2":
            to_drop.append(i)

    df = df.drop(to_drop)
    df = df.reset_index(drop=True)
    return df


def overlay_pose_estimation(frame, pandas_row, frame_width, frame_height):
    to_drop = []

    for name, value in pandas_row.items():
        if name.split("_")[-1] == "likelihood":
            to_drop.append(name)
    to_drop.append("arm")
    to_drop.append("p_dist")
    data = pandas_row.drop(labels=to_drop)
    points = [
        (
            int(float(data[i])),
            int(float(data[i + 1])),
        )
        for i in range(0, len(data), 2)
    ]
    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)  # Green circles for points



    return frame