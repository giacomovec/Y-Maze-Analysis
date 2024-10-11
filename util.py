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

def remove_points_below_likelihood_threshold(df, columns, likelihood_threshold=0.80):
    for column in columns:
        if column + "_likelihood" in df.columns:
            likelihood_column = column + "_likelihood"
            df.loc[
                df[likelihood_column] < likelihood_threshold,
                [column + "_x", column + "_y"],
            ] = np.nan
    return df

def create_dataset(directory):
    dataset = []
    print("Creating Dataset...")
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".h5"):
            temp_df = load_data(f"{directory}/{filename}")
            temp_df = arm_tracking(temp_df)
            clean_name = filename.split("DLC")[0]
            group = clean_name.split("_")[0]
            id = group + clean_name.split("_")[1]
            day = clean_name.split("_")[2]
            avg_pdist = temp_df["p_dist"].mean()
            temp_df = temp_df[temp_df["arm"] != "center"]
            temp_df.reset_index(inplace=True)
            alt = 0
            for i in range(len(temp_df) - 1):
                if temp_df.loc[i, "arm"] == "C" and temp_df.loc[i+1, "arm"] == "B":
                    alt = 1
                    break
                elif temp_df.loc[i, "arm"] == "C" and temp_df.loc[i+1, "arm"] == "A":
                    alt = 0
                    break

            if 'C' in temp_df['arm'].values:
                time_in_c = temp_df["arm"].value_counts()["C"]
            else:
                time_in_c = 0  # Or any other appropriate value if 'C' is not found

            if 'A' in temp_df['arm'].values:
                time_in_a = temp_df["arm"].value_counts()["A"]
            else:
                time_in_a = 0  # Or any other appropriate value if 'C' is not found

            if 'B' in temp_df['arm'].values:
                time_in_b = temp_df["arm"].value_counts()["B"]
            else:
                time_in_b = 0  # Or any other appropriate value if 'C' is not found

            c_arm_first_entry = temp_df[temp_df['arm'] == 'B'].index.min()
            
            if pd.notnull(c_arm_first_entry):
                frames_in_c_arm = temp_df.loc[c_arm_first_entry:].shape[0]
            else:
                frames_in_c_arm = 0

            correct_arm_entry = temp_df[(temp_df['arm'] == 'B') | (temp_df['arm'] == 'A')].index.min()
            if pd.notnull(correct_arm_entry):
                frames_to_correct_arm = correct_arm_entry - c_arm_first_entry
            else:
                frames_to_correct_arm = None

            if 'B' in temp_df['arm'].values:
                reentries_c_arm = ((temp_df['arm'] == 'B') & (temp_df['arm'].shift(1) != 'B')).sum()
            else:
                reentries_c_arm = 0  # Or any other appropriate value if 'C' is not found           

            perc_b = time_in_b/(time_in_c+time_in_a+time_in_b)



            dataset.append([group, id, day, avg_pdist, clean_name, alt, time_in_b, frames_in_c_arm, frames_to_correct_arm, reentries_c_arm, perc_b])
    # Create a DataFrame from the list
    print(dataset)
    df = pd.DataFrame(dataset, columns=["group", "id", "day", "avg_pdist", "file_name", "forced", "time_in_b", "first_b", "latency", "reentries", "perc_b"])
    df["group"] = df["group"].str.replace("g", "")
    df["day"] = df["day"].str.replace("d", "")
    df["condition"] = df["id"].str[:-1]
    df["condition"] = df["condition"].str[2:]
    return df



