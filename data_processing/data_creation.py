from PyAV.av.io import read
import re

import numpy as np
import pandas as pd
from glob import glob
import os
from io import BytesIO
import zipfile


def milliseconds_to_hertz(start, end, rate):
    """
    Function which converts milliseconds to hertz timestamps

    :param start: start time in milliseconds
    :param end: end time in milliseconds
    :param rate: employed sampling rate during recording
    :return: start and end time in hertz
    """
    adjusted_rate = rate / 1000
    return int(np.floor(float(start) * adjusted_rate)), int(np.floor(float(end) * adjusted_rate))


def create_wetlab_data_from_mkvs(feature_tracks, label_tracks, directory, sample_rate):
    """
    Funtion which creates the a csv file using the WetLab mkv dataset files as input.
    :param feature_tracks: tracks which contain the features that are to be used from the wetlab mkvs
    :param label_tracks: tracks which contain the labels that are to be used from the wetlab mkvs
    :param directory: directory where the resulting csv is to be saved to
    :param sample_rate: sampling rate 
    :return: pandas dataframe containing wetlab features
    """
    filenames = sorted(glob(os.path.join(directory, '*.mkv')))
    # obtain unique labels
    unique_labels = []
    for filename in filenames:
        unique_labels = np.unique(
            np.concatenate((unique_labels, np.vstack(read(label_tracks, file=filename)[0])[:, 2])))
    output = pd.DataFrame()
    for i, filename in enumerate(filenames):
        features = np.vstack(read(feature_tracks, file=filename)[0])
        features = features / 2 ** 16 * 8 * 9.81
        labels = np.vstack(read(label_tracks, file=filename)[0])
        idx = np.full(len(features), i)
        feat_output = pd.DataFrame(np.concatenate((np.array(idx)[:, None], features), axis=1))
        action_label_output = pd.DataFrame(np.full(len(features), 0))
        tasks_label_output = pd.DataFrame(np.full(len(features), 0))
        for label_triplet in labels:
            start, end = milliseconds_to_hertz(label_triplet[0], label_triplet[1], sample_rate)
            if any(char.isdigit() for char in label_triplet[2]):
                tasks_label_output[start:end] = label_triplet[2]
            else:
                action_label_output[start:end] = label_triplet[2]
        temp_output = pd.concat((feat_output, action_label_output, tasks_label_output), axis=1)
        if i == 0:
            output = temp_output
        else:
            output = pd.concat((output, temp_output), axis=0)
        print("Processed file: {0}".format(filename))
    print("Value counts (Actions): ")
    print(output.iloc[:, -2].value_counts())
    print("Value counts (Tasks): ")
    print(output.iloc[:, -1].value_counts())
    return output


def create_sbhar_dataset(folder):
    labels = np.loadtxt(os.path.join(folder, 'labels.txt'), delimiter=' ')
    acc_data = [f for f in os.listdir(folder) if 'acc' in f]
    # gyro_data = [f for f in os.listdir(folder) if 'gyro' in f]
    output_data = None

    for sbj in range(30):
        if sbj < 9:
            acc_sbj_files = [f for f in acc_data if 'user0' + str(sbj + 1) in f]
            # gyro_sbj_files = [f for f in gyro_data if 'user0' + str(sbj + 1) in f]
        else:
            acc_sbj_files = [f for f in acc_data if 'user' + str(sbj + 1) in f]
            # gyro_sbj_files = [f for f in gyro_data if 'user' + str(sbj + 1) in f]
        sbj_data = None
        # acc + gyro
        for acc_sbj_file in acc_sbj_files:
            acc_tmp_data = np.loadtxt(os.path.join(folder, acc_sbj_file), delimiter=' ')
            sbj = re.sub('[^0-9]', '', acc_sbj_file.split('_')[2])
            exp = re.sub('[^0-9]', '', acc_sbj_file.split('_')[1])
            # gyro_tmp_data = np.loadtxt(os.path.join(folder, 'gyro_exp' + exp + '_user' + sbj + '.txt'), delimiter=' ')
            sbj_labels = labels[(labels[:, 0] == int(exp)) & (labels[:, 1] == int(sbj))]
            # tmp_data = np.concatenate((acc_tmp_data, gyro_tmp_data), axis=1)
            tmp_data = np.concatenate((acc_tmp_data, np.zeros(acc_tmp_data.shape[0])[:, None]), axis=1)
            for label_triplet in sbj_labels:
                tmp_data[int(label_triplet[3]):int(label_triplet[4] + 1), -1] = label_triplet[2]
            tmp_data = np.concatenate((np.full(tmp_data.shape[0], int(sbj) - 1)[:, None], tmp_data), axis=1)
            if sbj_data is None:
                sbj_data = tmp_data
            else:
                sbj_data = np.concatenate((sbj_data, tmp_data), axis=0)
        if output_data is None:
            output_data = sbj_data
        else:
            output_data = np.concatenate((output_data, sbj_data), axis=0)
    return pd.DataFrame(output_data, index=None)


def create_hhar_dataset(folder):
    data = pd.read_csv(os.path.join(folder, 'Watch_accelerometer.csv'))

    user_dict = {
        'a': 0.0,
        'b': 1.0,
        'c': 2.0,
        'd': 3.0,
        'e': 4.0,
        'f': 5.0,
        'g': 6.0,
        'h': 7.0,
        'i': 8.0,
    }

    data = data.replace({"User": user_dict})

    data = data[['User', 'x', 'y', 'z', 'gt']]
    data = data.fillna(0)
    return data


def create_rwhar_dataset(folder):
    """
    Author : Sandeep Ramachandra, sandeep.ramachandra@student.uni-siegen.de
    :return:
    """
    RWHAR_ACTIVITY_NUM = {
        "climbingdown": 1,
        "climbingup": 2,
        "jumping": 3,
        "lying": 4,
        "running": 5,
        "sitting": 6,
        "standing": 7,
        "walking": 8,
    }

    RWHAR_BAND_LOCATION = {
        "chest": 1,
        "forearm": 2,
        "head": 3,
        "shin": 4,
        "thigh": 5,
        "upperarm": 6,
        "waist": 7,
    }

    def check_rwhar_zip(path):
        # verify that the path is to the zip containing csv and not another zip of csv

        if any(".zip" in filename for filename in zipfile.ZipFile(path, "r").namelist()):
            # There are multiple zips in some cases
            with zipfile.ZipFile(path, "r") as temp:
                path = BytesIO(temp.read(
                    max(temp.namelist())))  # max chosen so the exact same acc and gyr files are selected each time (repeatability)
        return path

    def rwhar_load_csv(path):
        # Loads up the csv at given path, returns a dictionary of data at each location

        path = check_rwhar_zip(path)
        tables_dict = {}
        with zipfile.ZipFile(path, "r") as Zip:
            zip_files = Zip.namelist()

            for csv in zip_files:
                if "csv" in csv:
                    location = RWHAR_BAND_LOCATION[
                        csv[csv.rfind("_") + 1:csv.rfind(".")]]  # location is between last _ and .csv extension
                    sensor = csv[:3]
                    prefix = sensor.lower() + "_"
                    table = pd.read_csv(Zip.open(csv))
                    table.rename(columns={"attr_x": prefix + "x",
                                          "attr_y": prefix + "y",
                                          "attr_z": prefix + "z",
                                          "attr_time": "timestamp",
                                          }, inplace=True)
                    table.drop(columns="id", inplace=True)
                    tables_dict[location] = table

        return tables_dict

    def rwhar_load_table_activity(path_acc):
        # Logic for loading each activity zip file for acc and gyr and then merging the tables at each location

        acc_tables = rwhar_load_csv(path_acc)
        data = pd.DataFrame()

        for loc in acc_tables.keys():
            acc_tab = acc_tables[loc]

            acc_tab = pd.DataFrame(acc_tab)
            acc_tab["location"] = loc

            data = data.append(acc_tab)

        return data

    def clean_rwhar(filepath, sel_location=None):
        # the function reads the files in RWHAR dataset and each subject and each activity labelled in a panda table
        # filepath is the parent folder containing all the RWHAR dataset.
        # Note: all entries are loaded but their timestamps are not syncronised. So a single location must be selected and
        # all entries with NA must be dropped.

        subject_dir = os.listdir(filepath)
        dataset = pd.DataFrame()

        for sub in subject_dir:
            if "proband" not in sub:
                continue
            #         files = os.listdir(filepath+sub)
            #         files = [file for file in files if (("acc" in file or "gyr" in file) and "csv" in file)]
            subject_num = int(sub[7:]) - 1  # proband is 7 letters long so subject num is number following that
            sub_pd = pd.DataFrame()

            for activity in RWHAR_ACTIVITY_NUM.keys():  # pair the acc and gyr zips of the same activity
                activity_name = "_" + activity + "_csv.zip"
                path_acc = filepath + sub + "/acc" + activity_name  # concat the path to acc file for given activity and subject
                table = rwhar_load_table_activity(path_acc)
                table["activity"] = RWHAR_ACTIVITY_NUM[activity]  # add a activity column and fill it with activity num
                sub_pd = sub_pd.append(table)

            sub_pd["subject"] = subject_num  # add subject id to all entries
            dataset = dataset.append(sub_pd)
            dataset = dataset.dropna()
            dataset = dataset[dataset.location == RWHAR_BAND_LOCATION[sel_location]]

        if sel_location is not None:
            print("Selecting location : ", sel_location)
            dataset = dataset[dataset.location == RWHAR_BAND_LOCATION[sel_location]]
            dataset = dataset.drop(columns="location")

        dataset = dataset.sort_values(by=['subject', 'timestamp'])
        dataset = dataset.drop(columns="timestamp")
        dataset = dataset.dropna()
        print(dataset['activity'].value_counts())
        return dataset

    data = clean_rwhar(folder, sel_location='forearm')
    data = data[['subject', 'acc_x', 'acc_y', 'acc_z', 'activity']]
    return data


if __name__ == '__main__':
    # wetlab
    feat = lambda streams: [s for s in streams if s.type == "audio"]
    label = lambda streams: [s for s in streams if s.type == "subtitle"]
    create_wetlab_data_from_mkvs(feat, label, '../data/raw/wetlab', 50).to_csv(
        '../data/wetlab_data.csv', index=False, header=False)
    # sbhar
    create_sbhar_dataset('../data/raw/sbhar').to_csv(
        '../data/sbhar_data.csv', index=False, header=False)
    # hhar
    create_hhar_dataset('../data/raw/hhar').to_csv(
        '../data/hhar_data.csv', index=False, header=False)
    # rwhar
    create_rwhar_dataset('../data/raw/rwhar/').to_csv(
        '../data/rwhar_data.csv', index=False, header=False)
