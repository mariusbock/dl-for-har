##################################################
# All functions related to creating datasets
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
##################################################

from PyAV.av.io import read
import re

import numpy as np
import pandas as pd
from glob import glob
import os
from io import BytesIO
import zipfile

pd.options.mode.chained_assignment = None


def milliseconds_to_hertz(start, end, rate):
    """
    Function which converts milliseconds to hertz timestamps

    :param start: float
        Start time in milliseconds
    :param end: float
        End time in milliseconds
    :param rate: int
        Employed sampling rate during recording
    :return int, int
        Start and end time in hertz
    """
    adjusted_rate = rate / 1000
    return int(np.floor(float(start) * adjusted_rate)), int(np.floor(float(end) * adjusted_rate))


def create_wetlab_data_from_mkvs(feature_tracks, label_tracks, raw_dir, save_dir, sample_rate):
    """
    Function which creates the a csv file using the WetLab mkv dataset files as input.

    :param feature_tracks: lambda expression
        Tracks which contain the features that are to be used from the wetlab mkvs
    :param label_tracks: lambda expression
        Tracks which contain the labels that are to be used from the wetlab mkvs
    :param raw_dir: string
        Directory where the raw files are located
    :param save_dir: string
        Directory where the resulting csv is to be saved to
    :param sample_rate: int
        Sampling rate
    """
    filenames = sorted(glob(os.path.join(raw_dir, 'wetlab', '*.mkv')))
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
        action_label_output = pd.DataFrame(np.full(len(features), 'null_class'))
        tasks_label_output = pd.DataFrame(np.full(len(features), 'null_class'))
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
    output.columns = ['subject_id', 'acc_x', 'acc_y', 'acc_z', 'activity_label_1', 'activity_label_2']
    output = output.astype({'subject_id': int, 'acc_x': float, 'acc_y': float, 'acc_z': float, 'activity_label_1': str, 'activity_label_2': str})

    output.to_csv(os.path.join(save_dir, 'wetlab_data.csv'), index=False, header=False)


def create_sbhar_dataset(raw_dir, save_dir):
    """
    Function to create SBHAR dataset (containing only acceleration data).

    :param raw_dir: string
        Folder in which raw data is contained
    :param save_dir: string
        Folder where the resulting csv is to be saved to
    """

    label_dict = {
        0: 'null_class',
        1: 'walking',
        2: 'walking_upstairs',
        3: 'walking_downstairs',
        4: 'sitting',
        5: 'standing',
        6: 'lying',
        7: 'stand-to-sit',
        8: 'sit-to-stand',
        9: 'sit-to-lie',
        10: 'lie-to-sit',
        11: 'stand-to-lie',
        12: 'lie-to-stand'
    }

    labels = np.loadtxt(os.path.join(raw_dir, 'sbhar', 'labels.txt'), delimiter=' ')
    acc_data = [f for f in os.listdir(os.path.join(raw_dir, 'sbhar')) if 'acc' in f]
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
            acc_tmp_data = np.loadtxt(os.path.join(raw_dir, 'sbhar', acc_sbj_file), delimiter=' ')
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
    output_data = pd.DataFrame(output_data)
    output_data.columns = ['subject_id', 'acc_x', 'acc_y', 'acc_z', 'activity_label']
    output_data = output_data.replace({'activity_label': label_dict})
    output_data = output_data.astype({'subject_id': int, 'acc_x': float, 'acc_y': float, 'acc_z': float, 'activity_label': str})

    output_data.to_csv(os.path.join(save_dir, 'sbhar_data.csv'), index=False, header=False)


def create_hhar_dataset(raw_dir, save_dir):
    """
    Function to create HHAR dataset (containing only acceleration data).

    :param raw_dir: string
        Folder in which raw data is contained
    :param save_dir: string
        Folder where the resulting csv is to be saved to
    """
    data = pd.read_csv(os.path.join(raw_dir, 'hhar', 'Watch_accelerometer.csv'))

    user_dict = {
        'a': 0,
        'b': 1,
        'c': 2,
        'd': 3,
        'e': 4,
        'f': 5,
        'g': 6,
        'h': 7,
        'i': 8,
    }

    data = data.replace({"User": user_dict})

    data = data[['User', 'x', 'y', 'z', 'gt']]
    data = data.fillna('null_class')

    data.to_csv(os.path.join(save_dir, 'hhar_data.csv'), index=False, header=False)


def create_rwhar_dataset(raw_dir, save_dir):
    """
    Function to create RWHAR dataset (containing only acceleration data).

    Original Author : Sandeep Ramachandra, sandeep.ramachandra@student.uni-siegen.de
    Modified by: Marius Bock
    :param raw_dir: string
        Folder in which raw data is contained
    :param save_dir: string
        Folder where the resulting csv is to be saved to
    """
    RWHAR_ACTIVITY_NUM = {
        "climbingdown": 'climbing_down',
        "climbingup": 'climbing_up',
        "jumping": 'jumping',
        "lying": 'lying',
        "running": 'running',
        "sitting": 'sitting',
        "standing": 'standing',
        "walking": 'walking',
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
                path_acc = filepath + '/' + sub + "/acc" + activity_name  # concat the path to acc file for given activity and subject
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

    data = clean_rwhar(os.path.join(raw_dir, 'rwhar'), sel_location='forearm')
    data = data[['subject', 'acc_x', 'acc_y', 'acc_z', 'activity']]
    data = data.astype({'subject': int, 'acc_x': float, 'acc_y': float, 'acc_z': float, 'activity': str})

    data.to_csv(os.path.join(save_dir, 'rwhar_data.csv'), index=False, header=False)


def create_opportunity_dataset(raw_dir, save_dir):
    """
    Funtion to create Opportunity dataset. Saves both data used in Ordonez et al. as well as full dataset. Preprocessing
    same as Ordonez et al.

    :param raw_dir: string
        Folder in which raw data is contained
    :param save_dir: string
        Folder where the resulting csv is to be saved to
    """

    # Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
    NB_SENSOR_CHANNELS = 113

    # Hardcoded names of the files defining the OPPORTUNITY challenge data. As named in the original data.
    OPPORTUNITY_DATA_FILES = [  # Ordonez training
        (0, 'OpportunityUCIDataset/dataset/S1-Drill.dat'),
        (0, 'OpportunityUCIDataset/dataset/S1-ADL1.dat'),
        (0, 'OpportunityUCIDataset/dataset/S1-ADL2.dat'),
        (0, 'OpportunityUCIDataset/dataset/S1-ADL3.dat'),
        (0, 'OpportunityUCIDataset/dataset/S1-ADL4.dat'),
        (0, 'OpportunityUCIDataset/dataset/S1-ADL5.dat'),
        (1, 'OpportunityUCIDataset/dataset/S2-Drill.dat'),
        (1, 'OpportunityUCIDataset/dataset/S2-ADL1.dat'),
        (1, 'OpportunityUCIDataset/dataset/S2-ADL2.dat'),
        (2, 'OpportunityUCIDataset/dataset/S3-Drill.dat'),
        (2, 'OpportunityUCIDataset/dataset/S3-ADL1.dat'),
        (2, 'OpportunityUCIDataset/dataset/S3-ADL2.dat'),
        # Ordonez validation
        (1, 'OpportunityUCIDataset/dataset/S2-ADL3.dat'),
        (2, 'OpportunityUCIDataset/dataset/S3-ADL3.dat'),
        # Ordonez testing
        (1, 'OpportunityUCIDataset/dataset/S2-ADL4.dat'),
        (1, 'OpportunityUCIDataset/dataset/S2-ADL5.dat'),
        (2, 'OpportunityUCIDataset/dataset/S3-ADL4.dat'),
        (2, 'OpportunityUCIDataset/dataset/S3-ADL5.dat'),
        # additional data
        (3, 'OpportunityUCIDataset/dataset/S4-ADL1.dat'),
        (3, 'OpportunityUCIDataset/dataset/S4-ADL2.dat'),
        (3, 'OpportunityUCIDataset/dataset/S4-ADL3.dat'),
        (3, 'OpportunityUCIDataset/dataset/S4-ADL4.dat'),
        (3, 'OpportunityUCIDataset/dataset/S4-ADL5.dat'),
        (3, 'OpportunityUCIDataset/dataset/S4-Drill.dat')
    ]

    # Hardcoded thresholds to define global maximums and minimums for every one of the 113 sensor channels employed in the
    # OPPORTUNITY challenge
    NORM_MAX_THRESHOLDS = [3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
                           3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
                           3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
                           3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
                           3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
                           3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
                           3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
                           3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
                           3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
                           250, 25, 200, 5000, 5000, 5000, 5000, 5000, 5000,
                           10000, 10000, 10000, 10000, 10000, 10000, 250, 250, 25,
                           200, 5000, 5000, 5000, 5000, 5000, 5000, 10000, 10000,
                           10000, 10000, 10000, 10000, 250, ]

    NORM_MIN_THRESHOLDS = [-3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000,
                           -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000,
                           -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000,
                           -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000,
                           -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000,
                           -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000,
                           -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000,
                           -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000,
                           -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000,
                           -250, -100, -200, -5000, -5000, -5000, -5000, -5000, -5000,
                           -10000, -10000, -10000, -10000, -10000, -10000, -250, -250, -100,
                           -200, -5000, -5000, -5000, -5000, -5000, -5000, -10000, -10000,
                           -10000, -10000, -10000, -10000, -250, ]

    def select_columns_opp(data):
        """Selection of the 113 columns employed in the OPPORTUNITY challenge
        :param data: numpy integer matrix
            Sensor data (all features)
        :return: numpy integer matrix
            Selection of features
        """

        #                     included-excluded
        features_delete = np.arange(46, 50)
        features_delete = np.concatenate([features_delete, np.arange(59, 63)])
        features_delete = np.concatenate([features_delete, np.arange(72, 76)])
        features_delete = np.concatenate([features_delete, np.arange(85, 89)])
        features_delete = np.concatenate([features_delete, np.arange(98, 102)])
        features_delete = np.concatenate([features_delete, np.arange(134, 243)])
        features_delete = np.concatenate([features_delete, np.arange(244, 249)])
        return np.delete(data, features_delete, 1)

    def normalize(data, max_list, min_list):
        """Normalizes all sensor channels
        :param data: numpy integer matrix
            Sensor data
        :param max_list: numpy integer array
            Array containing maximums values for every one of the 113 sensor channels
        :param min_list: numpy integer array
            Array containing minimum values for every one of the 113 sensor channels
        :return:
            Normalized sensor data
        """
        max_list, min_list = np.array(max_list), np.array(min_list)
        diffs = max_list - min_list
        for i in np.arange(data.shape[1]):
            data[:, i] = (data[:, i] - min_list[i]) / diffs[i]
        #     Checking the boundaries
        data[data > 1] = 0.99
        data[data < 0] = 0.00
        return data

    def adjust_idx_labels(data_y, label):
        """Transforms original labels into the range [0, nb_labels-1]
        :param data_y: numpy integer array
            Sensor labels
        :param label: string, ['gestures' (default), 'locomotion']
            Type of activities to be recognized
        :return: numpy integer array
            Modified sensor labels
        """
        data_y[data_y == 0] = 'null_class'
        if label == 'locomotion':  # Labels for locomotion are adjusted
            data_y[data_y == 1] = 'stand'
            data_y[data_y == 2] = 'walk'
            data_y[data_y == 4] = 'sit'
            data_y[data_y == 5] = 'lie'
        elif label == 'gestures':  # Labels for gestures are adjusted
            data_y[data_y == 406516] = 'open_door_1'
            data_y[data_y == 406517] = 'open_door_2'
            data_y[data_y == 404516] = 'close_door_1'
            data_y[data_y == 404517] = 'close_door_2'
            data_y[data_y == 406520] = 'open_fridge'
            data_y[data_y == 404520] = 'close_fridge'
            data_y[data_y == 406505] = 'open_dishwasher'
            data_y[data_y == 404505] = 'close_dishwasher'
            data_y[data_y == 406519] = 'open_drawer_1'
            data_y[data_y == 404519] = 'close_drawer_1'
            data_y[data_y == 406511] = 'open_drawer_2'
            data_y[data_y == 404511] = 'close_drawer_2'
            data_y[data_y == 406508] = 'open_drawer_3'
            data_y[data_y == 404508] = 'close_drawer_3'
            data_y[data_y == 408512] = 'clean_table'
            data_y[data_y == 407521] = 'drink_from_cup'
            data_y[data_y == 405506] = 'toggle_switch'
        return data_y

    def process_dataset_file(data):
        """Function defined as a pipeline to process individual OPPORTUNITY files
        :param data: numpy integer matrix
            Matrix containing data samples (rows) for every sensor channel (column)
        :return: numpy integer matrix, numy integer array
            Processed sensor data, segmented into features (x) and labels (y)
        """

        # Select correct columns
        data = select_columns_opp(data)
        data = data[:, 1:]

        # Perform linear interpolation
        data[:, :113] = np.array([pd.Series(i).interpolate() for i in data[:, :113].T]).T

        # Remaining missing data are converted to zero
        data[:, :113][np.isnan(data[:, :113])] = 0

        # All sensor channels are normalized
        data[:, :113] = normalize(data[:, :113], NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)

        return data

    def generate_data(dataset, target_folder):
        """Function to read the OPPORTUNITY challenge raw data and process all sensor channels
        :param dataset: string
            Path with original OPPORTUNITY zip file
        :param target_folder: string
            Processed file
        """
        full = np.empty((0, NB_SENSOR_CHANNELS + 3))

        zf = zipfile.ZipFile(dataset)
        print('Processing dataset files ...')
        for sbj, filename in OPPORTUNITY_DATA_FILES:
            try:
                data = np.loadtxt(BytesIO(zf.read(filename)))
                print('... file {0}'.format(filename))
                _out = process_dataset_file(data)
                _sbj = np.full((len(_out), 1), sbj)
                _out = np.concatenate((_sbj, _out), axis=1)
                full = np.vstack((full, _out))
                print(filename, full.shape)
            except KeyError:
                print('ERROR: Did not find {0} in zip file'.format(filename))

        # Dataset is segmented into train and test
        nb_test_samples = 676713

        print("Final dataset with size: {0} ".format(full.shape))
        # write full dataset
        full_data = pd.DataFrame(full, index=None)
        full_data = full_data.astype({0: int})
        # adjust labels to be counting from 1
        full_data[114] = adjust_idx_labels(full_data[114], 'locomotion')
        full_data[115] = adjust_idx_labels(full_data[115], 'gestures')

        full_data.to_csv(os.path.join(target_folder, 'opportunity_data.csv'), index=False, header=False)
        # write Ordonez split
        full_data.iloc[:nb_test_samples, :].to_csv(os.path.join(target_folder, 'opportunity_ordonez_data.csv'), index=False, header=False)

    generate_data(os.path.join(raw_dir, 'opportunity', 'OpportunityUCIDataset.zip'), save_dir)


if __name__ == '__main__':
    raw_path = '../data/raw'
    save_path = '../data'

    # opportunity
    create_opportunity_dataset(raw_path, save_path)
    # wetlab
    feat = lambda streams: [s for s in streams if s.type == "audio"]
    label = lambda streams: [s for s in streams if s.type == "subtitle"]
    create_wetlab_data_from_mkvs(feat, label, raw_path, save_path, 50)
    # sbhar
    create_sbhar_dataset(raw_path, save_path)
    # hhar
    create_hhar_dataset(raw_path, save_path)
    # rwhar
    create_rwhar_dataset(raw_path, save_path)
