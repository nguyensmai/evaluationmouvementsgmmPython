import os
import csv
import numpy as np
import scipy.signal
from scipy.interpolate import splev, splrep
import json
import math
from dtaidistance import dtw_ndim


# didn't work for some reason! didn't have time to debug further, but function left here for possible future develpment
def remove_idle_timeframes(data, window_size=21, threshold=100, dec=10):
    count = 0
    sum_sigma = 0
    # data = data[:,[1,2,3]]
    # avg_changes = []
    l = len(data)
    for x in [0, 3, *range(9, l, 3)]:
        data[x:x+3, :] = data[x:x+3, :] - data[6:9, :]
    for t in range(math.ceil(window_size / 2), data.shape[1] - math.floor(window_size / 2) + 1):
        sigma = 0
        w = math.floor(window_size / 2)
        sum_changes = 0
        for d in range(int(data.shape[0])):
            muMan = data[d * 4: d * 4 + 4, t - 1:t]
            sigma = sigma + np.linalg.norm(data[d * 3:d * 3 + 3, t - w - 1:t + w]) ** 2
            # change = abs(data[d][t + 1] - data[d][t])
            # sum_changes = sum_changes + change
        # avg_changes.append(sum_changes / data.shape[0])
        sigma = sigma / window_size
        # sum_sigma = sum_sigma + sigma
        # count = count + 1
        if sigma > threshold:
            deb = max(1, t - dec)
            out = data[:, deb - 1:]
            return out, deb
    deb = 1
    out = data
    return out, deb


def dynamic_time_warping(reference_data, data):
    warping_path = dtw_ndim.warping_path(reference_data, data)
    aligned_path = np.zeros(len(data))
    p = np.array(warping_path)[:, 0]
    q = np.array(warping_path)[:, 1]
    for index in range(len(data)):
        ind = np.where(p == index)[0]
        if ind.size == 0:
            aligned_path[index] = p[index]
        else:
            aligned_path[index] = q[ind[0]]
    aligned_path = aligned_path.astype(int)
    return data[aligned_path]


def prepare_kinect_data(filepath, number_of_datapoints = 100, filtering = 1, remove_start = 0):
    with open(filepath, 'r+', encoding='utf-8') as file:
        data_read = np.array([i[:-2].split(' ') for i in file.readlines()]).astype(float)
    number_of_timeframes = len(data_read)
    if number_of_timeframes < number_of_datapoints:
        return None
    data_read = np.reshape(data_read, (len(data_read), 25, 7))
    data_read = data_read[:, :, 0:3]
    data_read = np.reshape(data_read, (len(data_read), 75))

    if filtering == 1:
        b, a = scipy.signal.butter(3, 0.05)
        data_read = scipy.signal.lfilter(b, a, data_read, axis=0)

    if remove_start == 1:
        (out, deb) = remove_idle_timeframes(data_read.T, window_size=21, threshold=7, dec=10)
        data_read = data_read.T[:, deb - 1:].T

    length = data_read.shape[0]
    new_x = np.linspace(1, length, number_of_datapoints)
    data_read = np.array([splev(new_x, splrep(range(1, length + 1), line, k=3)) for line in data_read.transpose()])
    return data_read.transpose()


def prepare_openpose_data(filepath, number_of_datapoints = 100, filtering = 1, remove_start = 0):
    with open(filepath, 'r') as file:
        data = json.loads(file.read())
    data = data["positions"]
    number_of_timeframes = len(data)
    if number_of_timeframes < number_of_datapoints:
        return None
    body_parts = ['Head', 'mShoulder', 'rShoulder', 'rElbow', 'rWrist', 'lShoulder', 'lElbow', 'lWrist', 'rHip', 'rKnee',
                  'rAnkle', 'lHip', 'lKnee', 'lAnkle']
    data_read = np.zeros((number_of_timeframes, len(body_parts) * 2))
    for timeframe_id, (timestamp, readings) in enumerate(data.items()):

        for body_part_id, body_part in enumerate(body_parts):
            if body_part in readings:
                data_read[timeframe_id, 2 * body_part_id] = readings[body_part][0]
                data_read[timeframe_id, 2 * body_part_id + 1] = readings[body_part][1]

    for i in range(1, data_read.shape[1]):
        col = data_read[:, i]

        x = []
        v = []
        xq = []

        for j in range(number_of_timeframes):
            if not math.isclose(col[j], 0, abs_tol=1e-10):
                x.append(j)
                v.append(data_read[j, i])
            else:
                xq.append(j)

        if xq and len(x) > 1:
            vq = np.interp(xq, x, v)
            for j in range(len(xq)):
                data_read[xq[j], i] = vq[j]

    if filtering == 1:
        b, a = scipy.signal.butter(3, 0.05)
        data_read = scipy.signal.lfilter(b, a, data_read, axis=0)

    if remove_start == 1:
        (out, deb) = remove_idle_timeframes(data_read.T, window_size=21, threshold=7, dec=10)
        data_read = data_read.T[:, deb - 1:].T

    length = data_read.shape[0]
    new_x = np.linspace(1, length, number_of_datapoints)
    data_read = np.array([splev(new_x, splrep(range(1, length + 1), line, k=3)) for line in data_read.transpose()])
    return data_read.transpose()


def prepare_blazepose_data(filepath, number_of_datapoints = 100, filtering = 1, remove_start = 0):
    with open(filepath, 'r') as file:
        data = json.load(file)
    data = data["positions"]
    number_of_timeframes = len(data)
    if number_of_timeframes < number_of_datapoints:
        return None
    body_parts = ['Nose', 'Left_eye_inner', 'Left_eye', 'Left_eye_outer', 'Right_eye_inner', 'Right_eye',
                  'Right_eye_outer', 'Left_ear', 'Right_ear', 'Mouth_left', 'Mouth_right', 'Left_shoulder',
                  'Right_shoulder', 'Left_elbow', 'Right_elbow', 'Left_wrist', 'Right_wrist', 'Left_pinky', 'Right_pinky',
                  'Left_index', 'Right_index', 'Left_thumb', 'Right_thumb', 'Left_hip', 'Right_hip', 'Left_knee',
                  'Right_knee', 'Left_ankle', 'Right_ankle', 'Left_heel', 'Right_heel', 'Left_foot_index', 'Right_foot_index']
    data_read = np.zeros((number_of_timeframes, len(body_parts) * 3))
    for timeframe_id, (timestamp, readings) in enumerate(data.items()):

        for body_part_id, body_part in enumerate(body_parts):
            if body_part in readings:
                data_read[timeframe_id, 3 * body_part_id] = readings[body_part][0]
                data_read[timeframe_id, 3 * body_part_id + 1] = readings[body_part][1]
                data_read[timeframe_id, 3 * body_part_id + 2] = readings[body_part][2]

    for i in range(1, data_read.shape[1]):
        col = data_read[:, i]

        x = []
        v = []
        xq = []

        for j in range(number_of_timeframes):
            if not math.isclose(col[j], 0, abs_tol=1e-10):
                x.append(j)
                v.append(data_read[j, i])
            else:
                xq.append(j)

        if xq and len(x) > 1:
            vq = np.interp(xq, x, v)
            for j in range(len(xq)):
                data_read[xq[j], i] = vq[j]

    if filtering == 1:
        b, a = scipy.signal.butter(3, 0.05)
        data_read = scipy.signal.lfilter(b, a, data_read, axis=0)

    if remove_start == 1:
        (out, deb) = remove_idle_timeframes(data_read.T, window_size=21, threshold=7, dec=10)
        data_read = data_read.T[:, deb - 1:].T

    length = data_read.shape[0]
    new_x = np.linspace(1, length, number_of_datapoints)
    data_read = np.array([splev(new_x, splrep(range(1, length + 1), line, k=3)) for line in data_read.transpose()])
    return data_read.transpose()


def prepare_vicon_data(filepath, number_of_datapoints = 100, filtering = 1, remove_start = 0):
    with open(filepath, 'r+', encoding='utf-8') as file:
        data_read = np.array([i[:-2].split(' ') for i in file.readlines()]).astype(float)
    number_of_timeframes = len(data_read)
    if number_of_timeframes < number_of_datapoints:
        return None
    data_read = np.reshape(data_read, (len(data_read), 17, 7))
    data_read = data_read[:, :, 0:3]
    data_read = np.reshape(data_read, (len(data_read), 51))

    if filtering == 1:
        b, a = scipy.signal.butter(3, 0.05)
        data_read = scipy.signal.lfilter(b, a, data_read, axis=0)

    if remove_start == 1:
        (out, deb) = remove_idle_timeframes(data_read.T, window_size=21, threshold=7, dec=10)
        data_read = data_read.T[:, deb - 1:].T

    length = data_read.shape[0]
    new_x = np.linspace(1, length, number_of_datapoints)
    data_read = np.array([splev(new_x, splrep(range(1, length + 1), line, k=3)) for line in data_read.transpose()])
    return data_read.transpose()


# preprocessing data for GCN training
def prepare_gcn_data(base_dir, labels_type = 0,
                     exercises = ['CTK', 'ELK', 'RTK'], data_types = ['kinect', 'openpose', 'blazepose', 'vicon'],
                     groups = ['group1A', 'group1A2A', 'group1A2A3', 'group3']):

    for data_type in data_types:
        for group in groups:
            for exercise in exercises:
                data_filename = data_type + '_' + group + '_' + exercise + '_X.csv'
                print(data_filename)
                labels_filename = data_type + '_' + group + '_' + exercise + '_Y.csv'
                # open the file in the write mode
                data_file = open(os.path.join(base_dir, 'stgcn_interpolated', data_filename), 'w', newline='')
                labels_file = open(os.path.join(base_dir, 'stgcn_interpolated', labels_filename), 'w', newline='')

                # create the csv writer
                data_writer = csv.writer(data_file)
                labels_writer = csv.writer(labels_file)

                if labels_type == 0:
                    labels = [0, 0.25, 0.5, 0.75, 1.0]
                else:
                    labels = ['correct', 'incorrect']
                for label in labels:
                    path_to_files = os.path.join(base_dir, group, data_types[0], exercise, str(label))
                    files = os.listdir(path_to_files)
                    first_file = True
                    for filename in files:
                        if data_type == 'kinect':
                            func = prepare_kinect_data
                        elif data_type == 'openpose':
                            func = prepare_openpose_data
                        elif data_type == 'blazepose':
                            func = prepare_blazepose_data
                        else:
                            func = prepare_vicon_data
                        data = func(os.path.join(path_to_files, filename))
                        if data is None:
                            continue
                        if first_file:
                            reference_data = data
                            first_file = False
                        else:
                            data = dynamic_time_warping(reference_data, data)
                        data_writer.writerows(data)
                        labels_writer.writerow([1 if label == 'correct' else 0 if label == 'incorrect' else label])
                data_file.close()
                labels_file.close()


# preprocessing data for GMMs training
def prepare_gmm_data(base_dir, groups = ['group1A', 'group1A2A', 'group3', 'group1A2A3'],
                     exercises = ['CTK', 'ELK', 'RTK'], data_types = ['kinect', 'openpose', 'blazepose', 'vicon']):

    for data_type in data_types:
        if data_type == 'kinect':
            func = prepare_kinect_data
        elif data_type == 'openpose':
            func = prepare_openpose_data
        elif data_type == 'blazepose':
            func = prepare_blazepose_data
        else:
            func = prepare_vicon_data

        for group in groups:
            for exercise in exercises:
                data_filename = data_type + '_' + group + '_' + exercise + '_correct.npy'
                print(data_filename)

                # correct exercises
                data_filename = data_type + '_' + group + '_' + exercise + '_correct.npy'
                all_data = None
                path_to_files = os.path.join(base_dir, group, data_type, exercise, 'correct')
                files = os.listdir(path_to_files)
                for filename in files:
                    data = func(os.path.join(path_to_files, filename))
                    if data is None:
                        continue
                    if all_data is None:
                        reference_data = data
                        time = np.array(range(100)).reshape((1, -1)).T / 100
                        data = np.concatenate((time, data), axis=1)
                        all_data = data
                    else:
                        data = dynamic_time_warping(reference_data, data)
                        time = np.array(range(100)).reshape((1, -1)).T / 100
                        data = np.concatenate((time, data), axis=1)
                        all_data = np.concatenate((all_data, data), axis=0)

                np.save(os.path.join(base_dir, 'gmm_interpolated', data_filename), all_data)

                # incorrect exercises
                data_filename = data_type + '_' + group + '_' + exercise + '_incorrect.npy'
                all_data = None
                path_to_files = os.path.join(base_dir, group, data_type, exercise, 'incorrect')
                files = os.listdir(path_to_files)
                for filename in files:
                    data = func(os.path.join(path_to_files, filename))
                    if data is None:
                        continue
                    time = np.array(range(100)).reshape((1, -1)).T / 100
                    data = np.concatenate((time, data), axis=1)
                    if all_data is None:
                        all_data = data
                    else:
                        all_data = np.concatenate((all_data, data), axis=0)
                np.save(os.path.join(base_dir, 'gmm_interpolated', data_filename), all_data)


if __name__ == '__main__':
    prepare_gcn_data('G:\Datasets\Keraal dataset organized new',
                     labels_type=0,
                     data_types=['blazepose'],
                     groups=['group1A', 'group1A2A'])

    prepare_gmm_data('G:\Datasets\Keraal dataset organized new',
                              data_types=['openpose'],
                              groups=['group1A', 'group2A', 'group1A2A', 'group1A2A3'])
