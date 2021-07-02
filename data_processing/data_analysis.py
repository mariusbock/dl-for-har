import numpy as np


def max_min_values(data):
    for i, column in enumerate(data.T):
        print('Column {}:'.format(i), 'min: {} max: {}'.format(min(column), max(column)))


def analyze_window_lengths(labels, subject_idx, sampling_rate, class_names):
    curr_label = -1
    curr_window = 0
    curr_subject = -1
    windows = []
    for i, (label, subject_id) in enumerate(zip(labels, subject_idx)):
        if label != curr_label and i > 0:
            windows.append([int(curr_subject), curr_label, curr_window / sampling_rate])
            curr_label = label
            curr_subject = subject_id
            curr_window = 1
        elif label == curr_label:
            curr_window += 1
        else:
            curr_label = label
            curr_subject = subject_id
            curr_window += 1
    windows = np.array(windows)
    # per subject and label
    unique_subjects = np.unique(windows[:, 0])
    unique_labels = np.unique(windows[:, 1])
    print('\n#### PER SUBJECT-LABEL AVERAGES #####')
    for subject in unique_subjects:
        subject_windows = windows[windows[:, 0] == subject]
        print('\n#### SUBJECT {} #####'.format(int(subject)))
        for label in unique_labels:
            subject_label_windows = subject_windows[subject_windows[:, 1] == label]
            if subject_label_windows.size == 0:
                print('NO LABELS FOUND FOR: {}'.format(class_names[int(label)]))
            else:
                print('Label {}: '.format(class_names[int(label)]),
                      'avg. window length {:.1f} seconds, '.format(np.mean(subject_label_windows[:, 2])),
                      'min. window length {:.1f} seconds, '.format(np.min(subject_label_windows[:, 2])),
                      'max. window length {:.1f} seconds'.format(np.max(subject_label_windows[:, 2]))
                      )
    print('\n#### PER LABEL AVERAGES #####')
    for label in unique_labels:
        label_windows = windows[windows[:, 1] == label]
        if label_windows.size == 0:
            print('LABEL NON-EXISTENT!')
        else:
            print('Label {}: '.format(class_names[int(label)]),
                  'avg. window length {:.1f} seconds, '.format(np.mean(label_windows[:, 2])),
                  'min. window length {:.1f} seconds, '.format(np.min(label_windows[:, 2])),
                  'max. window length {:.1f} seconds'.format(np.max(label_windows[:, 2]))
                  )
