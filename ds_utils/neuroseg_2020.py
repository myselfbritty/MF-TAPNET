import os
import cv2
import numpy as np
import tqdm
from pathlib import Path
import argparse


"""
utils for EETS 2020 neurosurgery instrument segmentation dataset
"""

num_frames_video = 225 # number of frames per video
original_height, original_width = 1080, 1920 # original img h, w
cropped_height, cropped_width = 1080, 1920 # cropped img h, w
h_start, w_start = 0, 0 # crop origin


num_videos = {
    'train': 20,
    'test': 10,
}

# self defined train_val split
folds = {0: [1, 3, 5, 7, 9],
         1: [11, 12, 13, 14, 15],
         2: [2, 4, 6, 8, 10],
         3: [16, 17, 18, 19, 20]}

# number of classes for each type of problem
problem_class = {
    'binary': 2,
    'instruments': 10
}

# factor
problem_factor = {
    'binary' : 255,
    'instruments' : 40
}

# folder names
mask_folder = {
    'binary': 'binary_masks',
    'instruments': 'instruments_masks'
}

# for linear interpolation
mask_folder_linear = {
    'binary': 'binary_masks_linear',
    'instruments': 'instruments_masks_linear'
}


"""
file utilities:
get images, masks, optical flows, predictions, attention maps 
from specific instrument dataset
"""


def get_data(data_dir, data_type, mode='train', folder_id=-1):
    """
    @params:
    data_type: image, optflows or problem type

    return:
    a list of data of specific type
    """

    global num_videos, mask_folder

    num_folders = num_videos[mode]
    
    # if it belongs to problem_type, return masks
    if data_type in mask_folder.keys():
        folder_name = mask_folder[data_type]
    else:
        folder_name = data_type

    filenames = []

    if folder_id > 0:
        # get instrument_dataset_<folder_id>
        filenames += (Path(data_dir) / ('instrument_dataset_' + str(folder_id)) \
            / folder_name).glob('*')
    elif folder_id == -1:
        # get all data
        for folder_id in range(1, num_folders + 1):
            filenames += (Path(data_dir) / ('instrument_dataset_' + str(folder_id)) \
            / folder_name).glob('*')

    if len(filenames) <= 0:
        raise ValueError("Empty folder, data_type: %s, mode: %s, folder_id: %d"
            % (data_type, mode, folder_id))
    
    # sort by name
    return list(sorted(filenames))


# TODO
def get_preds(preds_dir, problem_type, folder_id):
    '''
    preds_dir: Pathlib Path object
    '''
    filenames = list(sorted((preds_dir / ('instrument_dataset_' + str(folder_id)) / problem_type).glob('*')))
    return filenames

def get_all_preds(preds_dir, problem_type):
    '''
    preds_dir: Pathlib Path object
    '''
    filenames = []
    for folder_id in range(1, 21):
        filenames += get_preds(preds_dir, problem_type, folder_id)
    return filenames


def trainval_split(data_dir, fold):
    """
    get train/valid filenames
    """
    global folds

    train_file_names = []
    val_file_names = []

    for idx in range(1, 21):
        # sort according to filename
        filenames = (Path(data_dir) / ('instrument_dataset_' + str(idx)) / 'images').glob('*')
        filenames = list(sorted(filenames))
        # set folds[fold] as validation set
        if idx in folds[fold]:
            val_file_names += filenames
        else:
            train_file_names += filenames

    return train_file_names, val_file_names



"""
[1] Merge masks with different instruments into one binary mask
[2] Crop black borders from images and masks
"""

def preprocess_data(args):
    """
    preprocess data following the instructions
    * crop image
    * for each problem type, squeeze the gt to one two-dim grayscale mask scaled to [0, 255]

    @param: argparse args
    """

    # data dir
    data_dir = Path(args.data_dir)
    assert data_dir.exists() == True

    # cropped data dir
    target_data_dir = Path(args.target_data_dir)
    target_data_dir.mkdir(exist_ok=True, parents=True)

    global num_videos, problem_factor, mask_folder
    # crop (cropped_height, cropped_width) from (h_start, w_start)
    global cropped_height, cropped_width, h_start, w_start

    num_folders = num_videos[args.mode] # number of folders

    for idx in range(1, num_folders + 1):
        # original dataset dir
        instrument_folder = data_dir / ('instrument_dataset_' + str(idx))

        # video frames dir (only read left frames)
        frames_dir = instrument_folder / 'images'

        # processed dataset dir
        processed_instrument_folder = target_data_dir / ('instrument_dataset_' + str(idx))

        # mkdir for each problem_type
        image_folder = processed_instrument_folder / 'images'
        image_folder.mkdir(exist_ok=True, parents=True)

        if args.mode == 'train':
            # original mask folder
            ori_mask_folders = list((instrument_folder / 'ground_truth').glob('*'))

            # new mask folder
            binary_mask_folder = processed_instrument_folder / mask_folder['binary']
            binary_mask_folder.mkdir(exist_ok=True, parents=True)

            instrument_mask_folder = processed_instrument_folder / mask_folder['instruments']
            instrument_mask_folder.mkdir(exist_ok=True, parents=True)

        for file_name in tqdm.tqdm(list(frames_dir.glob('*')),
            desc='preprocess dataset %d' % idx, dynamic_ncols=True):
            #print("Filename-->", file_name)
            img = cv2.imread(str(file_name))
            old_h, old_w, _ = img.shape

            img = img[h_start: h_start + cropped_height, w_start: w_start + cropped_width]
            # save cropped frame
            cv2.imwrite(str(image_folder / (file_name.name)), img)

            if args.mode == 'test':
                continue # test data has no masks

            # create empty masks
            mask_binary = np.zeros((old_h, old_w))
            
            mask_instruments = np.zeros((old_h, old_w))

            for ori_mask_folder in ori_mask_folders:
                # read in grayscale
                mask = cv2.imread(str(ori_mask_folder / file_name.name), 0)

                #print("Mask filename--> ", str(ori_mask_folder / file_name.name))
                # mark each type of instruments
                # background will be set to 0 in default
                try: 
                    if 'suction' in str(ori_mask_folder):
                        mask_instruments[mask > 0] = 1
                    elif 'irrigation' in str(ori_mask_folder):
                        mask_instruments[mask > 0] = 2
                    elif 'spachula' in str(ori_mask_folder):
                        mask_instruments[mask > 0] = 3
                    elif 'scissors' in str(ori_mask_folder):
                        mask_instruments[mask > 0] = 4
                    elif 'knife' in str(ori_mask_folder):
                        mask_instruments[mask > 0] = 5
                    elif 'navigation' in str(ori_mask_folder):
                        mask_instruments[mask > 0] = 6
                    elif 'biopsy1' in str(ori_mask_folder):
                        mask_instruments[mask > 0] = 7
                    elif 'curette' in str(ori_mask_folder):
                        mask_instruments[mask > 0] = 8
                    elif 'drill' in str(ori_mask_folder):
                        mask_instruments[mask > 0] = 9
                    elif 'tumor_biopsy' in str(ori_mask_folder):
                        mask_instruments[mask > 0] = 10
                    if 'Other' not in str(ori_mask_folder):
                    # if exists, will be add in
                        mask_binary += mask
                except:
                    print("No tool")

                    # # process dir exclude 'Other_labels'
                

                #     mask_parts[mask == 10] = 1  # Shaft
                #     mask_parts[mask == 20] = 2  # Wrist
                #     mask_parts[mask == 30] = 3  # Claspers

            # crop and save masks
            mask_binary = (mask_binary[h_start: h_start + cropped_height, 
                w_start: w_start + cropped_width] > 0).astype(np.uint8) * problem_factor["binary"]
            mask_instruments = (mask_instruments[h_start: h_start + cropped_height,
                w_start: w_start + cropped_width]).astype(np.uint8) * problem_factor["instruments"]

            cv2.imwrite(str(binary_mask_folder / file_name.name), mask_binary)
            cv2.imwrite(str(instrument_mask_folder / file_name.name), mask_instruments)
