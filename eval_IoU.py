#!/usr/bin/python3
import os
import cv2
import json
from glob import glob
import numpy as np
import argparse

def get_iou(det_mask, gt_mask):
    intersection = np.sum(cv2.bitwise_and(det_mask, gt_mask) > 0)
    union = np.sum(cv2.bitwise_or(det_mask, gt_mask) > 0)

    return intersection/union

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluation Script for segmentation of Images')
    #parser.add_argument('-i', '--img_path', type=str, default='img', required=True, help="Path for the image folder")
    parser.add_argument('-d', '--det_path', type=str, default='det', required=True, help="Path for the detected masks folder")
    parser.add_argument('-g', '--gt_path', type=str, default='gt', required=True, help="Path for the ground truth masks folder")
    
    args = parser.parse_args()

    
 
    instrument_folder_name = os.path.basename(os.path.dirname(os.path.dirname(args.det_path)))
    #print("instrument_folder_name-->", instrument_folder_name)

    # mask_folder/instrument_dataset_x/problem_type_masks/framexxx.png
    mask_folder = mask_save_dir / instrument_folder_name / utils.mask_folder[args.problem_type]
    mask_folder.mkdir(exist_ok=True, parents=True)
    mask_filename = mask_folder / os.path.basename(input_filename)

       #img_files = sorted(glob(os.path.join(args.img_path, "*jpg")))
    det_files = sorted(glob(os.path.join(args.det_path, "*jpg")))
    gt_files = sorted(glob(os.path.join(args.gt_path, "*jpg")))
    #print("Number of images: {}".format(len(img_files)))
    print("Number of detections: {}".format(len(det_files)))
    print("Number of ground truths: {}".format(len(gt_files)))

    assert(len(det_files) == len(gt_files))

    iou = []

      
    for fdet, fgt in zip(det_files, gt_files):
        gt_mask = cv2.imread(fgt, cv2.CV_8UC1)

        det_mask = cv2.imread(fdet, cv2.CV_8UC1)

        assert(det_mask.shape == gt_mask.shape)
        iou.append(get_iou(det_mask, gt_mask))
        print("IoU for image {} = {}".format(fimg, iou[-1]))
    print("Average IoU = ", np.mean(iou))
