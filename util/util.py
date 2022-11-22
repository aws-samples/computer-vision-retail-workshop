
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import boto3
import pandas as pd
import numpy as np
import json
import imageio
from PIL import Image
import io


def visualize_detection(img, dets, classes=[], thresh=0.6):
    """
    visualize detections in one image
    Parameters:
    ----------
    img : numpy.array
        image, in bgr format
    dets : numpy.array
        ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
        each row is one object
    classes : tuple or list of str
        class names
    thresh : float
        score threshold
    """
    
    SIZE_FONT = 8
    WIDTH_LINE = 2
    DPI = 150
    BLOCK_SIZE = 5
    HEIGHT = 7
    
    if (type(thresh) is list) or (type(thresh) is tuple):
        total_figs = len(thresh)
    else:
        total_figs = 1
        thresh=[thresh]
    
    
    # set fixed colors per class using the tab10 color map
    cm = plt.get_cmap('jet') 
    max_number_of_classes = 10
    cNorm  = colors.Normalize(vmin=0, vmax=max_number_of_classes-1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    fig = plt.figure(figsize=(total_figs * BLOCK_SIZE, HEIGHT), dpi=DPI)
    
    for i in range(total_figs):
        plt.subplot(1, total_figs, i+1)
        plt.imshow(img)
        plt.grid(False)
        plt.axis(False)
        height = img.shape[0]
        width = img.shape[1]
        num_detections = 0
        for det in dets:
            (klass, score, x0, y0, x1, y1) = det
            if score < thresh[i]:
                continue
            num_detections += 1
            cls_id = int(klass)
            color_val = scalarMap.to_rgba(cls_id)[0:3]  # keep only rgb, discard a
            xmin = int(x0 * width)
            ymin = int(y0 * height)
            xmax = int(x1 * width)
            ymax = int(y1 * height)
            rect = plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor=color_val,
                linewidth=WIDTH_LINE,
            )
            # plot bbox
            plt.gca().add_patch(rect)
            class_name = str(cls_id)
            if classes and len(classes) > cls_id:
                class_name = classes[cls_id]
            # plot class name
            plt.gca().text(
                xmin,
                ymin - 2,
                "{:s} {:.3f}".format(class_name, score),
                bbox=dict(facecolor=color_val, alpha=0.5),
                fontsize=SIZE_FONT,
                color="white",
            )
        plt.title("Detection threshold: " + str(thresh[i]) + "\nNumber of detections: " + str(num_detections), fontsize=SIZE_FONT)
    
    plt.show()
    




def predict(filename, runtime, endpoint_name, class_names, thresh=0.40, visualize=True):
    payload_bytes = ""
    with open(filename, "rb") as image:
        f = image.read()
        payload_bytes = bytearray(f)
    endpoint_response = runtime.invoke_endpoint(
        EndpointName=endpoint_name, 
        ContentType="image/jpeg", 
        Body=payload_bytes
    )
    results = endpoint_response["Body"].read()
    detections = json.loads(results)
    
    if visualize is True: 
        image = np.array(Image.open(io.BytesIO(payload_bytes)))  # image array from bytes
        visualize_detection(image, detections["prediction"], class_names, thresh)
    
    if (type(thresh) is list) or (type(thresh) is tuple): thresh=thresh[0]  # if many, return only the first threshold   
    df_results = pd.DataFrame(data=detections["prediction"], columns=['class', 'confidence', 'x1', 'y1', 'x2', 'y2'])
    df_results = df_results.drop(df_results[df_results.confidence < thresh].index)  # keep only detections above the threshold

    return df_results    
    
    

def get_iou(BBoxW1, BBoxH1, BBoxL1, BBoxT1, BBoxW2, BBoxH2, BBoxL2, BBoxT2):
    # intersection over union in order to match bboxes

    # get right and bottom coordinates
    BBoxR1 = BBoxL1 + BBoxW1
    BBoxB1 = BBoxT1 + BBoxH1
    BBoxR2 = BBoxL2 + BBoxW2
    BBoxB2 = BBoxT2 + BBoxH2
    
    int_L = max(BBoxL1, BBoxL2)
    int_R = min(BBoxR1, BBoxR2)
    int_T = max(BBoxT1, BBoxT2)
    int_B = min(BBoxB1, BBoxB2)
    intersection_area = (max(0, int_R - int_L) * max(0, int_B - int_T))
    
    bbox1_area = (BBoxR1 - BBoxL1) * (BBoxB1 - BBoxT1)
    bbox2_area = (BBoxR2 - BBoxL2) * (BBoxB2 - BBoxT2)
    
    iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)
    
    return iou




def evaluate_testset(runtime, endpoint_name, class_names, testset_folder, test_manifest_file, thr_iou, thr_conf):
    
    # initialize performance df
    data = np.zeros([len(class_names),6], dtype=float)
    df_class_performance = pd.DataFrame(data=data, columns=['TP', 'FP', 'FN', 'PR', 'RE', 'F1'])
    df_classes = pd.DataFrame(data=class_names, columns=['CLASS'])
    df_class_performance = pd.concat([df_classes,df_class_performance], axis=1)

    # open manifest file
    with open(test_manifest_file) as f:
        print(f'Evaluating {test_manifest_file}...')
        lines = f.readlines()

    # go through each JSON line
    for line in lines:
        ls_annotations = []
        line_dict = json.loads(line)

        # get image
        filename = Path(line_dict['source-ref'])
        print(f'Analyzing image {str(filename.name)}...')
        filename_with_path = Path(*filename.parts[2:])
        # s3.download_file(BUCKET_NAME, str(filename_with_path), f'{LOCAL_DATASET_FOLDER}/{str(filename.name)}')  # download image from s3
        # TODO: predict directly from binary in S3 without downloading file!

        # get detections from endpoint
        df_detections = predict(f'{testset_folder}/{str(filename.name)}', runtime, endpoint_name, class_names, thresh=thr_conf, visualize=False)
        im_width = line_dict['retail-object-labeling']['image_size'][0]['width']
        im_height = line_dict['retail-object-labeling']['image_size'][0]['height']
        df_detections.loc[:,'x1'] *= im_width
        df_detections.loc[:,'x2'] *= im_width
        df_detections.loc[:,'y1'] *= im_height
        df_detections.loc[:,'y2'] *= im_height
        df_detections.loc[:,['x1','x2','y1','y2']] = df_detections.loc[:,['x1','x2','y1','y2']].round(decimals=0)

        # get annotations from manifest GT file 
        for i,annotation in enumerate(line_dict['retail-object-labeling']['annotations']):
            ls_ground_truth = []
            ls_ground_truth.append(line_dict['retail-object-labeling']['annotations'][i]['class_id'])
            ls_ground_truth.append(line_dict['retail-object-labeling']['annotations'][i]['top'])
            ls_ground_truth.append(line_dict['retail-object-labeling']['annotations'][i]['left'])
            ls_ground_truth.append(line_dict['retail-object-labeling']['annotations'][i]['height'])
            ls_ground_truth.append(line_dict['retail-object-labeling']['annotations'][i]['width'])
            ls_annotations.append(ls_ground_truth)
        df_annotations = pd.DataFrame(data=ls_annotations, columns=['class_id', 'top', 'left', 'height', 'width'])

        # create IOU array
        mat_iou = np.zeros((len(df_annotations), len(df_detections)), dtype=float)
        for j in range(len(df_detections)):
            for i in range(len(df_annotations)):
                iou = get_iou(
                    BBoxW1=df_detections.loc[j,'x2']-df_detections.loc[j,'x1'], 
                    BBoxH1=df_detections.loc[j,'y2']-df_detections.loc[j,'y1'], 
                    BBoxL1=df_detections.loc[j,'x1'], 
                    BBoxT1=df_detections.loc[j,'y1'], 
                    BBoxW2=df_annotations.loc[i,'width'],
                    BBoxH2=df_annotations.loc[i,'height'], 
                    BBoxL2=df_annotations.loc[i,'left'],
                    BBoxT2=df_annotations.loc[i,'top'] 
                )
                mat_iou[i,j] = iou
        mat_iou[mat_iou < thr_iou] = 0  # binarize IOU array
        mat_iou[mat_iou > 0] = 1

        # analyzing IOU array
        for i in range(len(df_annotations)):
            class_id_annotation = int(df_annotations.loc[i,'class_id'])

            if mat_iou[i,:].sum() == 0:  # if no matches for this annotation
                df_class_performance.loc[class_id_annotation, 'FN'] += 1

            elif mat_iou[i,:].sum() == 1:  # if only one matching for this annotation
                indx_nonzero = np.nonzero(mat_iou[i,:])[0][0]
                class_id_detection = int(df_detections.loc[indx_nonzero, 'class'])

                if class_id_annotation == class_id_detection:
                    df_class_performance.loc[class_id_detection, 'TP'] += 1
                else:
                    df_class_performance.loc[class_id_detection, 'FP'] += 1

            elif mat_iou[i,:].sum() > 1:  # if more than one matching for this annotation
                indx_nonzero = np.squeeze(np.nonzero(mat_iou[i,:])[0])  # many indices of nonzero ious
                conf_detection = df_detections.loc[indx_nonzero, 'confidence']  # many confidences of nonzero ious
                indx_maxconf = indx_nonzero[np.argmax(conf_detection)]  # find the indx of max confidence
                class_id_maxconf = df_detections.loc[indx_maxconf, 'class']  # find the class of max confidence

                indx_lowconf = np.delete(indx_nonzero, np.argmax(conf_detection))  # keep all the indexes without the one of max confidence
                class_id_lowconf = df_detections.loc[indx_lowconf, 'class']  # find the classes of the rest

                if class_id_annotation == class_id_maxconf:
                    df_class_performance.loc[class_id_maxconf, 'TP'] += 1  # the max confidence is TP
                    df_class_performance.loc[class_id_lowconf, 'FP'] += 1  # the rest are FP
                else:
                    df_class_performance.loc[class_id_maxconf, 'FP'] += 1  # all are FP
                    df_class_performance.loc[class_id_lowconf, 'FP'] += 1 

            else:
                print('Problem with negative IOU values!')


        for j in range(len(df_detections)):
            class_id_detection = int(df_detections.loc[j, 'class'])
            if mat_iou[:,j].sum() == 0:  # if no matches for this detection
                df_class_performance.loc[class_id_detection, 'FP'] += 1

        
    # estimate metrics per class
    df_class_performance.loc[:,'PR'] = df_class_performance.loc[:,'TP'] / (df_class_performance.loc[:,'TP'] + df_class_performance.loc[:,'FP'])
    df_class_performance.loc[:,'RE'] = df_class_performance.loc[:,'TP'] / (df_class_performance.loc[:,'TP'] + df_class_performance.loc[:,'FN'])
    df_class_performance.loc[:,'F1'] = (2 * df_class_performance.loc[:,'PR'] * df_class_performance.loc[:,'RE']) / (df_class_performance.loc[:,'PR'] + df_class_performance.loc[:,'RE'])
    
    mean_macro = [ 
        'macro Average',
        '',
        '',
        '',
        df_class_performance.loc[:,'PR'].mean(),
        df_class_performance.loc[:,'RE'].mean(),
        df_class_performance.loc[:,'F1'].mean()
    ]
    
    df_mean = pd.DataFrame(data=[mean_macro], columns=['CLASS', 'TP', 'FP', 'FN', 'PR', 'RE', 'F1'])
    
    df_class_performance = df_class_performance.append(df_mean, ignore_index=True)

    
    return df_class_performance
