import pydicom as dcm
import mdai
import pandas as pd
import numpy as np
import cv2
from preprocess_functions import vertices_to_boxes
from tqdm import tqdm

#pandas 경고 메세지 끄기
pd.set_option('mode.chained_assignment',  None)

#annotation 전처리

anno_dir = '/Users/gim-yeongmin/DeepLearning/RICORD_data/MIDRC-RICORD-1a_annotations_labelgroup_all_2020-Dec-8 3.json'

def anno_preprocessing(anno_dir):
    anno_dir = anno_dir
    try:
        results = mdai.common_utils.json_to_dataframe(anno_dir)
    except:
        print("mdai ERROR")

    labels_df = results['labels']
    annots_df = results['annotations']
    columns_brief = ['id', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'labelName', 'data', 'annotationMode']
    annots_df = annots_df[columns_brief]
    shapes = annots_df[(annots_df.annotationMode == 'freeform') | (annots_df.annotationMode == 'polygon')]

    try:
        shapes = shapes.assign(x=0,y=0,w=0,h=0,bottom_x=0,bottom_y = 0)
    except:
        print("shapes xywh ERROR")

    shapes = shapes.reset_index(drop=True)
    for i in range(shapes.shape[0]):
        if shapes['data'][i] == None:
            shapes=shapes.drop(i)
    shapes = shapes.reset_index(drop=True)

    for i in tqdm(range(shapes.shape[0])):
        shapes['x'][i],shapes['y'][i],shapes['w'][i],shapes['h'][i],shapes['bottom_x'][i],shapes['bottom_y'][i] = vertices_to_boxes(shapes['data'][i])

    return shapes

anno_df = anno_preprocessing(anno_dir)
anno_df.to_csv("/Users/gim-yeongmin/Desktop/COVID_lung_CT/anno_df.csv")