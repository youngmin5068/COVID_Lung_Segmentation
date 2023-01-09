import pydicom as dcm
import mdai
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from preprocess_functions import load_mask_instance, save_labels
from tqdm import tqdm
from ast import literal_eval


anno_dir = "/Users/gim-yeongmin/Desktop/COVID_lung_CT/anno_df.csv"
anno_df = pd.read_csv(anno_dir,index_col=0)


mask_dir = "/Users/gim-yeongmin/Desktop/COVID_lung_CT/manifest-1608266677008/label"

anno_df['data'] = anno_df.apply(lambda x: literal_eval(x.data) , axis = 1)

metadata = pd.read_csv('/Users/gim-yeongmin/Desktop/COVID_lung_CT/manifest-1608266677008/metadata_edit.csv')

path_values = metadata['File Location'].values
userPath = '/Users/gim-yeongmin/Desktop/COVID_lung_CT/manifest-1608266677008/'

#카테고리별 라벨 만들기
save_labels(path_values=path_values,userPath=userPath,anno_df=anno_df,mask_dir=mask_dir)

 