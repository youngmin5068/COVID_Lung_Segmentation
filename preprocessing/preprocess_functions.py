import numpy as np
import cv2
from tqdm import tqdm
import pydicom as dcm
import os
from PIL import Image
import matplotlib.pyplot as plt
import shutil

def vertices_to_boxes(data):
    vertices = data['vertices']

    px=[v[0] for v in vertices]
    py=[v[1] for v in vertices]
    x = int(np.min(px))
    y = int(np.min(py))
    x2 = int(np.max(px))
    y2 = int(np.max(py))
    w = x2 - x
    h = y2 - y
    
    return (x, y, w, h, x2, y2)


def load_mask_instance(raw_file,shapes):
    
    rows = shapes[shapes.SOPInstanceUID == raw_file.SOPInstanceUID].reset_index(drop=True)

    if len(rows) < 1 :
        return 
        
    mask=np.zeros((512, 512), dtype=np.uint8)
    vertic_list = []
    for index in rows.index:
        vertices= np.array(rows['data'][index]['vertices'])
        vertices = vertices.reshape((-1, 2))
        mask_instance = mask[:,:].copy()
        cv2.fillPoly(mask_instance, np.int32([vertices]), (255, 255, 255))
        mask[:,:] = mask_instance
        
    return mask.astype(np.bool)


def save_labels(path_values, userPath,anno_df,mask_dir):
    new_dir = "/Users/gim-yeongmin/Desktop/COVID_lung_CT/manifest-1608266677008/"+"label/"
    new_train_dir = "/Users/gim-yeongmin/Desktop/COVID_lung_CT/manifest-1608266677008/"+"train/"
    try:
        os.mkdir(new_dir)
        os.mkdir(new_train_dir)
    except Exception as e:
        print(e)

    for path in tqdm(path_values):
        final_path = os.path.join(userPath, path)
        path_list = os.listdir(final_path)
        image_list = sorted([os.path.join(final_path,f) for f in path_list])

        
        for i in range(1, len(image_list)):
            img = dcm.dcmread(image_list[i])
            image_mask = img
            mask = load_mask_instance(image_mask,anno_df)
            if mask is not None:
                shutil.move(image_list[i], new_train_dir+path[-5:]+"_"+str(i)+".dcm")
                mask_2 = Image.fromarray(mask)
                mask_2.save(os.path.join(new_dir,path[-5:]+"_"+str(i)+"_"+anno_df['labelName'][i]+".png"),'PNG')

