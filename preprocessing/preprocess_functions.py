import numpy as np
import cv2
from tqdm import tqdm
import pydicom as dcm
import os
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import SimpleITK as sitk

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
        
    mask=np.zeros((512, 512), dtype=np.uint8)
    vertic_list = []
    for index in rows.index:
        vertices= np.array(rows['data'][index]['vertices'])
        vertices = vertices.reshape((-1, 2))
        mask_instance = mask[:,:].copy()
        cv2.fillPoly(mask_instance, np.int32([vertices]), (255, 255, 255))
        mask[:,:] = mask_instance
        
    return mask


def save_labels(path_values, userPath,anno_df,mask_dir):
    new_dir = "/Users/gim-yeongmin/Desktop/COVID_lung_CT/manifest-1608266677008/"+"label/"
    new_train_dir = "/Users/gim-yeongmin/Desktop/COVID_lung_CT/manifest-1608266677008/"+"train/"

    first = sitk.ReadImage("/Users/gim-yeongmin/Desktop/COVID_lung_CT/manifest-1608266677008/MIDRC-RICORD-1A/MIDRC-RICORD-1A-419639-000082/08-02-2002-NA-CT CHEST WITHOUT CONTRAST-04614/2.000000-ROUTINE CHEST NON-CON-97100/1-080.dcm")
    ITK_spacing = first.GetSpacing()
    ITK_origin = first.GetOrigin()

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

            #input image 
            imgITK = sitk.ReadImage(image_list[i])
            imgITK.SetOrigin(ITK_origin)
            imgITK.SetSpacing(ITK_spacing)
            img_arr = sitk.GetArrayFromImage(imgITK).squeeze()
            input_img = Image.fromarray(img_arr)
            plt.imsave(os.path.join(new_train_dir,path[-5:]+"_"+str(i)+".png"),input_img)

            #label image
            img = dcm.dcmread(image_list[i])
            image_mask = img
            mask = load_mask_instance(image_mask,anno_df) # numpy

            mask_2 = sitk.GetImageFromArray(mask)
            mask_2.SetOrigin(ITK_origin)
            mask_2.SetSpacing(ITK_spacing)
            mask_2_arr = sitk.GetArrayFromImage(mask_2).squeeze()
            label_img = Image.fromarray(mask_2_arr)
            label_img.save(os.path.join(new_dir,path[-5:]+"_"+str(i)+"_"+anno_df['labelName'][i]+".png"),'PNG')
