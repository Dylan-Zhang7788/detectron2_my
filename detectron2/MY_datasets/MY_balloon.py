import numpy as np
import os
import cv2
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager


def get_balloon_dicts(img_dir):
    json_file=os.path.join(img_dir,'via_region_data.json')
    with open(json_file) as f:
        imgs_anns=json.load(f)
    dataset_dicts=[]
    for idx,v in enumerate(imgs_anns.values()):
        record={}  #标准字典档

        filename=os.path.join(img_dir,v['filename'])
        height,width=cv2.imread(filename).shape[:2]  #获取尺寸

        record['file_name']=filename
        
        record['image_id']=idx
        record['height']=height
        record['width']=width

        annos=v['regions']  #范围

        objs=[]
        for _,anno in annos.items():
            assert not anno['region_attributes']
            anno=anno['shape_attributes']
            px=anno['all_points_x']
            py=anno['all_points_y']
            poly=[(x+0.5,y+0.5) for x,y in zip(px,py)] #标记框框
            poly=[p for x in poly for p in x]
            obj={
                'bbox':[np.min(px),np.min(py),np.max(px),np.max(py)], #左上角坐标和右下角坐标
                'bbox_mode':BoxMode.XYXY_ABS,
                'segmentation':[poly],
                'category_id':0, #类别id
                'iscrowd':0    #只有一个类别
            }
            objs.append(obj)
        record['annotations']=objs
        dataset_dicts.append(record)
    return dataset_dicts

def MY_register_balloon():
    for d in ['train','val']:  #注册数据集
        DatasetCatalog.register('balloon_'+d,lambda d=d: get_balloon_dicts('datasets/balloon/'+d))
        MetadataCatalog.get('balloon_'+d).set(thing_classes=['balloon'])

def MY_register_balloon_one():
    for d in ['train','val']:  #注册数据集
        DatasetCatalog.register('balloon_one_'+d,lambda d=d: get_balloon_dicts('datasets/balloon_one/'+d))
        MetadataCatalog.get('balloon_one_'+d).set(thing_classes=['balloon'])
