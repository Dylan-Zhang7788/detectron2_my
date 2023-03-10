import numpy as np
import os
import cv2
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

__all__ = ["MY_load_voc_instances", "MY_register_pascal_voc","MY_register"]

CLASS_NAMES = (
    "head", "hand", "foot"
)

def MY_load_voc_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):

    with PathManager.open(os.path.join(dirname, "ImageSets", "Layout", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid[0] + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid[0] + ".jpg")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid[0],
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            if obj.find("name").text == "person":
                # img=cv2.imread(jpeg_file)
                for part in obj.findall("part"):
                    cls=part.find("name").text
                    bbox = part.find("bndbox")
                    bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                    bbox[0] -= 1.0
                    bbox[1] -= 1.0
                    # cv2.circle(img, (int(bbox[0]),int(bbox[1])), 3, (255, 0, 0), 3)
                    # cv2.circle(img, (int(bbox[2]),int(bbox[3])), 3, (255, 0, 0), 3)
                    instances.append(
                        {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
                    )
                r["annotations"] = instances
                dicts.append(r)
                # cv2.imwrite(os.path.join("/home/zhangdi/zhangdi_ws/CenterNet2/AAA", fileid[0] + ".jpg"),img)
    return dicts


def MY_register_pascal_voc(name, dirname, split, year, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: MY_load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split, evaluator_type="MY_pascal_voc"
    )

def MY_register():
    for d in ['train','val','trainval']: MY_register_pascal_voc('voc_2012_Layout_'+d, 'datasets/VOC2012/', d, 2012)