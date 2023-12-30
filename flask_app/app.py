from flask import Flask,redirect, render_template, request, url_for
import os
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import torch
import sys, distutils.core
sys.path.insert(0, os.path.abspath('./detectron2'))

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode

# preparing the dataset
def get_sign_dicts(directory):
    classes = ['wind turbine', 'damaged wind turbine']
    dataset_dicts = []
    img_id = 0
    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            # print('opened file')
            img_anns = json.load(f)
            

        record = {}
        temp = img_anns['imagePath'].split('\\')
        filename = os.path.join(directory, temp[-1])
        

        record["file_name"] = filename
        record["image_id"] = img_id
        record["height"] = img_anns["imageHeight"]
        record["width"] = img_anns["imageWidth"]

        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']]
            py = [a[1] for a in anno['points']]
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
        img_id += 1
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("wind_turbine_test1_" + d, lambda d=d: get_sign_dicts('wind_turbine_dataset/' + d))
    MetadataCatalog.get("wind_turbine_test1_" + d).set(thing_classes=['wind turbine','damaged wind turbine'])
wind_turbine_test1_metadata = MetadataCatalog.get("wind_turbine_test1_train")

cfg = get_cfg()
cfg.OUTPUT_DIR = 'output/'
cfg.merge_from_file("config.yml")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model.pth")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.DATASETS.TEST = ('wind_turbine_test1_test',)
cfg.DATASETS.TRAIN = ("wind_turbine_test1_train",)
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)


app = Flask(__name__)
# path = r"C:\Users\Prakhar Jadaun\Documents\Celebal Tech\CLE\Project\Main"

@app.route('/')
def home():    
    return render_template('index.html')

# API to generate text from the uploaded image
@app.route('/process', methods=['POST'])
def process_image():
    
    if request.method=='POST':
        # form_data = request.form
        f = request.files['file1']
        img_path = os.path.join('static',f.filename)
        f.save(img_path)

        img = cv2.imread(img_path)
        print("Real image shape",img.shape)
        outputs = predictor(img)
        v = Visualizer(cv2.cvtColor(img, cv2.COLOR_RGB2BGR),MetadataCatalog.get('wind_turbine_test1_train' ),scale=0.3)
        out = v.draw_instance_predictions(outputs['instances'].to("cpu"))

        pred_img = cv2.cvtColor(out.get_image(),cv2.COLOR_BGR2RGB)
        # plt.imshow(pred_img)
        # plt.show()
        print("Predicted image shape",pred_img.shape)
        # pred_img = cv2.resize(pred_img,(586,371),interpolation=cv2.INTER_AREA)
        cv2.imwrite('static/result.png',pred_img)
        print("Predicted image shape",pred_img.shape)

        return render_template('predictions.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')