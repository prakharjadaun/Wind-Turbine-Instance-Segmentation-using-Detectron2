import streamlit as st
import numpy as np
import os
from utils import visualize
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Define the visualize_with_annotations function here...

# Load your custom-trained model
cfg = get_cfg()
cfg.OUTPUT_DIR = 'D:\streamlit\output'
cfg.merge_from_file("config.yml")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)

# Streamlit app
st.title("Object Detection with Your Custom Model")
file = st.file_uploader('', type=['png', 'jpg', 'jpeg'])

if file:
    image = Image.open(file).convert('RGB')

    image_array = np.asarray(image)

    # Detect objects
    outputs = predictor(image_array)

    threshold = 0.5

    # Display predictions
    bboxes_ = []
    labels_=[]

    for j, bbox in enumerate(outputs["instances"].pred_boxes):
        bbox = bbox.tolist()

        score = outputs["instances"].scores[j]

        if score > threshold:
            x1, y1, x2, y2 = [int(i) for i in bbox]
            bboxes_.append([x1, y1, x2, y2])
            labels_ = "Wind Turbine" if pred == 0 else "Damaged Wind Turbine"

    
    

    # Visualize the detections
    visualize(image, bboxes_,labels_)
