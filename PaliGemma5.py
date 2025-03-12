from autodistill_paligemma import PaliGemma
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import supervision as sv
import numpy as np
from supervision.metrics import Precision, Recall, F1Score
import sys
import cv2
import os

from huggingface_hub import login
os.environ["HF_ACCESS_TOKEN"] = "HF_ACCESS_TOKEN"
login(os.environ["HF_ACCESS_TOKEN"])

print("login successfull")

# define an ontology to map class names to our PaliGemma prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = PaliGemma(
    ontology=CaptionOntology(
        {
            'Boiler':'Boiler', 'Cable Tray fitting':'Cable Tray fitting', 'Electrical Panel':'Electrical Panel', 
            'Fire alarm detector':'Fire alarm detector', 'Pipe Fitting':'Pipe Fitting', 'Valve':'Valve', 
            'electrical Outlet':'electrical Outlet', 'generator': 'generator', 'light':'light', 'pump':'pump'
        }
    )
)

print("Fintuning the model PaliGemma on the 0.5 training dataset")
from autodistill_paligemma import paligemma_model 
dir = "/home/wahabu/data/construction/MepDetection/data/2YOLO-5/"
target_model = paligemma_model.PaliGemmaTrainer()
target_model.train(dir)

print("Finished fintuning the model PaliGemma on the .5 training dataset")


print("Evaluating the model PaliGemma on the test dataset")

# Define directories
main_dir = '/home/wahabu/data/construction/MepDetection/YOLO-5/'
test_yaml = main_dir + '/test.yaml'
test_dir = main_dir + '/test'
image_dir = test_dir + '/images'
labels_dir = test_dir + '/labels'
ds = sv.DetectionDataset.from_yolo(
    images_directory_path=image_dir,
    annotations_directory_path=labels_dir,
    data_yaml_path=test_yaml,
    force_masks=True
)
ds.classes = ['Boiler', 'Cable Tray fitting', 'Electrical Panel', 'Fire alarm detector', 'Pipe Fitting', 'Valve', 'electrical Outlet', 'generator', 'light', 'pump']

precision = Precision()
recall = Recall()
#f1 = F1Score()
i=0
for img, _, targets in ds:
    preds = base_model.predict(img)       
    precision.update(preds, targets)
    recall.update(preds, targets)
    #f1.update(preds, targets)
    if i % 10 == 0:
        print("Processed", i, "images")
    i+=1

model_name = "PaliGemma 0.5"
print(model_name, "Results:")
print("precision", precision.compute())
print("recall", recall.compute())
#print("f1", f1.compute())
print("Finished evaluation of", model_name)