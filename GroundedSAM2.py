from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2
import os

# define an ontology to map class names to our Grounded SAM 2 prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = GroundedSAM2(
    ontology=CaptionOntology(
        {
            'industrial indoor tank':'Boiler', 'Cable Tray fitting':'Cable Tray fitting', 'Electrical Panel':'Electrical Panel', 
            'Fire alarm detector':'Fire alarm detector', 'Not the pipe the pipe connector':'Pipe Fitting', 'Valve':'Valve', 
            'PowerHub or electrical Outlet':'electrical Outlet', 'generator': 'generator', 'lamp':'light', 'pump':'pump'
        }
    )
)
import torch
print(torch.cuda.is_available())

print("Model loaded", base_model.__dir__())
quit()

dir = "/media/wahabu/Extreme SSD/test"
for f in os.listdir(dir):
    file = dir+"/"+f
    results = base_model.predict(file)
    plot(image=cv2.imread(file),
    classes=base_model.ontology.classes(),
    detections=results)

# run inference on a single image
#base_model.label("/media/wahabu/Extreme SSD/test", extension=".jpg")