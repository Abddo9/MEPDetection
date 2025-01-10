from autodistill_gemini import Gemini
from autodistill.utils import plot
from autodistill.detection import CaptionOntology
import cv2
import os

# define an ontology to map class names to our Gemini prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = Gemini(
    ontology=CaptionOntology(
        {
            'industrial indoor tank':'Boiler', 'Cable Tray fitting':'Cable Tray fitting', 'Electrical Panel':'Electrical Panel', 
            'Fire alarm detector':'Fire alarm detector', 'pipes connection':'Pipe Fitting', 'Valve':'Valve', 
            'PowerHub or electrical Outlet':'electrical Outlet', 'generator': 'generator', 'lamp':'light', 'pump':'pump'
        }
    ),
    gcp_region="us-central1",
    gcp_project="project-name",
    #model="gemini-1.5-flash"
)


dir = "/media/wahabu/Extreme SSD/test"
for f in os.listdir(dir):
    file = dir+"/"+f
    results = base_model.predict(file)
    plot(image=cv2.imread(file),
    classes=base_model.ontology.classes(),
    detections=results)

# run inference on an image
# result = base_model.predict("image.jpg")

# print(result)

# # label a folder of images
# base_model.label("./context_images", extension=".jpeg")