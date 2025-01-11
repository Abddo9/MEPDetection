import supervision as sv
from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill_detic import DETIC
from autodistill_grounding_dino import GroundingDINO

from autodistill.detection import CaptionOntology
import numpy as np
from supervision.metrics import Precision, Recall, F1Score
import sys

# Define directories
main_dir = '/media/wahabu/Extreme SSD/YOLO-5'
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

ontology=CaptionOntology(
        {
            'Boiler':'Boiler', 'Cable Tray fitting':'Cable Tray fitting', 'Electrical Panel':'Electrical Panel', 
            'Fire alarm detector':'Fire alarm detector', 'Pipe Fitting':'Pipe Fitting', 'Valve':'Valve', 
            'electrical Outlet':'electrical Outlet', 'generator': 'generator', 'light':'light', 'pump':'pump'
        }
    )

# Define the base model
#model_name = sys.argv[1] if len(sys.argv)>1 else 'GroundedSAM2'
all_models = ['GroundingDINO', 'DETIC'] # ['GroundedSAM2', 'GroundingDINO', 'DETIC']

for model_name in all_models:
    if model_name == 'GroundedSAM2':
        base_model = GroundedSAM2(ontology=ontology)
    elif model_name == 'GroundingDINO':
        base_model = GroundingDINO(ontology=ontology)
    elif model_name == 'DETIC':
        base_model = DETIC(ontology=ontology)

    print("Evaluating ...", model_name)


    precision = Precision()
    recall = Recall()
    f1 = F1Score()
    for img, _, targets in ds:
        preds = base_model.predict(img)
        
        precision.update(preds, targets)
        recall.update(preds, targets)
        f1.update(preds, targets)

    print(model_name, "Results:")
    print("precision", precision.compute())
    print("recall", recall.compute())
    print("f1", f1.compute())
    print("Finished evaluation of", model_name)
