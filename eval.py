import os
import cv2
from sklearn.metrics import recall_score, precision_score, f1_score
from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology
from autodistill.utils import plot

# Define directories
imageDir = 'path/to/imageDir'
labelsDir = 'path/to/labelsDir'

# Initialize lists to store true labels and predictions
true_labels = []
pred_labels = []

# Define the base model
base_model = GroundedSAM2(
    ontology=CaptionOntology(
        {
            'Boiler':'Boiler', 'Cable Tray fitting':'Cable Tray fitting', 'Electrical Panel':'Electrical Panel', 
            'Fire alarm detector':'Fire alarm detector', 'Not the pipe the pipe connector':'Pipe Fitting', 'Valve':'Valve', 
            'PowerHub or electrical Outlet':'electrical Outlet', 'generator': 'generator', 'lamp':'light', 'pump':'pump'
        }
    )
)

# Loop through each image file in imageDir
for image_file in os.listdir(imageDir):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):
        # Read the image
        image_path = os.path.join(imageDir, image_file)
        image = cv2.imread(image_path)
        
        # Read the corresponding label file
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labelsDir, label_file)
        
        with open(label_path, 'r') as f:
            labels = f.readlines()
        
        # Process labels (assuming YOLO format: class x_center y_center width height)
        for label in labels:
            class_id, x_center, y_center, width, height = map(float, label.split())
            true_labels.append(class_id)
        
        # Make predictions using the base_model
        results = base_model.predict(image_path)
        
        # Process predictions (assuming the model returns class ids)
        for result in results:
            pred_labels.append(result['class_id'])
        
# Calculate Recall
recall = recall_score(true_labels, pred_labels, average='macro')

# Calculate Precision
precision = precision_score(true_labels, pred_labels, average='macro')

# Calculate F1 Score
f1 = f1_score(true_labels, pred_labels, average='macro')

print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")