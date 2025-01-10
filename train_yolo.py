import sys
from ultralytics import YOLO

model_name = sys.argv[1] if len(sys.argv)>1 else "yolo11n.pt"
num_epochs = int(sys.argv[2]) if len(sys.argv)>2 else 30
data_yaml = sys.argv[3] if len(sys.argv)>3 else "/home/abdalwhab/data/work/ETS/research/Construction/YOLO-5/data.yaml"

print("Training", model_name, "for", num_epochs, "epochs, using the data", data_yaml)

model = YOLO(model_name)

train_results = model.train(data=data_yaml, epochs=num_epochs, imgsz=640)

print("Training results", train_results)

print("Evaluating using the test data")
eval_metrics = model.val(data="/home/abdalwhab/data/work/ETS/research/Construction/YOLO-5/test.yaml", plots=True)

print("eval_metrics", eval_metrics) 


print("Maps")
print("eval_metrics.box.map", eval_metrics.box.map)  # map50-95
print("eval_metrics.box.map50", eval_metrics.box.map50)  # map50
print("eval_metrics.box.map75", eval_metrics.box.map75)  # map75
print("eval_metrics.box.maps", eval_metrics.box.maps)