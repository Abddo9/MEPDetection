from ultralytics import YOLO
model = YOLO("yolo11n.pt")

data_yaml = "/home/abdalwhab/data/work/ETS/research/Construction/YOLO-5/data.yaml"

#pretaiend_results = model.val(data=data_yaml, plots=True)
#print("pretaiend_results", pretaiend_results)

train_results = model.train(data=data_yaml, epochs=500, imgsz=640)
print("Training results", train_results)

eval_metrics = model.val()
print("eval_metrics", eval_metrics)


print("Maps")
print(eval_metrics.box.map)  # map50-95
print(eval_metrics.box.map50)  # map50
print(eval_metrics.box.map75)  # map75
print(eval_metrics.box.maps)
