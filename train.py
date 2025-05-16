from ultralytics import YOLO

model = YOLO("yolo11l.pt")

model.train(data="data.yaml", imgsz = 640 , batch = 8 , epochs = 7 , workers = 0, device = "cpu" )


