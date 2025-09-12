from ultralytics import YOLO

#model = YOLO("runs/detect/train/weights/last.pt")

#model.train(
#    data="dataset/data.yaml",
#    epochs=200,                       
#    resume=True,
#    device=0
#)


model = YOLO("runs/detect/train/weights/best.pt")

# Run inference
results = model("dataset/test/images/IMG_4893_JPG.rf.b5a8f3d0ff54b53aa057be06b1d8e296.jpg")

# results is a list, so take the first element
r = results[0]

# Show image with detections (opens a window)
r.show()

# Save annotated image to disk
r.save("runs/detect/test/")