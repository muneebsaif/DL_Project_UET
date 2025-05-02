from ultralytics import YOLO
import cv2

# Load a COCO-pretrained YOLO11n model
coco_model = YOLO("yolo11n.pt")  
model_path="./runs/train/weights/best.pt"

ppe_model = YOLO(model_path)


path="./video/construction1.mp4"

def crop_images(image, x1, y1, x2, y2):
	# Crop the image using the coordinates
	cropped_image = image[y1:y2, x1:x2]
	return cropped_image

def detect_ppe(frame):
	# Perform inference on the frame
	results=ppe_model(frame,device=0)[0]
	print(results)
	mask = results.boxes.cls == 0
	results=results[mask]
	xyxys=[list(map(int, res)) for res in results.boxes.xyxy]
	print(xyxys)
	if len(xyxys)>0:
		return True
	else:
		return False

def detect_person(frame):
	# Perform inference on the frame
	results=coco_model(frame,device=0)[0]
	mask = results.boxes.cls == 0 
	results=results[mask]
	xyxys=[list(map(int, res)) for res in results.boxes.xyxy]
	print(xyxys)
	for xyxy in xyxys:
		crop_image = crop_images(frame,xyxy[0],xyxy[1],xyxy[2],xyxy[3])
		if detect_ppe(crop_image):
			color=(0, 255, 0)
			print("PPE detected")
		else:
			color=(0, 0, 255)
			print("PPE not detected")
		cv2.rectangle(frame,(xyxy[0],xyxy[1]),(xyxy[2],xyxy[3]),color,2)
	return frame

cap=cv2.VideoCapture(path)
output_path="./video/output.mp4"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))

while True:
	ret,frame=cap.read()
	if frame is None:
		print("No frame")
		continue

	frame=detect_person(frame)
	out.write(cv2.resize(frame,(640, 480)))
	cv2.imshow("frame",frame)
	key=cv2.waitKey(1)
	if key == 27:  # Press 'Esc' to exit
		break
cv2.destroyAllWindows()
cap.release()
out.release()
