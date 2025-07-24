import cv2
import numpy as np
import time

# Load class labels
CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

# Load the pretrained model
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel"
)

# Start webcam
cap = cv2.VideoCapture(0)

print('Starting object detection....')
time.sleep(2)

print('Initializing.....')
time.sleep(5)

print('Now detecting objects.....')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]
    
    # Prepare the frame as input to the network
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )

    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.8:  # Filter weak detections
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, f"{label}: {confidence:.2f}", (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show output
    cv2.imshow("Real-Time Object Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break
    
quit = input('Enter exit to quit: ')

if quit == 'exit':
    print('Terminating detection...')
    time.sleep(2)
    print('Detection terminated, exiting program...')

cap.release()
cv2.destroyAllWindows()