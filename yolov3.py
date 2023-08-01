import cv2
import numpy as np

# Load YOLO model and coco names (coco names are the names of the classifications)
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
coco_names = []
with open("coco.names", "r") as f:
    coco_names = [line.strip() for line in f]

# Load YOLO network classes and get output layer names
layer_names = yolo_net.getLayerNames()
output_layers = yolo_net.getUnconnectedOutLayersNames()

# Function that applies filter to a specific frame that's person
def apply_color_filter(image, color_filter="gray"):
    """
    Apply a color filter to the input image.

    Parameters:
        image (numpy.ndarray): The input image (BGR format).
        color_filter (str): The color filter to apply. Supported options: "gray", "sepia", "negative".
                            Default is "gray".

    Returns:
        numpy.ndarray: The filtered image.
    """
    if color_filter == "gray":
        filtered_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif color_filter == "sepia":
        # Sepia filter
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        filtered_image = cv2.transform(image, kernel)
    elif color_filter == "negative":
        # Negative filter
        filtered_image = cv2.bitwise_not(image)
    else:
        raise ValueError("Invalid color filter. Supported options: 'gray', 'sepia', 'negative'.")

    return filtered_image

# Function to perform object detection
def detect_objects(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Threshold for object detection
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    detected_objects = []
    for i in range(len(boxes)):
        if i in indexes:
            detected_objects.append((coco_names[class_ids[i]], confidences[i], boxes[i]))

    return detected_objects

# Main loop for capturing video and detecting objects
def main():
    cap = cv2.VideoCapture(1)  # Use 0 for the default camera, or the camera index you want

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_objects = detect_objects(frame)

        for obj_name, confidence, (x, y, w, h) in detected_objects:
            color = (0, 255, 0)  # Green rectangle for each detected object
            person_color = (255, 0, 0) # Different color for pereson detection
            text = f"{obj_name}: {confidence:.2f}"
            if 'person'==obj_name:
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, person_color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), person_color, 2)
            else:
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
