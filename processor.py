import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64

def pre_process(input):
    image_data = base64.b64decode(input)
    # Open the image and convert to RGB
    original_image = Image.open(BytesIO(image_data))
    # Convert the image to a tensor
    image = original_image.resize((224, 224))
    # Convert the image to a numpy array and normalize it
    image_np = np.array(image)
    image_np = image_np.astype(np.uint8)
    # Convert to list
    image_list = image_np.tolist()
    return image_list
def post_process(input):
    predictions=input
    boxes = predictions['detection_boxes']
    scores = predictions['detection_scores']
    classes = predictions['detection_classes']
    num_detections = int(predictions['num_detections'])
    
    filtered_boxes = []
    filtered_scores = []
    filtered_classes = []
    
    for i in range(num_detections):
        if scores[i] > 0.5:
            filtered_boxes.append(boxes[i])
            filtered_scores.append(scores[i])
            filtered_classes.append(classes[i])
    
    filtered_predictions = {
        'detection_boxes': filtered_boxes,
        'detection_scores': filtered_scores,
        'detection_classes': filtered_classes,
        'num_detections': len(filtered_scores)
    }
    
    return filtered_predictions
