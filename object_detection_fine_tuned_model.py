import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# --- Model Setup ---
num_classes = 2  # background + pedestrian
backbone = resnet_fpn_backbone('resnet50', pretrained=False)
model = MaskRCNN(backbone, num_classes=num_classes)

model.load_state_dict(torch.load("object_detection_fine_tuning_model.pth", map_location="cpu"))
model.eval()
device = torch.device("cpu")
model.to(device)

# --- Camera Setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

print("Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (480, 360))  # Optional downscale
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(rgb).to(device)

    with torch.no_grad():
        outputs = model([image_tensor])

    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']
    masks = outputs[0]['masks']

    for i in range(len(scores)):
        score = scores[i].item()
        if score < 0.6:
            continue

        box = boxes[i].detach().cpu().numpy().astype(np.int32)
        mask = masks[i][0] > 0.5
        mask = mask.byte().cpu().numpy() * 255

        # Draw rectangle
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Overlay mask faster
        frame[mask > 0] = (0.5 * frame[mask > 0] + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

        # Put label
        cv2.putText(frame, f"Person {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.imshow("CPU MaskRCNN", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
