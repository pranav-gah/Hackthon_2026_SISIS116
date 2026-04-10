import cv2
import numpy as np
from ultralytics import YOLO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
import smtplib
from email.message import EmailMessage


MODEL_PATH = r"SORRY" #<--------- This is the path of the model that we have trained for 2 months so we can't give direct model access 
VIDEO_PATH = r"C:\Users\pranav\OneDrive\Desktop\hackathon_clips\Today_I.mp4"
c=False
PERSON_CLASS = 2
GARBAGE_CLASS = 0

ASSOCIATION_DISTANCE = 60
THROW_DISTANCE = 70


model = YOLO(MODEL_PATH)
print(model.names)
for k, v in model.names.items():
    print(k, "->", v)




def center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

cap = cv2.VideoCapture(VIDEO_PATH)

active_associations = {}
frame_no = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break



    frame_no += 1
    debug_frame = frame.copy()


    results = model.track(
        frame,
        conf=0.1,
        iou=0.3,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )

    if results[0].boxes is None or results[0].boxes.id is None:
        cv2.imshow("THROW DEBUG", debug_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    ids = results[0].boxes.id.cpu().numpy()

    persons = {}
    garbages = {}


    for b, c, tid in zip(boxes, classes, ids):
        x1, y1, x2, y2 = map(int, b)

        if int(c) == PERSON_CLASS:
            persons[tid] = center(b)
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                debug_frame,
                f"Person {tid}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        elif int(c) == GARBAGE_CLASS:
            garbages[tid] = center(b)
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                debug_frame,
                f"Garbage {tid}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    throw_detected = False
    for pid, pc in persons.items():
        for gid, gc in garbages.items():

            dist = int(np.linalg.norm(np.array(pc) - np.array(gc)))
            print(dist)

            # Draw line
            cv2.line(debug_frame, pc, gc, (255, 255, 0), 2)

            mx = int((pc[0] + gc[0]) / 2)
            my = int((pc[1] + gc[1]) / 2)

            # Distance text
            cv2.putText(
                debug_frame,
                f"D={dist}",
                (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            # Association
            if dist <= ASSOCIATION_DISTANCE:
                active_associations[pid] = gid
                cv2.putText(
                    debug_frame,
                    "ASSOCIATED",
                    (mx, my + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

            # Throw condition
            if pid in active_associations and active_associations[pid] == gid:
                if dist > THROW_DISTANCE:
                    throw_detected = True

    cv2.putText(
        debug_frame,
        f"ASSOC={ASSOCIATION_DISTANCE} | THROW={THROW_DISTANCE}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    if throw_detected:
        c=True
        cv2.putText(
            debug_frame,
            "THROW DETECTED",
            (50, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 255),
            3
        )
        
    cv2.imshow("THROW DEBUG", debug_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(active_associations)
cap.release()
cv2.destroyAllWindows()