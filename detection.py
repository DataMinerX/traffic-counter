import cv2
from ultralytics import YOLO
import os
import torch
import matplotlib.pyplot as plt
import json
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils
import torch
from collections import defaultdict

def main():
  model=YOLO('models/yolov8l.pt')
  cap = cv2.VideoCapture('data/data.mp4')
  _,frame=cap.read()
  mask_resized = cv2.imread('masks/masks.png')
  c=1
  count=0
  track_history=track_history = defaultdict(lambda: [])
  counted_vehicles = set()
  while cap.isOpened:
      ret,frame=cap.read()
      if c%3==0:
          
        
        if not ret:
            break
        width,height=640,640
        frame=cv2.resize(frame,(width,height))
        result_frame = cv2.bitwise_and(frame, mask_resized) # USE yolo on this results
        
        results=results = model.track(frame, persist=True,tracker="bytetrack.yaml")
        bboxes = results[0].boxes.xyxy.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        cv2.line(img=frame,pt1=(50,320),pt2=(350,320),color=(255,0,255),thickness=1)
        cv2.line(img=frame,pt1=(50,350),pt2=(350,350),color=(255,0,255),thickness=1)
        if len(bboxes)>0:
            
            for bbox,track_id in zip(bboxes,track_ids):
                x1,y1,x2,y2=bbox[0],bbox[1],bbox[2],bbox[3]
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                track = track_history[track_id]
                track.append((x1,y1,x2,y2))
                x,y=int((x1+x2)/2),int((y1+y2)/2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),4)
                cv2.circle(frame,(x,y),4,(0,255,0),-1)
                if 50 < x < 350 and 320 < y < 350 and track_id not in counted_vehicles:
                  counted_vehicles.add(track_id)
                  cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),4)
                  cv2.circle(frame,(x,y),4,(255,0,255),-1)
        cv2.putText(img=frame,text=f'Car Count ={len(counted_vehicles)}',org = (50, 50),color = (255, 0, 0),thickness = 2,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1)
        cv2.imshow('result',frame)
      if cv2.waitKey(1) & 0xFF==ord('q'):
          break
      c=c+1
      
  cap.release()
  cv2.destroyAllWindows()

if __name__=='__main__':
    main()
