import numpy as np
import os 
import tensorflow as tf

import numpy  as np 
import matplotlib.pyplot as plt 
import cv2


cascade_path='/home/alex/Downloads/haarcascade_frontalface_default.xml'
font_scale=1.5
font=cv2.FONT_HERSHEY_PLAIN

rectange_bgr=(255,255,255)
img=np.zeros((500,500))
text='Some text in box'
(text_width, text_height)= cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
text_offset_x=10
text_offset_y=img.shape[0]-25

box_coords=((text_offset_x, text_offset_y),(text_offset_x+text_width+2,text_offset_y-text_height-2))

cv2.rectangle(img, box_coords[0], box_coords[1], rectange_bgr,cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0,0,0), thickness=1)
new_model=tf.keras.models.load_model('/home/alex/Downloads/face_mask1.h5')


cap=cv2.VideoCapture(0)


if not cap.isOpened():
	cap=cv2.VideoCapture(0)
if not cap.isOpened():
	raise IOError("Can not found camera")

while True:
	_, frame= cap.read()
	faceCasCade = cv2.CascadeClassifier(cascade_path)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces= faceCasCade.detectMultiScale(gray, 1.3, 5)
	for x,y,w,h in faces:
		roi_gray=gray[y:y+h, x:x+w]
		roi_color=frame[y:y+h, x:x+w]
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0,0),2)
		facess= faceCasCade.detectMultiScale(roi_gray)
		if len(facess)==0:
			print(" no photos")
		else:
			for (ex, ey,ew, eh) in facess:
			
				face_roi=roi_color[ey:ey+eh,ex:ex+ew]
		final_face=cv2.resize(face_roi,(224,224))
		final_array= np.expand_dims(final_face, axis=0)
		final_array=final_array/255.0
		font=cv2.FONT_HERSHEY_PLAIN
		font_scale=1.5
		predictions=new_model.predict(final_array)
		print("predictions",predictions)
		if (predictions<0.5):
			status='Mask'
			x1,y1,w1, h1=0,0,175,75
			cv2.rectangle(frame,(x1,x1),(x1+w1, y1+h1),(0,0,0),-1)
			cv2.putText(frame, status, (x1+88, y1+38), font, font_scale, color=(0,0,255), thickness=2)
			cv2.putText(frame, status, (100, 150), font, 3,(0,255,0),2, cv2.LINE_4)
			cv2.rectangle(frame,(x,y),(x+w, y+h),(0,0,255))
		else:
			status='No Mask'
			x1,y1,w1, h1=0,0,175,75
			cv2.rectangle(frame,(x1,x1),(x1+w1, y1+h1),(0,0,0),-1)
			cv2.putText(frame, status, (x1+88, y1+38), font, font_scale, color=(0,0,255), thickness=2)
			cv2.putText(frame, status, (100, 150), font, 3,(0,255,0),2, cv2.LINE_4)
			cv2.rectangle(frame,(x,y),(x+w, y+h),(0,0,255))



	cv2.imshow("Face mask detection", frame)
	if cv2.waitKey(2) & 0xFF==ord('q'):
		break
cap.release()
cv2.destroyAllWindows()

