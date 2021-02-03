import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import cv2

cascade_path='/home/alex/Downloads/haarcascade_frontalface_default.xml'
model_paramaters='/home/alex/Downloads/face_mask1.h5'
name="face_mask_detection"
new_model=tf.keras.models.load_model(model_paramaters)
cv2.namedWindow(name)
cap=cv2.VideoCapture(0)

while True:
	image_size=224
	_,frame= cap.read()
	faceCascade=cv2.CascadeClassifier(cascade_path)
	gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces=faceCascade.detectMultiScale(gray,1.3,5)
	for x,y,w, h in faces:
		#burada sadece insan yüzünün oldugu kisimin matrix alinir
		roi_face=frame[y:y+h,y:x+w]
		final_array=cv2.resize(roi_face,(image_size,image_size))
		final_array=np.expand_dims(final_array,axis=0)
		final_array=final_array/255
		predictions=new_model.predict(final_array)
		font=cv2.FONT_HERSHEY_PLAIN
		font_scale=1.5
		print("predictions",predictions)
		#Maskeli durum 
		if predictions<0.5:
			status='Mask {} %'.format(round(100-predictions[0][0]*100,2))
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
			cv2.putText(frame,status,(x,y-45), font,font_scale,color=(0,255,0),thickness=3)
		else:
			status='No Mask {} %'.format(round(predictions[0][0]*100,2))
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
			cv2.putText(frame,status,(x,y-45), font,font_scale,color=(0,0,255),thickness=3)


	cv2.imshow('face_mask_detection',frame)





	if cv2.waitKey(1)==27:
		break
cap.release()
cv2.destroyWindow()
