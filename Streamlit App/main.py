import numpy as np
import streamlit as st
import cv2 
from keras.models import load_model
import tensorflow as tf

#loading the model
model=load_model('plant_disease_model.h5')

#name of classes
CLASS_NAMES=['Corn-Common_rust','Potato-Early_blight','Tomato-Bacterial_spot']

#setting title of app
st.title("Plant Leaf Disease Detection")
st.markdown("Upload an image of the plant leaf")

#uploading the image
plant_image=st.file_uploader("Choose an image...", type="jpg")
submit=st.button('Predict Disease')

#on predict button click
if submit:
    if plant_image is not None:
        #convert the file to an opencv image.
        file_bytes=np.asarray(bytearray(plant_image.read()),dtype=np.uint8)
        opencv_image=cv2.imdecode(file_bytes,1)

        #displaying the image
        st.image(opencv_image,channels="BGR")
        st.write(opencv_image.shape)

        #resizing the image
        opencv_image=cv2.resize(opencv_image,(256,256))

        #convert image to 4D
        opencv_image.shape=(1,256,256,3)

        #make prediction
        Y_pred=model.predict(opencv_image)
        result=CLASS_NAMES[np.argmax(Y_pred)]
        st.title(str("This is " +result.split('-')[0]+ " leaf with " +result.split('-')[1]))