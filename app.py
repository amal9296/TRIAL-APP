import streamlit as st 
import pickle 
from PIL import Image 
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import argmax
from tqdm import tqdm


st.write('amal')   
input_data = st.file_uploader("Upload your file here...")
if input_data is not None:
    st.write('file uploaded succesfully')
    #st.write(input_data)
    #st.write(type(input_data))

load_model = open('cnn_data_label.pkl', 'rb')   
st.write(pickle.load(load_model)) 

 #load_model = open(input_data._file_urls, 'rb') 
model = pickle.load(input_data)


st.write(model)
