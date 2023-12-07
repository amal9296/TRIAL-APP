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

    

load_model = open('cnn_data.pkl', 'rb') 
model = pickle.load(load_model)

st.write(model)
