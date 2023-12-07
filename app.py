import streamlit as st 
import pickle 
from PIL import Image 
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import argmax
from tqdm import tqdm



class Basic_functions:
    def upload_image():
        input_data = st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])
        if input_data is not None:
            file_bytes = np.asarray(bytearray(input_data.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            return opencv_image


    def open_model(model_name):
        load_model = open(model_name, 'rb') 
        model = pickle.load(load_model)
        return model
    


    def pred(input_data,model,model_name):

        if model_name == 'CNN_tumor2.pkl':

            img=Image.fromarray(input_data)
            img=img.resize((128,128))
            img=np.array(img)
            input_img = np.expand_dims(img, axis=0)
            res = model.predict(input_img)
            if res:
                st.write("Tumor Detected")
            else:
                st.write("No Tumor")




        if model_name == 'RNN_smsspam1.pkl':
            st.write('------------')
            load_model = open('RNN_smsspam_tokeniser.pkl', 'rb') 
            tokeniser = pickle.load(load_model)
            max_length = 10
            encoded_test = tokeniser.texts_to_sequences(input_data)
            input_data = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')
            preds = (model.predict(input_data) > 0.5).astype("int32")
            if preds[0] == [0]:
                st.write('The given message is ham')

            elif preds[0]== [1]:
                st.write('The given message is spam')


        if model_name == 'LSTM_custom1.pkl':
            st.write('------------')
            load_model = open('RNN_smsspam_tokeniser.pkl', 'rb') 
            tokeniser = pickle.load(load_model)
            max_length = 50
            encoded_test = tokeniser.texts_to_sequences([input_data])
            input_data = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')
            preds = (model.predict(input_data) > 0.5).astype("int32")
            if preds[0] == [0]:
                st.write('The given message is ham')

            elif preds[0]== [1]:
                st.write('The given message is spam') 



        if model_name == 'DNN_spam_model.pkl':
            st.write('------------')
            load_model = open('RNN_smsspam_tokeniser.pkl', 'rb') 
            tokeniser = pickle.load(load_model)
            max_length = 10
            encoded_test = tokeniser.texts_to_sequences([input_data])
            input_data = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')
            preds = (model.predict(input_data) > 0.5).astype("int32")
            if argmax(preds)==0:
                st.write('The given message is ham')

            elif argmax(preds)==1: 
                st.write('The given message is spam')   



        if model_name == 'Backprop_spam_model.pkl':
            st.write('------------')
            load_model = open('RNN_smsspam_tokeniser.pkl', 'rb') 
            tokeniser = pickle.load(load_model)
            max_length = 10
            encoded_test = tokeniser.texts_to_sequences([input_data])
            input_data = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')
            preds = model.predict([input_data])
            if preds==[0]:
                st.write('The given message is ham')

            elif preds==[1]: 
                st.write('The given message is spam')



        if model_name == 'perceptron_spam_model.pkl':
            st.write('------------')
            load_model = open('RNN_smsspam_tokeniser.pkl', 'rb') 
            tokeniser = pickle.load(load_model)
            max_length = 10
            encoded_test = tokeniser.texts_to_sequences([input_data])
            input_data = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')
            preds = model.predict([input_data])
            if preds==[0]:
                st.write('The given message is ham')

            elif preds==[1]: 
                st.write('The given message is spam') 






class BackPropogation:
    def __init__(self,learning_rate=0.01, epochs=100,activation_function='step'):
        self.bias = 0
        self.learning_rate = learning_rate
        self.max_epochs = epochs
        self.activation_function = activation_function


    def activate(self, x): 
        if self.activation_function == 'step':
            return 1 if x >= 0 else 0
        elif self.activation_function == 'sigmoid':
            return 1 if (1 / (1 + np.exp(-x)))>=0.5 else 0
        elif self.activation_function == 'relu':
            return 1 if max(0,x)>=0.5 else 0

    def fit(self, X, y):
        error_sum=0
        n_features = X.shape[1]
        self.weights = np.zeros((n_features))                                    # returns n dimensional matrics with zero
        for epoch in tqdm(range(self.max_epochs)):
            for i in range(len(X)):
                inputs = X[i]
                target = y[i]
                weighted_sum = np.dot(inputs, self.weights) + self.bias
                prediction = self.activate(weighted_sum)
                
                # Calculating loss and updating weights.
                error = target - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error
                
            print(f"Updated Weights after epoch {epoch} with {self.weights}")
        print("Training Completed")

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            inputs = X[i]
            weighted_sum = np.dot(inputs, self.weights) + self.bias
            prediction = self.activate(weighted_sum)
            predictions.append(prediction)
        return predictions
    


class Perceptron:
    
    def __init__(self,learning_rate=0.01, epochs=100,activation_function='step'):
        self.bias = 0
        self.learning_rate = learning_rate
        self.max_epochs = epochs
        self.activation_function = activation_function


    def activate(self, x):
        if self.activation_function == 'step':
            return 1 if x >= 0 else 0
        elif self.activation_function == 'sigmoid':
            return 1 if (1 / (1 + np.exp(-x)))>=0.5 else 0
        elif self.activation_function == 'relu':
            return 1 if max(0,x)>=0.5 else 0

    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.random.randint(n_features, size=(n_features))
        for epoch in tqdm(range(self.max_epochs)):
            for i in range(len(X)):
                inputs = X[i]
                target = y[i]
                weighted_sum = np.dot(inputs, self.weights) + self.bias
                prediction = self.activate(weighted_sum)
        print("Training Completed")

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            inputs = X[i]
            weighted_sum = np.dot(inputs, self.weights) + self.bias
            prediction = self.activate(weighted_sum)
            predictions.append(prediction)
        return predictions                






def main():

    st.title("ML MODEL")     

    option = st.selectbox("Why are you here?",('Tumor prediction','Sentiment analysis'),index=None,placeholder="Select problem method...",)

    st.write('You selected:', option)

    if option == None:
        pass

    elif option == 'Tumor prediction':

        
        st.write('Upload tumer data')
        model_name = 'CNN_tumor2.pkl'
        
        input_data = Basic_functions.upload_image()
        if input_data is not None:
            st.image(input_data, channels="BGR")

        but = st.button("Predict", type="primary")    
        if but:
            model = Basic_functions.open_model('CNN_tumor2.pkl')
            Basic_functions.pred(input_data,model,model_name)       
        
    elif option == 'Sentiment analysis':
        out =  st.radio(
            "Select your prediction 👉",
            key="visibility",
            options=["Recurrent Neural Network", "LSTM", "DEEP NEURAL NETWORK", "Back propagation", "Perceptron"],)  

    

        if out == "Recurrent Neural Network":
            model_name = 'RNN_smsspam1.pkl'
            input_data = st.text_input("Text to analyze",)
            but = st.button("Predict", type="primary") 

            if but:
                model = Basic_functions.open_model(model_name)
                Basic_functions.pred(input_data,model,model_name)


        elif out == "LSTM":
            model_name = 'LSTM_custom1.pkl'
            input_data = st.text_input("Text to analyze",)
            but = st.button("Predict", type="primary") 

            if but:
                model = Basic_functions.open_model(model_name)
                Basic_functions.pred(input_data,model,model_name) 

        elif out == "DEEP NEURAL NETWORK": 
            model_name = 'DNN_spam_model.pkl'
            input_data = st.text_input("Text to analyze",)
            but = st.button("Predict", type="primary") 

            if but:
                model = Basic_functions.open_model(model_name)
                Basic_functions.pred(input_data,model,model_name)


        elif out == "Back propagation":
            model_name = 'Backprop_spam_model.pkl'
            input_data = st.text_input("Text to analyze",)
            but = st.button("Predict", type="primary") 

            if but:
                model = Basic_functions.open_model(model_name)
                Basic_functions.pred(input_data,model,model_name)




        elif out == "Perceptron": 
            model_name = 'perceptron_spam_model.pkl'
            input_data = st.text_input("Text to analyze",)
            but = st.button("Predict", type="primary") 

            if but:
                model = Basic_functions.open_model(model_name)
                Basic_functions.pred(input_data,model,model_name)



if __name__ == "__main__":
    main()           
