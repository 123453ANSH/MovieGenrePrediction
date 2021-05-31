import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd
import pickle

#def load_model():

#data=load_model()
color_names = {
    '#000000': 'black',
    '#ffffff': 'white',
    '#808080': 'dark gray',
    '#b0b0b0': 'light gray',
    '#ff0000': 'red',
    '#800000': 'dark red',
    '#00ff00': 'green',
    '#008000': 'darkgreen',
    '#0000ff': 'blue',
    '#000080': 'dark blue',
    '#ffff00': 'yellow',
    '#808000': 'olive',
    '#00ffff': 'cyan',
    '#ff00ff': 'magenta',
    '#800080': 'purple'
    }

def show_predict_page():
    st.title("Movie Rating Prediction")
    st.write("Enter the plot overview! ")
    #To get Overview
    overview=st.text_area(label='Movie Overview', max_chars=400)
    st.write("Draw the poster!")
    #To get drawing
    #canvas_result = st_canvas(stroke_width = 25,stroke_color = "#fff",background_color = "#000",height = 400,width = 600,drawing_mode = "freedraw",key = "canvas",)
    #Sidebar components
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee", key="sidebar")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
    )
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    #Displaying drawing canvas
    canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="" if bg_image else bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=400,
    width=600,
    drawing_mode=drawing_mode,
    key="canvas",
)
    # Using the data
    #if canvas_result.image_data is not None:
        #st.image(canvas_result.image_data)
    #if canvas_result.json_data is not None:
        #st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))
    getGenre = st.button("Get the Genre!")
    getRating = st.button("Get the Rating!")

    if canvas_result.image_data is not None and getGenre:
        #display genre
        st.write(type(canvas_result.image_data))
        st.write(canvas_result.image_data)
        genre=getGenrePrediction(canvas_result.image_data, overview)
        st.write("Prediction: "+genre)

    if canvas_result.image_data is not None and getRating:
        rating= getRatingPrediction(canvas_result.image_data, overview)
        st.write("Rating: "+str(rating))
    #display genre
def getGenrePrediction(image, text):
    #get the genre prediction from your model and return it
    return 'Comedy'

def getRatingPrediction(image, text):
    #get the rating prediction from the model and return it
    return 7
#def processImageData(image):
    #Dividing the numbers by 255
