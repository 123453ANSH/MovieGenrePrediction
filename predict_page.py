import streamlit as st
import numpy as np
import pickle

#def load_model():

#data=load_model()

def show_predict_page():
    st.title("Movie Rating Prediction")
    st.write("Enter the plot overview and draw the poster! ")
    overview=st.text_area(label='Movie Overview', max_chars=400)
<<<<<<< HEAD
=======
    st.write("Draw the poster!")
    #To get drawing
    #canvas_result = st_canvas(stroke_width = 25,stroke_color = "#fff",background_color = "#000",height = 400,width = 600,drawing_mode = "freedraw",key = "canvas",)
    #Sidebar components
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
    )
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    #Displaying canvas
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
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)
    if canvas_result.json_data is not None:
        st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))

    ok = st.button("Get the Rating!")
    #if ok:
    #display rating
>>>>>>> 4ed517f (added canvas)
