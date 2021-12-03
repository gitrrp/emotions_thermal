import streamlit as st
from PIL import Image
from io import BytesIO

from cnn_svm import load_model

# set page layout
st.set_page_config(
    page_title='Emotion Detection',
    initial_sidebar_state='expanded'
)

# load model
model = load_model()

st.title('Emotion Detection using Thermal Facial Images.')
st.sidebar.subheader(body='Input')
src_img = st.sidebar.file_uploader(label='Choose Image', type=['jpg', 'jpeg', 'png'])


if src_img:
    # display image
    st.write('#### Image:')
    img = Image.open(BytesIO(src_img.read()))
    st.image(image=img, width=500)

    submit = st.button('Submit')

    # detect emotion
    if submit:
        emotion, is_lying = model.predict(img)
        if is_lying:
            st.error(f'Emotion: **{emotion}**, Person is **lying**.')
        else:
            st.success(f'Emotion: **{emotion}**, Person is **honest**.')
