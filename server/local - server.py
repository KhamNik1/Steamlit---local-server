import io
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Функция для распознавания объектов на изображении
def detect_objects(model, image):
    output = model.predict(image)  # , save=True
    return output


# Функция для загрузки модели YOLO
def load_model():
    model = YOLO("C:\\Users\\Desktop\\[]\\job\\RUNы\\DS1.7 yolon-100\\runs\\segment\\train\\weights\\last.pt")
    return model


# Функция для загрузки изображения через Streamlit
def load_image():
    # Виджет для загрузки файла с изображением
    uploaded_file = st.file_uploader(label='Выберите изображение-видео для распознавания')

    # Проверка, было ли выбрано изображение
    if uploaded_file is not None:
        # Открытие изображения с использованием библиотеки PIL
        image = Image.open(io.BytesIO(uploaded_file.read()))

        # Отображение изображения в интерфейсе Streamlit
        st.image(image)

        # Возвращение объекта изображения
        return image
    else:
        return None


# Загрузка модели при запуске приложения
model = load_model()

# Заголовок Streamlit
st.title('Сегментация изображения')

# Загрузка изображения через пользовательский интерфейс
img = load_image()

# Кнопка для запуска распознавания объектов на изображении
result = st.button('Распознать изображение')

# Проверка, была ли нажата кнопка, и выбрано ли изображение
if result and img is not None:
    # Отображение результатов распознавания
    st.write('**Результаты распознавания:**')

    # Вызов функции для распознавания
    output = detect_objects(model, img)

    print(output[0].boxes.xywh.numpy())
    print(output[0].boxes.xywhn.numpy())
    print(output[0].masks)
    # Отображение изображений сегментации
    for segmented_image in output:
        # Отображение изображения в интерфейсе Streamlit
        img_array = segmented_image.plot()
        seg_img = Image.fromarray(img_array[..., ::-1])
        st.image(seg_img)

        st.write('Класс', (output[0].boxes.cls.numpy()))
        st.write('xywh', output[0].boxes.xywh.numpy())
        st.write('xywhn', output[0].boxes.xywhn.numpy())
        st.write('masks', output[0].masks.data.shape)
