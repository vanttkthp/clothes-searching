import os
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np

# Hàm tạo model
def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.get_layer("fc1").output)
    return extract_model

# Hàm tiền xử lý ảnh
def image_preprocess(img):
    img = img.resize((224, 224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Hàm trích xuất vector đặc trưng
def extract_vector(model, image_path):
    img = Image.open(image_path)
    img_tensor = image_preprocess(img)
    vector = model.predict(img_tensor)[0]
    vector = vector / np.linalg.norm(vector)  # Chuẩn hóa vector
    return vector

# Thư mục ảnh và nhãn
base_folder = r"C:\Users\Acer\Dropbox\PC\Desktop\doan\python\crawler\downloaded_images"
categories = ["tops", "footwear", "bottoms", "accessories"]  # Các nhãn

# Khởi tạo model
model = get_extract_model()

vectors = []
paths = []
labels = []

for label in categories:
    category_folder = os.path.join(base_folder, label)
    for image_name in os.listdir(category_folder):
        image_path = os.path.join(category_folder, image_name)
        vector = extract_vector(model, image_path)
        vectors.append(vector)
        paths.append(image_path)
        labels.append(label)

# Lưu vector, đường dẫn và nhãn vào file
vector_file = r"C:\Users\Acer\Dropbox\PC\Desktop\doan\python\MiAI_Image_Search\vectors.pkl"
path_file = r"C:\Users\Acer\Dropbox\PC\Desktop\doan\python\MiAI_Image_Search\paths.pkl"
label_file = r"C:\Users\Acer\Dropbox\PC\Desktop\doan\python\MiAI_Image_Search\labels.pkl"

pickle.dump(vectors, open(vector_file, "wb"))
pickle.dump(paths, open(path_file, "wb"))
pickle.dump(labels, open(label_file, "wb"))

print("Đã lưu vectors, paths và labels.")
