import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from PIL import Image
import matplotlib.pyplot as plt
import math
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

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

# Load vector, đường dẫn và nhãn
vector_file = r"C:\Users\Acer\Dropbox\PC\Desktop\doan\python\MiAI_Image_Search\vectors.pkl"
path_file = r"C:\Users\Acer\Dropbox\PC\Desktop\doan\python\MiAI_Image_Search\paths.pkl"
label_file = r"C:\Users\Acer\Dropbox\PC\Desktop\doan\python\MiAI_Image_Search\labels.pkl"

vectors = pickle.load(open(vector_file, "rb"))
paths = pickle.load(open(path_file, "rb"))
labels = pickle.load(open(label_file, "rb"))

# Train mô hình KNN để dự đoán nhãn
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(vectors, labels)

# Dự đoán nhãn cho ảnh tìm kiếm
search_image = r"C:\Users\Acer\Dropbox\PC\Desktop\doan\python\MiAI_Image_Search\testimage\merlin_176571720_78faf934-4447-42a9-92eb-2ac9fe15587b-superJumbo.jpg"

model = get_extract_model()
search_vector = extract_vector(model, search_image)
predicted_label = knn.predict([search_vector])[0]
print(f"Ảnh tìm kiếm thuộc nhãn: {predicted_label}")

# Lọc vector và đường dẫn theo nhãn dự đoán
filtered_vectors = [v for v, l in zip(vectors, labels) if l == predicted_label]
filtered_paths = [p for p, l in zip(paths, labels) if l == predicted_label]

# Tính khoảng cách và tìm ảnh gần nhất
distance = np.linalg.norm(np.array(filtered_vectors) - search_vector, axis=1)
K = 16
ids = np.argsort(distance)[:K]
nearest_images = [(filtered_paths[id], distance[id]) for id in ids]

# Hiển thị kết quả
axes = []
grid_size = int(math.sqrt(K))
fig = plt.figure(figsize=(10, 5))

for id in range(K):
    draw_image = nearest_images[id]
    axes.append(fig.add_subplot(grid_size, grid_size, id + 1))
    axes[-1].set_title(draw_image[1])
    plt.imshow(Image.open(draw_image[0]))

fig.tight_layout()
plt.show()
