import numpy as np
from Model import *
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

# Convert image to numpy
def load_image_into_numpy_array(data):
    image = Image.open(BytesIO(data))
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    return np.array(image)

# Function detect person
def detect_person(image):
    # Ensure the image is in RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Run YOLOv8 detection
    results = person_model(image_rgb)
    # Check for detections of class 'person'
    person_class_id = list(person_model.names.keys())[list(person_model.names.values()).index("person")]
    
    for result in results:
        for box in result.boxes:
            if int(box.cls) == int(person_class_id):
                confidence = float(box.conf) 
                return True, confidence 
    return False, 0.0



# Function to extract features for a single image
def extract_features(image, model, transform):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        image=image.to(device)
        features = model(image)
    return features.cpu().numpy()

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model Image Retrival
def retrieve_top_n_closest_labels(input_features_list, feature_database, labels, N=10):
    # top_n_closest_labels = []
    feature_database = np.array(feature_database)
    # Duyệt qua từng vector đầu vào
    for input_features in input_features_list:
        input_features = np.array(input_features)
        if input_features.ndim == 1:
            input_features_reshaped = input_features.reshape(1, -1)
        else:
            input_features_reshaped = input_features 
        # Tính độ tương đồng cosine với toàn bộ cơ sở dữ liệu
        similarities = cosine_similarity(input_features_reshaped, feature_database)
        # Lấy chỉ số của N đặc trưng gần nhất
        top_n_indices = np.argsort(similarities[0])[::-1][:N]
        closest_labels = [labels[i] for i in top_n_indices]
        # top_n_closest_labels.append(closest_labels)
        # print("Closest labels:", closest_labels)

    return closest_labels

def get_filename_from_url(url: str) -> str:
    return url.split("/")[-1]

# def get_formatted_filename(url):
#     filename_with_extension = os.path.basename(url)
#     # Split filename and discard the numbers at the end, keep the first part and the extension
#     name_parts = filename_with_extension.split('_')
#     if '.' in name_parts[-1]:
#         first_part = '_'.join(name_parts[:-1])  # Combine everything except the last number part
#         extension = name_parts[-1].split('.')[-1]
#         return f"{first_part}.{extension}"
    
#     return filename_with_extension