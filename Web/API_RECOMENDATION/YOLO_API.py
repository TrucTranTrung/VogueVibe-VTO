import numpy as np
from typing import List, Optional
import requests
from Model import *
from Support import *
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# CORS Config
origins = [
    "http://localhost:3000",  # Front-end Address
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# Class of fashion model
# class_list = ['Bag', 'Dress', 'Head_wear', 'Pants', 'Shirt', 'Shoes', 'Shorts']
class_list = ['bag', 'dress', 'headwear', 'pants', 'shirt', 'shoes', 'short']
url_Images = "http://localhost:4000/allimages/detect"
nameImage_list = []
urlImage_list =[]

class FashionResponse(BaseModel):
    dataRes: List[dict]
    status: bool
    message: str

@app.post("/api-detect", response_model=FashionResponse)
async def detect_gender(file: UploadFile = File(...)):
    #detect-gender
    img = load_image_into_numpy_array(await file.read())
    result_person, confidence_person =detect_person(img)
    
    if result_person==True and confidence_person>0.5:
        inputs = processor(images=img, return_tensors="pt")
        inputs=inputs.to(device)
        with torch.no_grad():
            outputs = model_gen(**inputs)
        # Apply softmax to get probabilities
        probabilities = F.softmax(outputs.logits, dim=1)
        # Define label map for interpreting the output
        label_map = {0: "FEMALE", 1: "MALE"}
        # Get the predicted class index and confidence
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        predicted_label_Gen = label_map[predicted_class_idx]
        # print(predicted_label_Gen)
        # Dont need
        # confidence_gen = probabilities[0, predicted_class_idx].item() 
    
        #detect-fashion
        results = fashion_model(img)
        result_fashion = results[0]
        data = result_fashion.boxes
        labels = data.cls.tolist()
        detections = data.xyxy.cpu().numpy() 
        cropped_images = []
        categorys = []

        for i, detection in enumerate(detections):
            label_idx = int(labels[i]) 
            xmin, ymin, xmax, ymax = map(int, detection[:4])

            fashion_confidence = float(data.conf[i])
            if fashion_confidence <= 0.6 :
                continue
            # dataRes
            categorys.append(class_list[label_idx])

            # Crop images
            cropped_img = img[ymin:ymax, xmin:xmax]
            cropped_images.append(cropped_img)
        # print("Categorys:", cropped_images)
        try:
            # Gửi yêu cầu tới server bên ngoài
            response = requests.get(url_Images, json={"gender": predicted_label_Gen, "category": categorys})
            Yolo_result =[] 
            if response.status_code == 200:
                serverResImg = response.json()
                if not serverResImg:
                    return FashionResponse(dataRes=[], status=False, message="Not found proudct match")
                #image retrival
                data = np.load(path_saved_features)
                labels = np.load(path_label_model)
                # print(labels[1])
                #i=0
                vector=[]
                vector_name=[]
                top_n =[]
                input_features_list = []
                
                mask1 = np.isin(labels, serverResImg)
                new_labels = np.char.lower(labels)
                for i in range(len(categorys)):
                    category = categorys[i]
                    # print("Category:", category)
                    if category == "head_wear":
                        category = "headwear"
                    if category == "shorts":
                        category = "short"
                    # print("Category:", category)
                    image = cropped_images[i]
                    mask2 = np.char.find(new_labels, category) >= 0
                    combined_mask = mask1 & mask2
                    indices = np.where(combined_mask)[0]
                    # print("Any element in labels exists in serverResImg:", np.any(mask1))
                    # print("Any element in labels exists in category:", np.any(mask2))
                    # print(len(mask1))
                    # print(len(mask2))
                    # print(len(combined_mask))
                    # print("Indices:", indices)
                    # print("Indices:", indices.shape)
                    # Check if index array is empty
                    if len(indices) == 0:
                        continue
                    vector = data[indices, :]
                    # print("Vector:", vector[0])
                    vector_name = labels[indices]
                    # print("Vector Name:", len(vector_name))
                    # print("Vector:", vector_name.shape)
                    # print("Vector Name:", vector_name)
                    input_features = extract_features(image, model, transform)
                    input_features_list.append(input_features)
                    top_n_closest_labels = retrieve_top_n_closest_labels(input_features_list, vector, vector_name)
                    top_n.append(top_n_closest_labels)
                    

                # for url in serverResImg:
                #     # url = get_filename_from_url(url)
                #     print("URL:", url)
                #     if url in labels:
                #         # print("Labels:", labels)
                #         index = np.where(labels == url)[0][0]  # Get the index of the label
                #         vector.append(data[index, :])
                #         vector_name.append(labels[index])
                # print("Vector_name:", vector_name)
                
                # print("Vector:", vector_name)
                
                # image retrival
                # for i in range(len(categorys)):
                #     label = categorys[i]
                #     img = cropped_images[i]
                #     input_features = extract_features(image, model, transform)
                #     if label in labels:
                #         index = np.where(labels == label)[0]
                #         vector = data[index]
                #         vector_name = labels[index]


                
                # print("Top N Closest Labels:", top_n_closest_labels)
                #image retrival
                # print("Input Features Shape:", [f.shape for f in input_features_list])
                # print("Vector Shape:", np.array(vector).shape)
                # print("Vector Names:", vector_name[:10])
                # print("Top N:", top_n)
                Yolo_result.append({"categorys": categorys, "gender": predicted_label_Gen , "data": top_n})
                if top_n:
                    return FashionResponse(dataRes=Yolo_result, status=True, message="Success request")
                else:
                    return FashionResponse(dataRes=[], status=False, message="Server not response")
            else:
                return FashionResponse(dataRes=[], status=False, message="Server not response")

        except Exception as e:
            print(e)
            return FashionResponse(dataRes=[], status=False, message=str(e))
        
    # Detect not Human
    else:
        return FashionResponse(dataRes=[], status=False, message="Server not response") 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("YOLO_API:app", host="127.0.0.1", port=8000, reload=True)
