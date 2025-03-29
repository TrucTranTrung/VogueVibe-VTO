import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
import cv2
import traceback
import base64
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
import logging

logger = logging.getLogger(__name__)

sys.path.append('./')
# PyTorch includes
from torch.autograd import Variable
from pydantic import BaseModel
from torchvision import transforms
from src.torch_openpose import torch_openpose
import torch.nn as nn

# Custom includes
import graph
from model import net, device, NormalizeImage, Parsing_model
from dataloaders import custom_transforms as tr
from Support import *
from utils import draw_bodypose

sys.path.append('VITON-HD')  # Thêm đường dẫn tới thư mục chứa test.py
from viton_test import main

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

class Try_on_Response(BaseModel):
    image : str
    message : str

# Transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    NormalizeImage(0.5, 0.5)
])



@app.post("/Virtual-Try-On", response_model=Try_on_Response)
async def merge_images_api(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:    
        product_image = load_image_into_numpy_array(await file1.read())
        human_image = load_image_into_numpy_array(await file2.read())
        product_image = product_image.resize((768, 1024), Image.BICUBIC)
        human_image = human_image.resize((768, 1024), Image.BICUBIC)

        product_image_cv2 = np.array(product_image)
        human_image_cv2 = np.array(human_image)

        # OpenCV sử dụng định dạng BGR thay vì RGB, vì vậy bạn cần chuyển đổi từ RGB sang BGR
        product_image_cv2 = cv2.cvtColor(product_image_cv2, cv2.COLOR_RGB2BGR)
        human_image_cv2 = cv2.cvtColor(human_image_cv2, cv2.COLOR_RGB2BGR)
        # product_image.save(os.path.join("datasets", "cloth", "01260_00.jpg"))
        product_image.save(os.path.join("VITON-HD", "datasets", "test", "cloth", "02783_00.jpg"))
        human_image.save(os.path.join("VITON-HD", "datasets", "test", "image", "00891_00.jpg"))
        
        # Masking clothes
        # img = Image.fromarray(product_image)
        img_resized = product_image.resize((768, 768), Image.BICUBIC)

        # Apply transformations
        img_tensor = transform(img_resized).unsqueeze(0).to(device)

        # Run model
        with torch.no_grad():
            output_tensor = net(img_tensor)
            output_tensor = F.softmax(output_tensor[0], dim=1)
            output_tensor = torch.argmax(output_tensor, dim=1)

        # Convert output tensor to mask
        output_arr = output_tensor.squeeze().cpu().numpy()
        clothing_mask = (output_arr == 1).astype(np.uint8) * 255  # 255 for clothing, 0 for background

        # Save mask image
        mask_img = Image.fromarray(clothing_mask).resize((768,1024), Image.BICUBIC)
        mask_img.save(os.path.join("VITON-HD", "datasets", "test", "cloth-mask", "02783_00.jpg"))

        # Human parsing
        
        use_gpu = True
        adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
        adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)

        adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
        adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

        cihp_adj = graph.preprocess_adj(graph.cihp_graph)
        adj3_ = Variable(torch.from_numpy(cihp_adj).float())
        adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()

        # multi-scale
        scale_list = [1, 0.5, 0.75, 1.25, 1.5, 1.75]
        img_parsing = human_image
        img_parsing = img_parsing.resize((192, 256), Image.NEAREST)
        testloader_list = []
        testloader_flip_list = []
        for pv in scale_list:
            composed_transforms_ts = transforms.Compose([
                tr.Scale_only_img(pv),
                tr.Normalize_xception_tf_only_img(),
                tr.ToTensor_only_img()])

            composed_transforms_ts_flip = transforms.Compose([
                tr.Scale_only_img(pv),
                tr.HorizontalFlip_only_img(),
                tr.Normalize_xception_tf_only_img(),
                tr.ToTensor_only_img()])

            testloader_list.append(img_transform(img_parsing, composed_transforms_ts))
            testloader_flip_list.append(img_transform(img_parsing, composed_transforms_ts_flip))
        Parsing_model.eval()

        for iii, sample_batched in enumerate(zip(testloader_list, testloader_flip_list)):
            inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
            inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
            inputs = inputs.unsqueeze(0)
            inputs_f = inputs_f.unsqueeze(0)
            inputs = torch.cat((inputs, inputs_f), dim=0)
            if iii == 0:
                _, _, h, w = inputs.size()

            inputs = Variable(inputs, requires_grad=False)

            with torch.no_grad():
                if use_gpu >= 0:
                    inputs = inputs.cuda()
                outputs = Parsing_model.forward(inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda())
                outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
                outputs = outputs.unsqueeze(0)

                if iii > 0:
                    outputs = nn.functional.interpolate(outputs, size=(h, w), mode='bilinear', align_corners=True)
                    outputs_final = outputs_final + outputs
                else:
                    outputs_final = outputs.clone()
        ################ plot pic
        predictions = torch.max(outputs_final, 1)[1]
        results = predictions.cpu().numpy()
        vis_res = decode_labels(results)

        parsing_im = Image.fromarray(vis_res[0])
        parsing_im = parsing_im.convert("RGB")
        # Your label colors
        label_colours = [
        (0, 0, 0), (128, 0, 0), (255, 0, 0), (0, 85, 0), (170, 0, 51),
        (255, 85, 0), (0, 0, 85), (0, 119, 221), (85, 85, 0), (0, 85, 85),
        (85, 51, 0), (52, 86, 128), (0, 128, 0), (0, 0, 255), (51, 170, 221),
        (0, 255, 255), (85, 255, 170), (170, 255, 85), (255, 255, 0), (255, 170, 0)
        ]

        # Flatten the label colors into a single list for the palette
        palette = [value for color in label_colours for value in color]
        palette += [0] * (768 - len(palette))

        # Convert the RGB image to P mode with the custom palette
        p_image = Image.new("P", parsing_im.size)
        p_image.putpalette(palette)

        # Quantize the image to use only the custom palette colors
        p_image = parsing_im.quantize(palette=p_image)
        resized_img = p_image.resize((768, 1024), Image.NEAREST)
        resized_img.save(os.path.join("VITON-HD", "datasets", "test", "image-parse", "00891_00.png"))

        # Open pose
        tp = torch_openpose('body_25')
        # img_pose = cv2.imread(os.path.join("datasets", "image", "test.png"))
        img_pose = human_image_cv2
        poses = tp(img_pose)
        img_posing = draw_bodypose(img_pose, poses,'body_25')
        img_pil = Image.fromarray(img_posing)
        img_pil.save(os.path.join("VITON-HD", "datasets", "test", "openpose-img", "00891_00_rendered.png"))
        # Convert nested list into one list
        flattened_list = [coord for sublist in poses for point in sublist for coord in point]
        # JSON format
        data = {
            "people": [
                {
                    "pose_keypoints_2d": flattened_list
                }
            ]
        }
        file_path = os.path.join("VITON-HD", "datasets", "test", "openpose-json", "00891_00_keypoints.json")
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        sys.argv = [
            'test.py',
            '--name', 'Results',
            '--dataset_dir', './VITON-HD/datasets',  
            '--dataset_mode', 'test', 
            '--dataset_list', 'test_pairs.txt',
            '--checkpoint_dir', './VITON-HD/checkpoints', 
        ]
        main()

        result_path = r".\results\Results\00891_02783_00.jpg"
        
        if not os.path.exists(result_path):
            # return Try_on_Response(image="hehe", message=f"Failed request")
            return JSONResponse(
                    status_code=400,  
                    content={"message": "Failed request"}
                )
            
        with open(result_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
            
        return Try_on_Response(
            image=f"{base64_image}", 
            message="Success request"
        )

    except Exception as e:
        error_msg = f"Error: {str(e)}\n"
        error_msg += f"Traceback:\n{traceback.format_exc()}"
        print(error_msg)  # In ra console
        logging.error(error_msg)  # Ghi vào log file
        # return Try_on_Response(image="hehe", message=f"Failed request: {str(e)}")
        return JSONResponse(
                    status_code=400,  # Bad Request
                    content={"message": "Failed request"}
                )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("API_Virtual:app", host="127.0.0.1", port=8000, reload=True)
