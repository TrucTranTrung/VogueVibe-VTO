import os
import cv2
import numpy as np
from PIL import Image
import torch
from collections import defaultdict
import math


# draw the body keypoint and lims
def draw_bodypose(img, poses,model_type = 'coco'):
    stickwidth = 4
    w,h = img.shape[:2]
    img_black = np.zeros((w,h,3), np.uint8)
    limbSeq = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], \
                   [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], \
                   [0, 15], [15, 17]]
    njoint = 18
    if model_type == 'body_25':    
        limbSeq = [[1,0],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],\
                            [10,11],[8,12],[12,13],[13,14],[0,15],[0,16],[15,17],[16,18],\
                                [11,24],[11,22],[14,21],[14,19],[22,23],[19,20]]
        njoint = 25

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255,255,0], [255,255,85], [255,255,170],\
                  [255,255,255],[170,255,255],[85,255,255],[0,255,255]]
    for i in range(njoint):
        for n in range(len(poses)):
            pose = poses[n][i]
            if pose[2] <= 0:
                continue
            x, y = pose[:2]
            cv2.circle(img_black, (int(x), int(y)), 4, colors[i], thickness=-1)
    
    for pose in poses:
        for limb,color in zip(limbSeq,colors):
            p1 = pose[limb[0]]
            p2 = pose[limb[1]]
            if p1[2] <=0 or p2[2] <= 0:
                continue
            cur_canvas = img_black.copy()
            X = [p1[1],p2[1]]
            Y = [p1[0],p2[0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            img_black = cv2.addWeighted(img_black, 0.4, cur_canvas, 0.6, 0)
   
    return img_black


def rgb_to_indexed_with_class_positions(image_path, label_colours):
    """
    Chuyển đổi ảnh RGB sang chế độ Indexed Color (P), đồng thời in thông tin vị trí các lớp.

    Args:
        image_path (str): Đường dẫn tới ảnh RGB.
        label_colours (list of tuples): Bảng màu (RGB tuples).
    Returns:
        Image: Ảnh chế độ P.
    """
    # Mở ảnh và chuyển sang numpy array
    img = Image.open(image_path).convert("RGB")
    img_rgb = np.array(img)

    # Tạo ảnh đầu ra với chế độ "P"
    img_p = Image.new("P", img.size)

    # Xây dựng bảng màu từ label_colours
    flat_palette = [c for color in label_colours for c in color]
    img_p.putpalette(flat_palette + [0] * (768 - len(flat_palette)))

    # Khởi tạo ảnh indexed (dạng chỉ số)
    indexed_img = np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    # Lưu vị trí của các lớp
    class_positions = defaultdict(list)
    unmatched_pixels = set()

    # Ánh xạ pixel RGB sang chỉ số trong bảng màu
    for y in range(img_rgb.shape[0]):
        for x in range(img_rgb.shape[1]):
            pixel_color = tuple(img_rgb[y, x])
            if pixel_color in label_colours:
                class_index = label_colours.index(pixel_color)
                indexed_img[y, x] = class_index
                class_positions[class_index].append((x, y))
            else:
                indexed_img[y, x] = 0  # Gán về lớp mặc định
                unmatched_pixels.add(pixel_color)

    # In thông tin vị trí các lớp
    # print("Thông tin vị trí các lớp trước khi chuyển đổi:")
    for class_index, positions in class_positions.items():
        print(f"- Lớp {class_index} (màu {label_colours[class_index]}): {len(positions)} pixel")

    # Cảnh báo nếu có màu không hợp lệ
    if unmatched_pixels:
        print(f"\nCảnh báo: Có {len(unmatched_pixels)} pixel không thuộc bảng màu:")
        print(unmatched_pixels)

    # Gán giá trị indexed vào ảnh "P"
    img_p.putdata(indexed_img.flatten())
    return img_p

def gen_noise(shape):
    noise = np.zeros(shape, dtype=np.uint8)
    ### noise
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = torch.tensor(noise, dtype=torch.float32)
    return noise


def save_images(img_tensors, img_names, save_dir):
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone()+1)*0.5 * 255
        tensor = tensor.cpu().clamp(0,255)

        try:
            array = tensor.numpy().astype('uint8')
        except:
            array = tensor.detach().numpy().astype('uint8')

        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)

        im = Image.fromarray(array)
        im.save(os.path.join(save_dir, img_name), format='JPEG')


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError("'{}' is not a valid checkpoint path".format(checkpoint_path))
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))


