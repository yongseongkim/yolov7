import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F

import dwnls3imgs


def get_person_masks(mask_rcnn, image, threshold=0.5):
    image_tensor = F.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        predictions = mask_rcnn(image_tensor)[0]

    masks = []
    for i in range(len(predictions['labels'])):
        if predictions['labels'][i] == 1 and predictions['scores'][i] > threshold:
            # Extract mask and convert it to a binary mask
            mask = predictions['masks'][i, 0].mul(255).byte().cpu().numpy()
            masks.append(mask > 127)
    return masks


def blur_people_in_image(mask_rcnn, img):
    masks = get_person_masks(mask_rcnn, img)

    blurred_img = img.copy()
    for mask in masks:
        total_blurred_img = cv2.GaussianBlur(img, (21, 21), 3)
        blurred_img[mask] = total_blurred_img[mask]
    return blurred_img


def show_comparison(original_img, blurred_img):
    plt.figure(figsize=(20, 10))
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')
    # Blurred image
    plt.subplot(1, 2, 2)
    plt.imshow(blurred_img)
    plt.title("Blurred Image")
    plt.axis('off')
    # Show the images
    plt.show()


def blurpp(mask_rcnn, key, boxes):
    img_path = dwnls3imgs.download_obj('scc-prod-accessibility-images', key, './tmp')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform Mask R-CNN on each detected person bounding box
    output_img = img.copy()
    print(f'Blurring {len(boxes)} people in the image({key}).')

    if not boxes:
        output_img = blur_people_in_image(mask_rcnn, img)
    else:
        for box in boxes:
            box = {key: int(value) for key, value in box.items()}
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            partial_img = img[y1:y2, x1:x2:]
            partial_blurred_img = blur_people_in_image(mask_rcnn, partial_img)
            output_img[y1:y2, x1:x2, :] = partial_blurred_img
            show_comparison(img, partial_img)
    os.remove(img_path)
    print(f'Blurring done.', end='\n')
    # show_comparison(img, output_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='runs/detected/result.json', help='json path for person boxes')
    opt = parser.parse_args()

    mask_rcnn = maskrcnn_resnet50_fpn_v2(pretrained=True)
    mask_rcnn.eval()
    with open(opt.source) as src:
        boxes = json.load(src)
        print(f'Load boxes from {opt.source}, {len(boxes)} found.')
        for (k, v) in boxes.items():
            blurpp(mask_rcnn, k, v)
