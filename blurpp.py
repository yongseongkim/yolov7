import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
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


def blur_people_with_masks(img, masks):
    blurred_img = img.copy()
    for mask in masks:
        total_blurred_img = cv2.GaussianBlur(img, (0, 0), 4)
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


def add_suffix_to_filename(filename, suffix):
    # Split the filename into the base name and the extension
    base, ext = os.path.splitext(filename)

    # Concatenate the suffix to the base name
    new_filename = base + suffix + ext

    return new_filename


def blurpp_with_sam(sam, img_path, key, boxes, output_dir):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    predictor = SamPredictor(sam)
    predictor.set_image(img)
    output_img = img.copy()
    print(f'Blurring {len(boxes)} people in the image({key}).')
    for box in boxes:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        masks, _, _ = predictor.predict(box=np.array([x1, y1, x2, y2]), point_coords=None, point_labels=None)
        output_img = blur_people_with_masks(output_img, masks)
    print(f'Blurring done.', end='\n')
    show_comparison(img, output_img)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{output_dir}/{add_suffix_to_filename(key, "_b")}', output_img)


def blurpp(mask_rcnn, img_path, key, boxes, output_dir):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform Mask R-CNN on each detected person bounding box
    output_img = img.copy()
    print(f'Blurring {len(boxes)} people in the image({key}).')

    if not boxes:
        masks = get_person_masks(mask_rcnn, img)
        output_img = blur_people_with_masks(img, masks)
    else:
        for box in boxes:
            box = {key: int(value) for key, value in box.items()}
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            partial_img = img[y1:y2, x1:x2:]
            masks = get_person_masks(mask_rcnn, img)
            partial_blurred_img = blur_people_with_masks(partial_img, masks)
            output_img[y1:y2, x1:x2, :] = partial_blurred_img
            # show_comparison(img, partial_img)
    # show_comparison(img, output_img)
    print(f'Blurring done.', end='\n')
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{output_dir}/{add_suffix_to_filename(key, "_b")}', output_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source', default='runs/detected/result.json', help='json path for person boxes')
    opt = parser.parse_args()

    # model = maskrcnn_resnet50_fpn_v2(pretrained=True)
    model = sam_model_registry['vit_b'](checkpoint="sam_vit_b_01ec64.pth")
    model.eval()

    with open(opt.source) as src:
        boxes = json.load(src)
        print(f'Load boxes from {opt.source}, {len(boxes)} found.')
        client = dwnls3imgs.get_vault_session('scc')
        idx = 0
        output_dir = f'./runs/blurred'
        for (k, v) in boxes.items():
            img_path = dwnls3imgs.download_obj(client, 'scc-prod-accessibility-images', k, output_dir)
            blurpp_with_sam(model, img_path, k, v, output_dir)
            # blurpp(model, img_path, k, v, output_dir)
            idx += 1
            if idx > 10:
                break
