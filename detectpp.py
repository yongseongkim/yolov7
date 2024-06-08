import argparse
import json
import time
from pathlib import Path

import torch

import dwnls3imgs
from models.experimental import attempt_load
from utils.datasets import LoadRemoteImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, increment_path
from utils.torch_utils import select_device, time_synchronized


def detectpp(urls):
    # Initialize
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load('yolov7.pt', map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = opt.img_size
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    modelc = None
    # modelc = load_classifier(name='resnet101', n=2)  # initialize
    # modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    dataset = LoadRemoteImages(urls, img_size=imgsz, stride=stride)
    # Get names
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    result = {}
    output_dir = Path(increment_path(Path(opt.output), False))  # increment run
    for p, img, im0s in dataset:
        path = Path(p)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                _ = model(img, augment=opt.augment)[0]
        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if modelc:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        ppos = []
        for i, det in enumerate(pred):  # detections per image
            detected, im0, frame = '', im0s, getattr(dataset, 'frame', 0)
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    detected += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                for *xyxy, conf, cls in reversed(det):
                    label = names[int(cls.item())]
                    x1, y1, x2, y2 = [v.item() for v in xyxy]
                    if cls.item() == 0.0:
                        ppos.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
                    print(f'{label} is found. ({x1}, {y1}) - ({x2}, {y2})', end='\n')
            # Print time (inference + NMS)
            if detected:
                print(f' {detected} detected.')
            else:
                print(f' Nothing detected.', end='\n')
            print(f'{path.name} Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        if ppos:
            result[path.name] = ppos
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)  # make dir
            with open(output_dir / 'result.json', 'w') as fp:
                json.dump(result, fp)
    print(f'Total Done. ({time.time() - t0:.3f}s)')

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/detected', help='save results to file')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.75, help='object confidence threshold')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()

    print(f'Searching urls from scc-prod-accessibility-images.')
    urls = dwnls3imgs.get_object_keys('scc-prod-accessibility-images')
    print(f'Stating detection on {len(urls)} images.')
    result = detectpp(urls)
    print(f'Detection done on {len(urls)} images.')
