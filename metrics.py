from yolov3 import YoloV3Net
import torch
import pandas
from collections import defaultdict, Counter
from iou import intersection_over_union
from util import load_classes
from torch.autograd import Variable
from preprocess import prep_image
from util import write_results
import os
from PIL import ImageDraw, Image

def main():
    print('Loading network.....')
    model = YoloV3Net('cfg/yolov3.cfg')
    model.load_weights('weights/yolov3.weights')
    model.net_info['height'] = '416'
    CUDA = torch.cuda.is_available()
    print('Network successfully loaded')

    classes = load_classes('data/coco.names')
    confidence = 0.4
    nms_thresh = 0.4
    dir = os.path.join('coco128', 'images', 'train2017')

    preds, target = generate_predictions(model, dir, classes, CUDA=CUDA)

    mAP = mean_average_precision(
        pred_boxes=preds,
        true_boxes=target,
        iou_thresh=0.5,
        box_format='corners',
        num_classes=len(classes)
    )
    print(f'mAP: {mAP}')

def generate_predictions(model, dir, classes, confidence=0.4, nms_thresh=0.4, CUDA=False):
    preds = []
    target = []
    for i,image_file in enumerate(os.listdir(dir)):
        image_path = os.path.join(dir, image_file)
        image, orig_im, dim = prep_image(image_path, int(model.net_info['height']))
        bboxes = get_annotation_bbox(image_file[:-4], dim, classes, box_format='corner')
        # if bboxes is None:
        #     print(bboxes, image_path)
        for bbox in bboxes:
            bbox[0] = i
            target.append(bbox)
        with torch.no_grad():
            output = model(Variable(image), CUDA)
            predictions = write_results(output, confidence, len(classes), nms=True, nms_conf=nms_thresh)
            img = Image.open(image_path)
            for prediction in predictions:
                inp_dim = int(model.net_info['height'])
                xmin, ymin, width, height = [int(p) for p in prediction[1:5]]
                acopy = xmin, ymin, width, height
                w, h = img.size
                scale = min([inp_dim / w, inp_dim / h, 1])
                # scale_y = min(inp_dim / h, 1)
                # scale_y = scale_x
                xmin = (xmin - (inp_dim - scale * w) / 2) / scale
                ymin = (ymin - (inp_dim - scale * h) / 2) / scale
                width = (width - (inp_dim - scale * w) / 2) / scale
                height = (height - (inp_dim - scale * h) / 2) / scale

                label, probability = prediction[-1], prediction[5]
                preds.append([i, label, probability, xmin, ymin, width, height])
                p1, p2 = (xmin, ymin), (width, height)
                draw_bbox(img, (p1, p2, classes[int(label)]))
    return preds, target

def draw_bbox(image, bbox):
    p1, p2, label = bbox
    image_draw = ImageDraw.Draw(image)
    image_draw.rectangle((p1, p2), outline ="purple", width=3)
    image_draw.text((p1[0], p2[1]), label)

def get_annotation_bbox(image_file, size, classes, box_format='midpoint'):
    annotation_file = os.path.join('coco128', 'labels', 'train2017', f'{image_file}.txt')
    if annotation_file is None or not os.path.isfile(annotation_file):
        return None
    w, h = size
    bboxes = []

    with open(annotation_file) as f:
        for line in f.readlines():
            label, xmin, ymin, width, height = line.split()
            label_name = classes[int(label)]
            width = float(width) * w
            height = float(height) * h
            p1 = (float(xmin) * w - width / 2, float(ymin) * h - height / 2)
            p2 = (float(xmin) * w + width / 2, float(ymin) * h + height / 2)
            if box_format == 'midpoint':
                bboxes.append((p1, p2, label_name))
            elif box_format == 'corner':
                bboxes.append([0, int(label), 1, p1[0], p1[1], p2[0], p2[1]])
    return bboxes

def mean_average_precision(pred_boxes, true_boxes, iou_thresh=0.5, box_format='midpoint', num_classes=20):
    '''
    Calculates mean average precision.

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
            specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_thresh (float): threshold where predicted bboxes is correct
        box_format (str): 'midpoint' or 'corners' used to specify bboxes
        num_classes (int): number of classes
    
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    '''
    # list storing all AP for respective classes
    average_precisions = []
    recall_tensor = torch.tensor([])
    precision_tensor = torch.tensor([])

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets, and only add the ones
        # that belong to the current class c.
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
        # count[len(ground_truths) > 0] += 1

        # Find the amount of bboxes for each training example
        # defaultdict finds how many ground truth bboxes we get for each training example
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary and convert to
        # the following (w.r.t. same example):
        # amount_bboxes = {0:torch.tensor[0,0,0]. 1:torch.tensor[0,0,0,0,0]}
        for key,val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # Sort by box probabilities (index 2)
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for di, detection in enumerate(detections):
            # Only take out the ground_truths that have the same training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for gi, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_i = gi

            if best_iou > iou_thresh:
                # only detect gorund truth detection once
                if amount_bboxes[detection[0]][best_gt_i] == 0:
                    # True positive and add this bounding box to seen
                    TP[di] = 1
                    amount_bboxes[detection[0]][best_gt_i] = 1
                else:
                    FP[di] = 1
            # If IOU is lower than the detection is a false positive
            else:
                FP[di] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))
        precision_tensor = torch.cat((precision_tensor, precisions))
        recall_tensor = torch.cat((recall_tensor, recalls))
    # print(len(average_precisions))
    return sum(average_precisions) / len(average_precisions)

if __name__ == '__main__':
    main()