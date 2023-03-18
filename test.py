import torch
from torch.autograd import Variable
from yolov3 import YoloV3Net
from preprocess import prep_image
from util import load_classes, write_results

def main():
    print('Loading network.....')
    model = YoloV3Net('cfg/yolov3.cfg')
    model.load_weights('weights/yolov3.weights')
    CUDA = torch.cuda.is_available()
    print('Network successfully loaded')

    # Other hyperparameters and global variables
    model.net_info['height'] = '416'
    classes = load_classes('data/coco.names')
    confidence = 0.4
    nms_thresh = 0.4

    # Set the model in evaluation mode
    model.eval()

    # Preprocess image
    img, orig_img, dim = prep_image('dog-cycle-car.png', int(model.net_info['height']))
    
    # Evaluate image
    with torch.no_grad():
        predictions = model(Variable(img), CUDA)
        predictions = write_results(predictions, confidence, len(classes), nms=True, nms_conf=nms_thresh)
        print(predictions)
        print('Detected:', ', '.join([classes[int(prediction[-1].item())] for prediction in predictions]))
        
if __name__ == '__main__':
    main()