import torch
import torch.nn as nn
import numpy as np

from yolov3.layers import predict_transform
from yolov3.util import parse_cfg, create_modules, convert2cpu

class YoloV3Net(nn.Module):
    def __init__(self, cfgfile):
        super(YoloV3Net, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0
        
    def forward(self, x, CUDA):
        detections = []
        modules = self.blocks[1:]
        outputs = {} # Route layer output cache
        write = 0
        for i in range(len(modules)):        
            module_type = (modules[i]["type"])
            if module_type == "convolutional" or module_type == "upsample" or module_type == "maxpool":
                x = self.module_list[i](x)
                outputs[i] = x
            elif module_type == "route":
                layers = modules[i]["layers"]
                layers = [int(a) for a in layers]
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i  
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                outputs[i] = x
            elif  module_type == "shortcut":
                from_ = int(modules[i]["from"])
                x = outputs[i-1] + outputs[i+from_]
                outputs[i] = x
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                # Get the input dimensions
                inp_dim = int (self.net_info["height"])
                # Get the number of classes
                num_classes = int (modules[i]["classes"])
                # Output the result
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if type(x) == int:
                    continue
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
                outputs[i] = outputs[i-1]
        try:
            return detections
        except:
            return 0

    def get_blocks(self):
        return self.blocks
    
    def get_module_list(self):
        return self.module_list

    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")
        # The first 4 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4. IMages seen 
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        # The rest of the values are the weights
        # Let's load them up
        weights = np.fromfile(fp, dtype = np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]
                if (batch_normalize):
                    bn = model[1]
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr += num_biases
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr += num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

    def save_weights(self, savedfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks) - 1
        fp = open(savedfile, 'wb')
        # Attach the header at the top of the file
        self.header[3] = self.seen
        header = self.header
        header = header.numpy()
        header.tofile(fp)
        # Now, let us save the weights 
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"] 
            if (module_type) == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0  
                conv = model[0]
                if (batch_normalize):
                    bn = model[1]
                    #If the parameters are on GPU, convert them back to CPU
                    #We don't convert the parameter to GPU
                    #Instead. we copy the parameter and then convert it to CPU
                    #This is done as weight are need to be saved during training
                    convert2cpu(bn.bias.data).numpy().tofile(fp)
                    convert2cpu(bn.weight.data).numpy().tofile(fp)
                    convert2cpu(bn.running_mean).numpy().tofile(fp)
                    convert2cpu(bn.running_var).numpy().tofile(fp)
                else:
                    convert2cpu(conv.bias.data).numpy().tofile(fp)
                #Let us save the weights for the Convolutional layers
                convert2cpu(conv.weight.data).numpy().tofile(fp)
