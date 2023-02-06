import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils1 import *

import numpy as np  
np.set_printoptions(threshold=np.inf)  
# model = GINConvNet()
# model.load_state_dict(torch.load('model_GINConvNet_davis.model'))
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
    return total_preds.numpy().flatten()

test_data = TestbedDataset(root='autodl-tmp/GraphDTA-master/data', dataset='FDA_test')
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
cuda_name = "cuda:0"
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
model = GINConvNet().cuda()
save_path='autodl-tmp/GraphDTA-master/model_NONE_SE_GINConvNet_davis.model'
model.load_state_dict(torch.load(save_path))
P = predicting(model, device, test_loader)
print(P.shape)
# print(P)

np.savetxt('autodl-tmp/GraphDTA-master/davis_predict.txt', P, delimiter=',')
