from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import scipy.io as sio
import build_core_by_index 


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args('')

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.set_default_tensor_type('torch.FloatTensor')



torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}



test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.fc1 = nn.Linear(784, 2048)
        #self.fc2 = nn.Linear(2048, 1024)
        #self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        #x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = x.view(-1, 784) # 13*13*16 = 2704
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        #return F.log_softmax(x, dim=1)
        return x
model = Net()
#print(model.fc1.weight)

#
# load model
#

pretrained_dict = torch.load('/home/wade/Document2/experiment/TT-haar-devolope/pytorch-training-mobile.pt')
model_dict = model.state_dict()


# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# 3. load the new state dict
model.load_state_dict(pretrained_dict)


#print(model)

model.cuda()



model.eval()
test_loss = 0
correct = 0
for data, target in test_loader:
    if args.cuda:
        data = data.cuda()
    data = Variable(data)
    output = model(data)


#output.shape is (1000, 784)
#target.shape is (1000, 1)
    
    
# output vector for the rest of model

tmp = output[2,:]
print(tmp.shape)
# convert torch.Size to numpy array
feature_x = tmp.data.cpu().numpy()

print('-------------start---------------')

feature_x = np.reshape(feature_x,[784,1])
print('feature_x.shape',feature_x.shape)
# allocate memory 
layer1 = np.zeros(2048, dtype=float, order='F')


input_test = feature_x



filename = '/home/wade/Document2/experiment/TT-haar-devolope/weights/mobile_net_fc1_out.mat'
layer1   = build_core_by_index.tt_construct_mobile(filename, input_test)
layer1_1 = layer1 
layer1_1 = build_core_by_index.Relu_Function(layer1_1)


print('layer1_1 done')
print(layer1_1.shape)


print('----------------------------')

feature_x = layer1_1
print('feature_x.shape',feature_x.shape)
# allocate memory 
layer2 = np.zeros(1024, dtype=float, order='F')


input_test = feature_x

filename = '/home/wade/Document2/experiment/TT-haar-devolope/weights/mobile_net_fc2_out.mat'
layer2   = build_core_by_index.tt_construct_mobile(filename, input_test)

layer2_1 = layer2 
layer2_1 = build_core_by_index.Relu_Function(layer2_1)

print('layer2_1 done')
print(layer2_1.shape)


print('----------------------------')


feature_x = layer2_1
print('feature_x.shape',feature_x.shape)
# allocate memory 
layer3 = np.zeros(10, dtype=float, order='F')


input_test = feature_x

filename = '/home/wade/Document2/experiment/TT-haar-devolope/weights/mobile_net_fc3_out.mat'
layer3   = build_core_by_index.tt_construct_mobile(filename, input_test)

layer3_1 = layer3 
layer3_1 = build_core_by_index.Relu_Function(layer3_1)

print('layer3_1 done')
print(layer3_1.shape)


out = build_core_by_index.softmax(layer3_1)
pred_ans = out


print(pred_ans)
print(target[0:5])

