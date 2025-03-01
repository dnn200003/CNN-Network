#=====================================================
# Single program
# ====================================================
from settings import RESNETTYPE, EPOCHS, FRAC, LR, TRAINING_SORCE, MODELTYPE


if TRAINING_SORCE=="intelnet":  
    from Dictionary_Types.dic_intelnet import DATA_NAME, NUM_CHANNELS, IMG_TYPE
elif TRAINING_SORCE=="IP102_FC_EC":  
    from Dictionary_Types.dic_IP102_FC_EC import DATA_NAME, NUM_CHANNELS, IMG_TYPE
elif TRAINING_SORCE=="IP102_FC":  
    from Dictionary_Types.dic_IP102_FC import DATA_NAME, NUM_CHANNELS, IMG_TYPE
elif TRAINING_SORCE=="IP102_EC":  
    from Dictionary_Types.dic_IP102_EC import DATA_NAME, NUM_CHANNELS, IMG_TYPE

import os
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pandas import DataFrame
from torch import Tensor
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
import random
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
from datetime import date, datetime
today = f"{date.today()}".replace("-","_")
timeS=f"{datetime.now().strftime('%H:%M:%S')}".replace(":","_")
program="CNN_"+MODELTYPE+"_D"+today+"_T"+timeS+DATA_NAME+f"_E{EPOCHS}.xlsx"


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# To print in color during test/train 
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))   

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    

    
#===================================================================    
program = f"Results\\{program}"
print(f"---------{program}----------")              

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#=============================================================================
#                         Data loading 
#============================================================================= 
df = pd.read_csv(f'data/MyFile({DATA_NAME}).csv')
print(df.head())



# merging both folders of  part1 and part2 -- into a single directory
imageid_path = {os.path.splitext(os.path.basename(x))[0]: x
                for x in glob(os.path.join("data", f'*({DATA_NAME})', '*.jpg'))}

if DATA_NAME=="IntelNet" or "IP102_FC_EC":
        for i in range(len(df["image_id"])):
            df["image_id"][i]=str(df["image_id"][i])
#print("path---------------------------------------", imageid_path.get)
df['path'] = df['image_id'].map(imageid_path.get)
df['cell_type'] = df[' fine_label'].map(IMG_TYPE.get)
df['target'] = pd.Categorical(df['cell_type']).codes
#==============================================================
# Custom dataset prepration in Pytorch format
class SkinData(Dataset):
    def __init__(self, df, transform = None):
        
        self.df = df
        self.transform = transform
        
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, index):
        
        if NUM_CHANNELS==3:
            X = Image.open(self.df['path'][index]).resize((64, 64)).convert('RGB')
        else:
            X = Image.open(self.df['path'][index]).resize((64, 64))
        y = torch.tensor(int(self.df['target'][index]))
        
        if self.transform:
            X = self.transform(X)
        
        return X, y


#=============================================================================
# Train-test split    
train, test = train_test_split(df, test_size = 0.2)

train = train.reset_index()
test = test.reset_index()
#=============================================================================
#                         Data preprocessing
#=============================================================================  
# Data preprocessing: Transformation 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), 
                        transforms.RandomVerticalFlip(),
                        transforms.Pad(3),
                        transforms.RandomRotation(10),
                        transforms.CenterCrop(64),
                        transforms.ToTensor(), 
                        transforms.Normalize(mean = mean, std = std)
                        ])
    
test_transforms = transforms.Compose([
                        transforms.Pad(3),
                        transforms.CenterCrop(64),
                        transforms.ToTensor(), 
                        transforms.Normalize(mean = mean, std = std)
                        ])    


dataset_train = SkinData(train, transform = train_transforms)
dataset_test = SkinData(test, transform = test_transforms)

train_iterator = DataLoader(dataset_train, shuffle = True, batch_size = 256)
test_iterator = DataLoader(dataset_test, batch_size = 256)



print(f'Number of training examples: {len(train)}')
print(f'Number of testing examples: {len(test)}')

for x, y in train_iterator:
    print("shape of x = ", x.shape)
    print(type(x))
    break

#=============================================================================
#                    Model definition: ResNet18
#============================================================================= 
# building a ResNet18 Architecture
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet18(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(NUM_CHANNELS, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
#Building blocks for google net 
class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()
        
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size = 1)
        self.branch2 = nn.Sequential(conv_block(in_channels, red_3x3, kernel_size=1), 
                                     conv_block(red_3x3, out_3x3, kernel_size =3, stride=1, padding =1)
                                     )
        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, stride=1, padding=2)
            )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1)
            )

    def forward(self, x):
        #N x filters x 28 x 28
        return torch.cat([self.branch1(x), self.branch2(x),self.branch3(x),self.branch4(x)], 1)


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))
    
class GoogLeNetClient(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000,conv_block=None,Inception_block=None):
        super(GoogLeNetClient, self).__init__()
        self.conv1 = conv_block(in_channels=in_channels, out_channels = 64, kernel_size=7, stride =2, padding = 3)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size =3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024,num_classes)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        #Layer3
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

class MobileNetV3Block(nn.Module):
    # Convolution Block with Conv2d layer, Batch Normalization and ReLU. Act is an activation function. 
    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        kernel_size : int,
        stride : int,
        act = nn.ReLU(),
        groups = 1,
        bn = True,
        bias = False     
        ):
        super().__init__()

        # If k = 1 -> p = 0, k = 3 -> p = 1, k = 5, p = 2. 
        padding = kernel_size // 2 
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.c(x)))
    
class SeBlock(nn.Module):
    # Squeeze and Excitation Block. 
    def __init__(
        self, 
        in_channels : int
        ):
        super().__init__()

        C = in_channels
        r = C // 4
        self.globpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(C, r, bias=False)
        self.fc2 = nn.Linear(r, C, bias=False)
        self.relu = nn.ReLU() 
        self.hsigmoid = nn.Hardsigmoid()

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [N, C, H, W].  
        f = self.globpool(x)
        f = torch.flatten(f,1)
        f = self.relu(self.fc1(f))
        f = self.hsigmoid(self.fc2(f))
        f = f[:,:,None,None]
        # f shape: [N, C, 1, 1]  

        scale = x * f
        return scale

# BNeck
class BNeck(nn.Module):
    # MobileNetV3 Block 
    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        kernel_size : int, 
        exp_size : int,
        se : bool, 
        act : torch.nn.modules.activation,
        stride : int
        ):
        super().__init__()

        self.add = in_channels == out_channels and stride == 1

        self.block = nn.Sequential(
            MobileNetV3Block(in_channels, exp_size, 1, 1, act),
            MobileNetV3Block(exp_size, exp_size, kernel_size, stride, act, exp_size),
            SeBlock(exp_size) if se == True else nn.Identity(),
            MobileNetV3Block(exp_size, out_channels, 1, 1, act=nn.Identity())
        )

    def forward(self, x: Tensor) -> Tensor:
        res = self.block(x)
        if self.add:
            res = res + x

        return res

class MobileNetV3(nn.Module):
    def __init__(self, config_name ="large",in_channels = 3,classes = 1000, ConvBlock=None,BNeck=None):
        super().__init__()
        config = self.config(config_name)
        self.Layer_Count=[2,4]
        
        # First convolution(conv2d) layer. 
        self.conv = ConvBlock(in_channels, 16, 3, 2, nn.Hardswish())
        # Bneck blocks in a list. 
        self.blocks = nn.ModuleList([])
        for c in config:
            kernel_size, exp_size, in_channels, out_channels, se, nl, s = c
            self.blocks.append(BNeck(in_channels, out_channels, kernel_size, exp_size, se, nl, s))
        
        # Classifier 
        last_outchannel = config[-1][3]
        last_exp = config[-1][1]
        out = 1280 if config_name == "large" else 1024
        self.classifier = nn.Sequential(
            ConvBlock(last_outchannel, last_exp, 1, 1, nn.Hardswish()),
            nn.AdaptiveAvgPool2d((1,1)),
            ConvBlock(last_exp, out, 1, 1, nn.Hardswish(), bn=False, bias=True),
            nn.Dropout(0.8),
            nn.Conv2d(out, classes, 1, 1)
        )
        print(self.state_dict().keys())
        self.layers=[]
        for i in range(6):
            if i<self.Layer_Count[0]:
                self.layers.append(f"layer{i+1}")
            else:
                self.layers.append(None)
        print(self.layers)
        
    def _make_layers(self, in_channels, out_channels, block_cfg, width_multiplier):
        layers = []
        for cfg in block_cfg:
            expansion_factor, num_blocks, stride = cfg
            out_channels = int(out_channels * width_multiplier)
            for _ in range(num_blocks):
                layers.append(self.MobileNetV3Block(in_channels, out_channels, expansion_factor, stride))
                in_channels = out_channels
                stride = 1  # Only the first block in each stage has a stride > 1
        return nn.Sequential(*layers)
        
    
    def forward(self, x, volly=None):
        x = self.conv(x)
        x =self.blocks[0](x)
        x =self.blocks[1](x)
        x =self.blocks[2](x)
        x =self.blocks[3](x)
        x =self.blocks[4](x)
        x =self.blocks[5](x)
        x =self.blocks[6](x)
        x =self.blocks[7](x)
        x =self.blocks[8](x)
        x =self.blocks[9](x)
        x =self.blocks[10](x)
        x =self.blocks[11](x)
        x =self.blocks[12](x)
        x =self.blocks[13](x)
        x =self.blocks[14](x)

        x = self.classifier(x)
        y_hat =torch.flatten(x, 1)
        return y_hat
    
    def config(self, name):
        HE, RE = nn.Hardswish(), nn.ReLU()
        # [kernel, exp size, in_channels, out_channels, SEBlock(SE), activation function(NL), stride(s)] 
        large = [
                [3, 16, 16, 16, False, RE, 1],
                [3, 64, 16, 24, False, RE, 2],
                [3, 72, 24, 24, False, RE, 1],
                [5, 72, 24, 40, True, RE, 2],
                [5, 120, 40, 40, True, RE, 1],
                [5, 120, 40, 40, True, RE, 1],
                [3, 240, 40, 80, False, HE, 2],
                [3, 200, 80, 80, False, HE, 1],
                [3, 184, 80, 80, False, HE, 1],
                [3, 184, 80, 80, False, HE, 1],
                [3, 480, 80, 112, True, HE, 1],
                [3, 672, 112, 112, True, HE, 1],
                [5, 672, 112, 160, True, HE, 2],
                [5, 960, 160, 160, True, HE, 1],
                [5, 960, 160, 160, True, HE, 1]
        ]

        small = [
                [3, 16, 16, 16, True, RE, 2],
                [3, 72, 16, 24, False, RE, 2],
                [3, 88, 24, 24, False, RE, 1],
                [5, 96, 24, 40, True, HE, 2],
                [5, 240, 40, 40, True, HE, 1],
                [5, 240, 40, 40, True, HE, 1],
                [5, 120, 40, 48, True, HE, 1],
                [5, 144, 48, 48, True, HE, 1],
                [5, 288, 48, 96, True, HE, 2],
                [5, 576, 96, 96, True, HE, 1],
                [5, 576, 96, 96, True, HE, 1]
        ]

        if name == "large": return large
        if name == "small": return small

       
if MODELTYPE=="ResNet18":
    print("Running ResNet")
    net_glob = ResNet18(BasicBlock, RESNETTYPE, len(IMG_TYPE)) #7 is my numbr of classes
elif MODELTYPE=="GoogleNet":
    print("Running GoogleNet")
    net_glob =GoogLeNetClient(NUM_CHANNELS,len(IMG_TYPE),conv_block,Inception_block)
if MODELTYPE=="MobileNet":
    net_glob = MobileNetV3("large", NUM_CHANNELS,len(IMG_TYPE),MobileNetV3Block,BNeck)
    
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob = nn.DataParallel(net_glob)   # to use the multiple GPUs 

net_glob.to(device)
print(net_glob)        


#=============================================================================
#                    ML Training and Testing
#============================================================================= 

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc

#==========================================================================================================================     
def train(model, device, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    ell = len(iterator)
    for (x, y) in iterator:
        
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad() # initialize gradients to zero
        
        # ------------- Forward propagation ----------
        fx = model(x)
        loss = criterion(fx, y)
        acc = calculate_accuracy (fx , y)
        
        # -------- Backward propagation -----------
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / ell, epoch_acc / ell
        
def evaluate(model, device, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    ell = len(iterator)
    
    with torch.no_grad():
        for (x,y) in iterator:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            
            fx = model(x)       
            loss = criterion(fx, y)
            acc = calculate_accuracy (fx , y)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss/ell, epoch_acc/ell
 

# =======================================================================================
epochs = EPOCHS
LEARNING_RATE = LR
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net_glob.parameters(), lr = LEARNING_RATE)

loss_train_collect = []
loss_test_collect = []
acc_train_collect = []
acc_test_collect = []
        
start_time = time.time()    
for epoch in range(epochs):
    train_loss, train_acc = train(net_glob, device, train_iterator, optimizer, criterion)
    #print(f'Train completed - {epoch} Epoch")
    test_loss, test_acc = evaluate(net_glob, device, test_iterator, criterion)
    #print(f'Test completed - {epoch} Epoch")
    
    loss_train_collect.append(train_loss)
    loss_test_collect.append(test_loss)
    acc_train_collect.append(train_acc)
    acc_test_collect.append(test_acc)
    
    
    prRed(f'Train => Epoch: {epoch} \t Acc: {train_acc*100:05.2f}% \t Loss: {train_loss:.3f}')
    prGreen(f'Test =>               \t Acc: {test_acc*100:05.2f}% \t Loss: {test_loss:.3f}')
    

  
elapsed = (time.time() - start_time)/60
print(f'Total Training Time: {elapsed:.2f} min')

#===================================================================================     

print("Training and Evaluation completed!")    

#===============================================================================
# Save output data to .excel file (we use for comparision plots)
round_process = [i for i in range(1, len(acc_train_collect)+1)]
df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect, 'acc_test':acc_test_collect, "Loss":loss_test_collect})     
file_name = program+".xlsx"    
df.to_excel(file_name, sheet_name= "v1_test", index = False)     

#=============================================================================
#                         Program Completed
#============================================================================= 







    

