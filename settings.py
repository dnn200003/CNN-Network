RESNETTYPE=[2,2,2,2] #34 up [3,4,6,3], #18 [2,2,2,2]
NUM_USERS = 5
EPOCHS = 1
LOCAL_EP=1
FRAC = 1
LR = 0.0001
#intelnet, IP102_FC_EC, IP102_FC, IP102_EC
TRAINING_SORCE="IP102_EC"
EPOCHSPLIT=3
#CL: 1 (Make client 1 and local_EP 1), SL: 2, DSL: 3 
SPLITTYPE=3
if SPLITTYPE==1:
    NUM_USERS=1
    LOCAL_EP=1
    
#ResNet18, ResNet34, ResNet50, GoogleNet, MobileNet
MODELTYPE="MobileNet"
clientlayers=2
NOISE=False


