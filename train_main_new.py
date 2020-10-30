import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from models import *
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from data_loader_new import TrainSet
from data_loader_new import ValSet
import PIL
from PIL import Image
import numpy as np
writer = SummaryWriter('runs/Ass1')


#### this function is used to handle batches contain images of different sizes ####
def my_collate(batch):
    data =[item[0] for item in batch]
    target = [item[1] for item in batch]
    max1 = max([x.shape[1] for x in data])
    max2 = max([x.shape[2] for x in data])
    max3 = max([x.shape[1] for x in target])
    max4 = max([x.shape[2] for x in target])
    data = [torch.nn.ConstantPad2d((max2 - x.shape[2],0,max1 - x.shape[1],0),0)(x) for x  in data]
    target = [torch.nn.ConstantPad2d((max4 - x.shape[2],0, max3 - x.shape[1],0),0)(x) for x  in target]
    
    data = torch.stack(data)
    target = torch.stack(target)
    data = data.type(torch.FloatTensor)
    target = target.type(torch.FloatTensor)

    return data,target

########################## Evaluation Metric  ############################################
def Jacc(logits,targets):
    true = targets.long()
    num_classes = 3
    eps = 1e-10
    true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc = (intersection / (union + eps)).mean()
    return jacc


def DiceScore(logits, targets):
    targets = targets.long()
    num_classes = 3
    eps = 1e-10
    true_1_hot = torch.eye(num_classes)[targets.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, targets.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_score = (2. * intersection / (cardinality + eps)).mean()
    return dice_score
#---------------------------------------
def Inv_DiceScore(logits, targets):
    smooth = 1.
    logits = F.sigmoid(logits)
    iflat = 1-logits.view(-1)
    tflat = 1-targets.view(-1)
    intersection = (iflat * tflat).sum()
    return  ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))

################################# Loss Function ############################
def Loss_fun(logits, targets):
    
    ## Dice Loss ##
    targets = targets.long()
    num_classes = 3
    eps= 1e-10
    true_1_hot = torch.eye(num_classes)[targets.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, targets.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_score = (2. * intersection / (cardinality + eps)).mean()
    Dice_Loss = 1-dice_score
    ## Inverse Dice Loss 



    ## CE Loss ##
    iflat = logits.reshape(logits.shape[0],logits.shape[1],(logits.shape[2])*(logits.shape[3]))
    tflat = targets.reshape(logits.shape[0],(logits.shape[2])*(logits.shape[3]))
    tflat = tflat.long()
    if torch.cuda.is_available():
        class_weights = [0.1,0.45,0.45]
        class_weights = torch.FloatTensor(class_weights).cuda()
    CE_Loss = nn.CrossEntropyLoss()(iflat,tflat)
    ## Final Loss ##
    """
    cn = 0
    ct = 0
    t = 0.3
    Lambda = 0.75
    for i in tflat:
        ct = ct+1
        if i != 0:
            cn = cn+1
    if (cn/ct)> t:
        Final_loss = Lambda*Dice_Loss + (1-Lambda)*Inv_Dice_Loss +  CE_Loss  
    else:
        Final_loss = (1-Lambda)*Dice_Loss + (Lambda)*Inv_Dice_Loss +  CE_Loss 
    return Final_loss
    """
    return 0.75 * Dice_Loss+ CE_Loss
############################################################################



transformations_train = transforms.Compose([transforms.ToTensor()])
    
transformations_val = transforms.Compose([transforms.ToTensor()]) 

train_set = TrainSet(transforms = transformations_train)
val_set = ValSet(transforms = transformations_val)

batch_size = 8
epochs = 1000

print("preparing training data ...")
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers = 4,collate_fn=my_collate)
val_loader = DataLoader(val_set, batch_size=batch_size,shuffle=True,num_workers = 4,collate_fn=my_collate)
print("Done :)")
net = UNet(1,3)
net.init_weights(net)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)
epoch_pre = 0


def train():
    net = UNet(1,3)
    net.init_weights(net)
    epoch_pre = 0
    PATH = "model.pt"
    if os.path.isfile(PATH):
        checkpoint = torch.load(PATH)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_pre = checkpoint['epoch']
    cuda = torch.cuda.is_available()
    if cuda:
        net = net.cuda()
    min_val_loss = 100
    for epoch in range(epoch_pre+1, epochs+1):
        print('Welcome to epoch:',epoch)
        net.train()
        for i, (images, masks) in tqdm(enumerate(train_loader)):
            images = Variable(images)
            masks = Variable(masks)
            if cuda:
                images = images.cuda()
                masks = masks.cuda()
            optimizer.zero_grad()
            outputs = net(images)
            loss = Loss_fun(outputs,masks)
            writer.add_scalar('Train Loss',torch.mean(loss),epoch)
            loss.backward()
            optimizer.step()
            scheduler.step() 
        net.eval()
        for images, masks in tqdm(val_loader):
            images = Variable(images)
            masks = Variable(masks)
            if cuda:
                images = images.cuda()
                masks = masks.cuda()

            outputs = net(images)
            vloss = Loss_fun(outputs,masks)
            Dscore = DiceScore(outputs,masks)
            jacc = Jacc(outputs,masks)
            writer.add_scalar('Val Loss',torch.mean(vloss),epoch)
            writer.add_scalar('Dice Score',torch.mean(Dscore),epoch)
            writer.add_scalar('Jaccard',torch.mean(jacc),epoch)
        out_path = './results/'
        if not os.path.exists(out_path):
                os.makedirs(out_path)
        print(outputs[0].shape)
        output = outputs[0].data.cpu().squeeze(0).numpy()
        #output = np.transpose(output, (1,2,0))
        output = output
        output = output.astype(np.uint8)
        print(output.shape)
        out = np.argmax(output, axis=0)
        print(out)
        for i in range(0,out.shape[0]):
            for j in range(0,out.shape[1]):
                if out[i,j]==2:
                    out[i,j]=1.0
                elif out[i,j]==1:
                    out[i,j]=0.5
                else:
                    out[i,j]=0.0
                
        out =Image.fromarray((out * 255).astype(np.uint8))
        out.save(out_path+"img_"+str(epoch)+'.png')

        print("Epoch: {}, Train Loss: {}, Val Loss: {} , Val DScore: {}, Val Jacc: {}".format(epoch+1, torch.mean(loss), torch.mean(vloss), torch.mean(Dscore),torch.mean(jacc)))
        out_path = './epochs/'
        if not os.path.exists(out_path):
                os.makedirs(out_path)

        if torch.mean(vloss)< min_val_loss:
            min_val_loss = torch.mean(vloss)
            torch.save(net,'./epochs/model_'+str(epoch)+'.pth')

        # checkpoint
        EPOCH = epoch
        PATH = "model.pt"
        torch.save({'epoch': EPOCH,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, PATH)

if __name__ == "__main__":
    train()













