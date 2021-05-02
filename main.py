import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from data_pytorch import Data
from rotnet import RotNet
import time
import shutil
import yaml

parser = argparse.ArgumentParser(description='Configuration details for training/testing rotation net')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--train', action='store_true')
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--image', type=str)
parser.add_argument('--model_number', type=str, required=True)

args = parser.parse_args()

#config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

"""
epochs = 10

for epoch in range(epochs):
    total_loss = 0
    
    #for batch in train_dataloader:
    for i, data in enumerate(train_dataloader, 0):
        #print(i)
        #print(data)
        #X_batch, y_batch = batch[0], batch[1] #.view(-1, 784)
        #print(y_batch)
        #print(X_batch, y_batch)
        #print(X_batch, y_batch)
        # THESE 5 LINES ARE IMPORTANT, UNDERSTAND WHAT IS HAPPENING HERE
        optimizer.zero_grad()
        predicted_batch = model(data) 
        #print(predicted_batch)
        loss = criterion(predicted_batch, y_batch)
        #print(loss)
        loss.backward()
        optimizer.step()
        total_loss += loss
        
        
       
    print("Epoch {0}: {1}".format(epoch, total_loss))
    #if epoch%5 == 0 and epoch != 0:
    #    test_batch = next(iter(test_dataloader))
    #    X_test, y_test = test_batch[0].view(-1, 784), test_batch[1]
    #    predicted = model(X_test)
    #    test_acc = torch.sum(y_test == torch.argmax(predicted, dim=1), dtype=torch.double) / len(y_test)
    #    print("\tTest Accuracy {0}".format(test_acc))
"""

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    for i, (input, target) in enumerate(train_loader):
        optimizer.zero_grad()
        predicted_batch = model(input)
        loss = criterion(predicted_batch, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss

def validate(val_loader, model, criterion):
    model.eval()
    for i, (input, target) in enumerate(val_loader):
        optimizer.zero_grad()
        outputs = model(input)
        #TODO: implement the validation. Remember this is validation and not training
        #so some things will be different.

def save_checkpoint(state, best_one, filename='rotationnetcheckpoint.pth.tar', filename2='rotationnetmodelbest.pth.tar'):
    torch.save(state, filename)
    #best_one stores whether your current checkpoint is better than the previous checkpoint
    if best_one:
        shutil.copyfile(filename, filename2)

def main():
    n_epochs = 10 #config["num_epochs"]
    model = ResNet34()
    criterion = torch.optim.MSELoss() #what is your loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum = 0.9) #temp

    train_dataset = loadData()
    train_loader =  DataLoader(newX, batch_size=256, shuffle=True)
    #val_loader = #how will you get your dataset
    #val_loader = #how will you use pytorch's function to build a dataloader

    for epoch in range(n_epochs):
            train(train_loader, model, criterion, optimizer, epoch)
         #TODO: make your loop which trains and validates. Use the train() func

         #TODO: Save your checkpoint





if __name__ == "__main__":
    main()
