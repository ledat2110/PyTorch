import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import datetime

class Supervised(object):
    def __init__(self, model, optimizer, loss_fn):
        super().__init__()
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        print(f'Trainig on device {self.device}.')
        self.loader = {}
    
    def data_loader(self, dataset, batch_size, shuffle, mode):
        assert mode in ['train', 'val', 'test']
        
        self.loader[mode] = loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            
    def training_loop(self, n_epochs, model, optimizer, loss_fn, reg=0):
        assert self.loader['train'] is not None
        assert model is not None
        assert loss_fn is not None
        assert optimizer is not None
        
        n_batches = len(self.loader['train'])
        
        model = model.to(device=self.device)
        model.train()
        
        for epoch in range(1, n_epochs+1):
            loss_train = 0.0
            # Loops over dataset in the batches the data loader created
            for imgs, labels in self.loader['train']:
                imgs = imgs.to(device=self.device)
                labels = labels.to(device=self.device)
                
                # Feeds a batch through the model
                outputs = model(imgs)

                # Compute the loss
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss_fn(outputs, labels) + reg * l2_norm
                
                # Get rid of the gradients from the last round
                optimizer.zero_grad()
                
                # Compute the gradients of all parameters
                loss.backward()
                
                # Update the model
                optimizer.step()
                
                # Sum the losses over the epoch
                loss_train += loss.item()
            
            # Get the average loss per batch
            avg_loss = loss_train * 1.0 / n_batches
            
            if epoch == 1 or epoch % 10 == 0:
                print(f'{datetime.datetime.now()} Epoch {epoch}, Avg trainig loss {avg_loss}')
        
        return model
    
    def validate(self, model, mode):
        assert mode in ['train', 'val', 'test']
        assert self.loader[mode] is not None
        
        loader = self.loader[mode]
        correct = 0
        total = 0
        
        model.to(device=self.device)
        model.eval()
        
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=self.device)
                labels = labels.to(device=self.device)
                
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                
                total += labels.shape[0]
                correct += int((predicted==labels).sum())
        
        accuracy = correct * 1.0 / total
        print(f'Accuracy {mode}: {accuracy:.2f}')
        return accuracy

    
