'''
Train SVRCNet.
'''
from unittest import result
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, SequentialSampler, BatchSampler
from models.SVRCNet.svrc import SVRC
from utils.mydataset import SVRCDataset

class ResnetTrainVal(object):
    '''
    Class for training Resnet.
    '''
    def __init__(self, model, device, EPOCH, BATCH_SIZE, LR) -> None:
        '''
        ResNet training parameters initialization.
        '''
        self.model = model
        self.device = device
        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, labels, features, validation:tuple, transform, path) -> dict: #, val_ratio=0.7):
        '''
        Training function for ResNet.
        Input args:
            labels, features: Traning data path
            validation: Validation data path
            transform: Data transformation chosen
            path: Path for saving optimal model weights
        Outputs: 
            Dictionary of training history
        '''
        print('Training ResNet: ')

        # TRAIN_SIZE = int(val_ratio * len(features))
        # TEST_SIZE = len(features) - TRAIN_SIZE

        train = SVRCDataset(features, labels, transform['train'])
        test = SVRCDataset(validation[0], validation[1], transform['valid'])
        #train, test = random_split(dataset, [TRAIN_SIZE, TEST_SIZE])
        print('length of train:', len(train))
        print('length of validation', len(test))

        train_loader = DataLoader(train, self.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test, self.BATCH_SIZE, shuffle=True)

        self.model.pretrain = True

        hist_train_loss = []
        hist_train_acc = []
        hist_valid_loss = []
        hist_valid_acc = []

        for epoch in range(self.EPOCH):
            self.model.train()

            train_loss = 0.0
            train_acc = 0.0

            for i, data in enumerate(train_loader):
            
                features = data['feature'].float()
                labels = data['label']
                features, labels = features.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(features)
                loss = self.criterion(predictions, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                preds = torch.max(predictions.data, 1)[1]
                train_acc += (preds==labels).sum().item()
            
            train_loss /= len(train)
            train_acc /= len(train)
            torch.save(self.model.state_dict(),path)
            hist_train_loss.append(train_loss)
            hist_train_acc.append(train_acc)


            valid_loss = 0.0
            valid_acc = 0.0
            total = 0
            self.model.eval()
            for i, data in enumerate(test_loader):
                features = data['feature']
                labels = data['label']

                features, labels = features.to(self.device), labels.to(self.device)
                predictions = self.model(features)
                loss = self.criterion(predictions,labels)
                valid_loss += loss.item()

                preds = torch.max(predictions.data, 1)[1]
                valid_acc += (preds==labels).sum().item()
                total += features.size(0)

            valid_loss /= len(test)
            valid_acc /= len(test)
            hist_valid_loss.append(valid_loss)
            hist_valid_acc.append(valid_acc)

            print(
                f'Epoch {epoch+1} Training Loss: {train_loss} Train_acc: {train_acc}'
                f' || Validation Loss: {valid_loss} Valid_acc: {valid_acc}'
            )

        return {
            'train_loss': hist_train_loss,
            'train_acc': hist_train_acc,
            'valid_loss': hist_valid_loss,
            'valid_acc': hist_valid_acc
        }


class LstmTrainVal(object):
    '''
    Class for training SVRCNet.
    '''
    def __init__(self, model,device, EPOCH, BATCH_SIZE, LR) -> None:
        '''
        LSTM training parameters initialization.
        '''
        self.model = model
        self.device = device
        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, labels, features, validation:tuple, transform, path, eval_intval=3):
        '''
        Training function for SVRCNet.
        Inputs:
            Similar to the trainer for LSTM.
            eval_intval: Step for validation (One validation for each eval_intval epochs)
        Outputs:
            Dictionary of training history
        '''
        print('Training ResNet: ')

        dataset = SVRCDataset(features, labels, transform['train'])
        data_loader = DataLoader(
            dataset, batch_sampler=BatchSampler(
                SequentialSampler(dataset), 
                self.BATCH_SIZE, 
                drop_last=True
            )
        )
        valid_set = SVRCDataset(validation[0], validation[1], transform['valid'])
        valid_loader = DataLoader(
            valid_set, batch_sampler=BatchSampler(
                SequentialSampler(valid_set), 
                self.BATCH_SIZE, 
                drop_last=True
            )
        )
        print('length of train:', len(dataset))
        print('length of validation', len(valid_set))

        self.model.pretrain = False

        hist_train_loss = []
        hist_train_acc = []
        hist_valid_loss = []
        hist_valid_acc = []

        for epoch in range(self.EPOCH):
            self.model.lstm.train()
            self.model.full.train()

            train_loss = 0.0
            train_acc = 0.0

            for i, data in enumerate(data_loader):
                features  = data['feature'].float()
                labels = data['label']
                features, labels = features.to(self.device), labels.to(self.device)
                predictions = self.model(features)
                loss = self.criterion(predictions, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                preds = torch.max(predictions.data, 1)[1]
                train_acc += (preds==labels).sum().item()

            torch.save(self.model.state_dict(),path)


            train_loss /= len(dataset)
            train_acc /= len(dataset)
            hist_train_loss.append(train_loss)
            hist_train_acc.append(train_acc)

            valid_loss = 0.0
            valid_acc = 0.0

            if (epoch + 1) % eval_intval == 0:
                total = 0
                self.model.eval()
                for i, data in enumerate(valid_loader):
                    features = data['feature']
                    labels = data['label']

                    features, labels = features.to(self.device), labels.to(self.device)
                    predictions = self.model(features)
                    loss = self.criterion(predictions,labels)
                    valid_loss += loss.item()

                    preds = torch.max(predictions.data, 1)[1]
                    valid_acc += (preds==labels).sum().item()
                    total += features.size(0)

                valid_loss /= len(valid_set)
                valid_acc /= len(valid_set)
                hist_valid_loss.append(valid_loss)
                hist_valid_acc.append(valid_acc)

            print(
                f'Epoch {epoch+1} Training Loss: {train_loss} Train_acc: {train_acc}'
                f' || Validation Loss: {valid_loss} Valid_acc: {valid_acc}'
            )

        return {
            'train_loss': hist_train_loss,
            'train_acc': hist_train_acc,
            'valid_loss': hist_valid_loss,
            'valid_acc': hist_valid_acc
        }


class Evaluator:
    '''
    Evaluate dataset without labels.
    '''
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, images, transform, pretrain):
        '''
        Predict input data.
        Input args:
            images: Path for evaluating data.
            transform: Chosen data transformation method
            pretrain: ResNet or SVRCNet mode
        '''
        dataset = SVRCDataset(images, None, transform)
        loader = DataLoader(dataset, batch_size=3, drop_last=True)
        preds = []
        self.model.pretrain = pretrain
        self.model.eval()
        for i,data in enumerate(loader):
            feature = data['feature'].float().to(self.device)
            pred = torch.max(self.model(feature).data, 1)[1]
            preds.append(pred)
        return sum(list(map(torch.Tensor.tolist, preds)), [])

    def eval(self, preds, labels):
        ''' 
        Evaluate the predictions with ground truth labels. 
        There should not be batched in preds and labels
        '''
        acc = sum([p == l for p,l in zip(preds, labels)]) / len(labels)
        print('Accuracy: {}'.format(acc))
        return acc