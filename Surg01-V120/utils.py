'''
Training utilities. 
'''

import torch
from torch import nn, optim
import wandb

from sklearn.metrics import accuracy_score, f1_score, classification_report

import os
import time
from datetime import datetime
import pytz
from statistics import mean

from data import RandomLoader, SequentialLoader


class Metrics:
    ''' Metrics Recorder '''

    class MetricRecorder:
        '''
        Batch metric recorder

        Keep track of the metric retrieved in each epoch, updated and averaged with each batch
        '''

        def __init__(self):
            super().__init__()
            # metric of each batch in one epoch
            self.batch = []
            self.hist = []

        def record(self, item):
            ''' Record the metric for a batch '''
            self.batch.append(item)

        def average(self):
            ''' Get average of one batch '''
            avg = mean(self.batch)
            # record it in history
            self.hist.append(avg)
            # clear the batch data
            self.batch.clear()
            return avg

        def history(self):
            ''' Get all recorded history '''
            return self.hist


    def __init__(self):
        # record the gourd truth and predicted labels
        # initialize a recorder for each metric
        self.metrics = {}


    def record(self, **metric_values):
        '''
        Record the metrics

        Args
        ----
            **metric_values: metric_name=metric_value
        '''

        for name, value in metric_values.items():
            # construct a new recorder if not constructed yet
            self.metrics.setdefault(name, self.MetricRecorder()).record(value)


    def average(self, *names):
        ''' Get average of all metrics in one epoch '''
        names = names or self.metrics.keys()
        return {name: self.metrics[name].average() for name in names}


    def history(self, *names):
        ''' Get history of metrics '''
        names = names or self.metrics.keys()
        return {name: self.metrics[name].history() for name in names}


class ResnetTrainer:
    ''' ResNet Trainer '''

    def __init__(self, model, device, optimizer=optim.Adam, loss=nn.CrossEntropyLoss):
        '''
        Args
        ----
            model: a pytorch model to train
            device: the device to train on
            optimizer: type of optimizer (default Adam)
            loss: type of loss function (default CrossEntropy)
        '''

        self.model = model
        self.device = device
        self.optimizer = optimizer
        # use of RandomLoader is manditory when training ResNet
        self.train_loader = RandomLoader
        self.valid_loader = SequentialLoader
        self.loss = loss


    def train(
        self, train_set, valid_set, num_epochs, batch_size, learning_rate, run_name, main_metric='valid_f1_weighted', 
        early_stopping_metric='valid_loss', early_stopping_tolerance=3, min_delta=0.0, 
        num_workers=0, prefetch_factor=2, pin_memory=True
    ):
        '''
        Args
        ----
            train_set: training set
            valid_set: validation set

            num_epochs: number of epochs to train
            batch_size: training batch size (for validation, batch size is always 1)
            learning_rate: learning_rate for the optimizer

            run_name: W&B name of the current run

            main_metric: main metric to save the best model (default valid_f1_weighted)
            early_stoppping_tolerance: stop the training if main_metric drops continuously (default 3 epochs)

            num_workers: number of workers for dataloader (default 0)
            pin_memory: whether use memory pinning for the dataloader (default True)
        '''

        # W&B init
        log_name = run_name + '-' + datetime.now(pytz.timezone("America/New_York")).strftime('%y%m%d%H%M%S')
        run = wandb.init(
            project='surgery-hernia', # name of the W&B project
            entity='eezklab',         # name of the orgnization
            name=log_name,            # name of the run
        )

        print('Running', run_name)
        print('Datasets: num_train = {}, num_validation = {}'.format(
            len(train_set), len(valid_set)
        ))
        print('Main metric:', main_metric)

        # dataloaders
        train_loader = self.train_loader(
            train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
        )
        validation_loader = self.valid_loader(
            valid_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
        )

        device = self.device
        model = self.model.to(device)
        optimizer = self.optimizer(model.parameters(), lr=learning_rate)
        loss = self.loss()

        # metrics
        metrics = Metrics()
        early_stopping_cnt = 0
        # path to save model state dicts
        model_path = os.path.join('./model', run_name+'.pt')

        for e in range(num_epochs):
            # train
            model.train()
            gt_labels, pr_labels = [], []

            for i, data in enumerate(train_loader):
                if i % (len(train_loader) // 100) == 0:
                    print('Epoch {}/{}: training {:.2f}% (batch {}/{})'.format(
                        e + 1, num_epochs, i / len(train_loader) * 100, i + 1, len(train_loader)
                    ), end='\r')

                X, y = data['feature'].to(device), data['label'].to(device)
                with torch.enable_grad():
                    optimizer.zero_grad()
                    preds = model(X)
                    l = loss(preds, y)
                    l.backward()
                    optimizer.step()

                metrics.record(train_loss=l.item())
                gt_labels += y.tolist()
                pr_labels += torch.max(preds.data, 1)[1].tolist()

            # train metrics
            metrics.record(
                train_accuracy=accuracy_score(gt_labels, pr_labels), 
                train_f1_macro=f1_score(gt_labels, pr_labels, average='macro'), 
                train_f1_weighted=f1_score(gt_labels, pr_labels, average='weighted')
            )

            # validation
            model.eval()
            gt_labels, pr_labels = [], []

            for i, data in enumerate(validation_loader):
                if i % (len(validation_loader) // 100) == 0:
                    print('Epoch {}/{}: validating {:.2f}% (batch {}/{})'.format(
                        e + 1, num_epochs, i / len(validation_loader) * 100, i + 1, len(validation_loader)
                    ), end='\r')

                X, y = data['feature'].to(device), data['label'].to(device)
                with torch.no_grad():
                    preds = model(X)
                    l = loss(preds, y)

                metrics.record(valid_loss=l.item())
                gt_labels += y.tolist()
                pr_labels += torch.max(preds.data, 1)[1].tolist()

            # validation metrics
            metrics.record(
                valid_accuracy=accuracy_score(gt_labels, pr_labels), 
                valid_f1_macro=f1_score(gt_labels, pr_labels, average='macro'), 
                valid_f1_weighted=f1_score(gt_labels, pr_labels, average='weighted')
            )

            epoch_metric = metrics.average()

            # save the model with highest validation weighted f1-score
            if epoch_metric[main_metric] == max(metrics.history(main_metric)[main_metric]):
                torch.save(model.state_dict(), model_path)

            # add data to W&B table
            wandb.log(epoch_metric)

            print(
                'Epoch {}/{}:'.format(e + 1, num_epochs), 
                ', '.join(['{} {:.8f}'.format(name, value) for name, value in epoch_metric.items()])
            )

            # early stopping
            hist = metrics.history(early_stopping_metric)[early_stopping_metric]
            if len(hist) > 1 and hist[-2] - hist[-1] <= min_delta:
                early_stopping_cnt += 1
            else:
                early_stopping_cnt = 0
            if early_stopping_cnt >= early_stopping_tolerance:
                print('Training stopped')
                break

        return metrics.history()


class ResnetEvaluator:
    ''' ResNet Evaluator '''

    def __init__(self, model, device, names=None):
        '''
        Args
        ----
            model: a pytorch model to train
            device: the device to train on
            labels -> list: a list of label names to display in report
        '''

        self.model = model
        self.device = device
        self.test_loader = SequentialLoader
        self.names = names


    def evaluate(self, test_set, num_workers=0, prefetch_factor=2, pin_memory=True):
        '''
        Args
        ----
            test_set: testing set

            num_workers: number of workers for dataloader (default 0)
            pin_memory: whether use memory pinning for the dataloader (default True)
        '''

        print('Testing ResNet')
        print('Datasets: num_test =', len(test_set))

        test_loader = self.test_loader(
            test_set, batch_size=1, num_workers=num_workers, pin_memory=pin_memory
        )

        device = self.device
        model = self.model.to(device)

        metrics = Metrics()
        gt_labels, pr_labels = [], []

        model.eval()

        with torch.no_grad():

            for i, data in enumerate(test_loader):
                if i % (len(test_loader) // 100) == 0:
                    print('Testing {:.2f}% (batch {}/{})'.format(
                        i / len(test_loader) * 100, i + 1, len(test_loader)
                    ), end='\r')

                start = time.time()
                X, y = data['feature'].to(device), data['label'].to(device)
                preds = model(X)
                elapsed = time.time() - start

                metrics.record(time=elapsed)
                gt_labels.append(y.item())
                pr_labels.append(torch.max(preds.data, 1)[1].item())

        metrics.record(
            accuracy=accuracy_score(gt_labels, pr_labels), 
            f1_macro=f1_score(gt_labels, pr_labels, average='macro'), 
            f1_weighted=f1_score(gt_labels, pr_labels, average='weighted')
        )

        return metrics.average(), classification_report(gt_labels, pr_labels, target_names=self.names)
