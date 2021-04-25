import numpy as np
import torch
from torchvision.utils import make_grid

from base import BaseTrainer
from tqdm import tqdm

from utils.tools import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, add_histogram=False):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config.config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.add_histogram = add_histogram
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        bar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
        # collect all the outputs and targets for evaluation
        outputs, targets = [], []
        for batch_idx, (data, target) in bar:
            # load data and target to device
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            # get output from model
            if self.config["name"] == "VAE":
                output, z, mu, log_var = self.model(data)
                loss = self.criterion(output, data, mu, log_var)
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # extend outputs and targets
            outputs.extend(output.detach().cpu().tolist())
            targets.extend(target.detach().cpu().tolist())
            # writer training information to
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            if batch_idx % self.log_step == 0:
                bar.set_description(f"Train Epoch: {epoch} Loss: {round(loss.item(), 4)}")
                if self.config["name"] == "VAE":
                    # add original and reconstructed images to tensorboard
                    self.writer.add_image('train_origin', make_grid(data.cpu(), nrow=8, normalize=True))
                    self.writer.add_image('train_recons', make_grid(output.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        # do evaluation on training dataset with the metric functions defined
        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(outputs, targets))
        log = self.train_metrics.result()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: round(v, 4) for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        outputs, targets = [], []
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(self.valid_data_loader), total=len(self.valid_data_loader)):
                data, target = data.to(self.device), target.to(self.device)
                if self.config["name"] == "VAE":
                    output, z, mu, log_var = self.model(data)
                    loss = self.criterion(output, data, mu, log_var)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                # extend outputs and targets
                outputs.extend(output.cpu().tolist())
                targets.extend(target.cpu().tolist())
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', round(loss.item(), 4))
            # add original and reconstructed images to tensorboard
            if self.config["name"] == "VAE":
                self.writer.add_image('valid_origin', make_grid(data.cpu(), nrow=8, normalize=True))
                self.writer.add_image('valid_recons', make_grid(output.cpu(), nrow=8, normalize=True))
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(outputs, targets))
        from sklearn.metrics import confusion_matrix
        pred = np.argmax(outputs, axis=1)
        self.matrix = confusion_matrix(targets, pred)
        # add histogram of model parameters to the tensorboard
        if self.add_histogram:
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()
