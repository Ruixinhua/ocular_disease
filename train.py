import argparse
import collections

import torch
import torch.backends.cudnn as cudnn
import utils.loss_helper as module_loss

import models as module_arch
import utils.data_loader as module_data
import utils.metric as module_metric
from utils.parse_config import ConfigParser
from utils.tools import prepare_device
# fix random seeds for reproducibility
from utils.trainer import Trainer

seed = 42
torch.manual_seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    data_loader = config.init_obj("data_loader", module_data)

    # build model architecture, then print to console
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    # define trainer of model
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader.train_loader,
                      valid_data_loader=data_loader.valid_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="ODIR training process")
    args.add_argument("-c", "--config", default="pretrained_model.json", type=str,
                      help="config file path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str,
                      help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str,
                      help="indices of GPUs to enable (default: all)")

    # add some model configuration here
    args.add_argument("-m", "--pretrained_model", default=None, type=str,
                      help="specify the pretrained model (such as: resnet50)")
    args.add_argument("-f", "--freeze_param", default=None, type=int,
                      help="whether freeze the pretrained model (such as: 1, means freeze parameters)")
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size")
    ]
    main(ConfigParser.from_args(args, options))
