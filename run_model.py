import logging

import torch
import torch.backends.cudnn as cudnn
import utils.loss_helper as module_loss

import models as module_arch
import utils.data_loader as module_data
import utils.metric as module_metric
from utils.parse_config import ConfigParser
from utils.tools import prepare_device, read_json
# fix random seeds for reproducibility
from utils.trainer import Trainer


def run(cfg_name, modification=None, run_id=None, log_model=False):
    seed = 42
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    cfg = read_json(cfg_name)
    config = ConfigParser(cfg, modification=modification, run_id=run_id)
    # create logger
    logger = config.get_logger("train")

    # setup data_loader instances
    data_loader = config.init_obj("data_loader", module_data)

    # build model architecture, then print to console
    model = config.init_obj("arch", module_arch)
    if log_model:
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
    return trainer


if __name__ == "__main__":
    option = {
        "arch;args;pretrained_model": "googlenet",  # pretrained model name
        "arch;args;freeze_param": 1,  # freeze the parameters of pretrained model
        "data_loader;args;img_size": 512,
        "data_loader;args;batch_size": 32,
        "optimizer;args;lr": 0.001
    }
    run("pretrained_model.json", option)
