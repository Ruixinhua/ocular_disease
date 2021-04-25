import logging
import os
import time

import pandas as pd
import torch

from thop import profile, clever_format
import models as module_arch
import utils.data_loader as module_data
import utils.loss_helper as module_loss
import utils.metric as module_metric
from run_model import run
from utils.parse_config import ConfigParser
from utils.tools import prepare_device
# fix random seeds for reproducibility
from utils.trainer import Trainer


def trainer_hyper_tuning():
    # define the hyper-parameters here
    image_sizes = [128, 256, 512]
    learning_rates = [0.01, 0.001, 0.0001]
    batch_sizes = [16, 32, 64]
    # define default training option here
    option = {
        "name": "googlenet",
        "arch;args;pretrained_model": "googlenet",  # pretrained model name
        "arch;args;freeze_param": 1,  # freeze the parameters of pretrained model
        "data_loader;args;img_size": 512,
        "data_loader;args;batch_size": 32,
        "optimizer;args;lr": 0.001
    }
    for image_size in image_sizes:
        option.update({"data_loader;args;img_size": image_size})
        run_id = f"{image_size}_{option['data_loader;args;batch_size']}_{option['optimizer;args;lr']}"
        # close old logging process
        logging.shutdown()
        run("pretrained_model.json", option, run_id, False)
    # I choose image_size=256, because the time consuming is much lower then image_size=512
    option.update({"data_loader;args;img_size": 256})
    for bs in batch_sizes:
        option.update({"data_loader;args;batch_size": bs})
        run_id = f"{option['data_loader;args;img_size']}_{bs}_{option['optimizer;args;lr']}"
        logging.shutdown()
        run("pretrained_model.json", option, run_id)
    # update with best batch_size: a larger batch size performs better
    option.update({"data_loader;args;batch_size": 64})
    for lr in learning_rates:
        option.update({"optimizer;args;lr": lr})
        run_id = f"{option['data_loader;args;img_size']}_{option['data_loader;args;batch_size']}_{lr}"
        logging.shutdown()
        run("pretrained_model.json", option, run_id)


def model_hyper_tuning():
    # define best suitable training option here
    option = {
        "arch;args;pretrained_model": "googlenet",  # pretrained model name
        "arch;args;freeze_param": 1,  # freeze the parameters of pretrained model
        "data_loader;args;img_size": 256,
        "data_loader;args;batch_size": 32,
        "optimizer;args;lr": 0.0001
    }
    resnet_models = ["resnet34", "resnet50", "resnet101", "resnet152"]
    vgg_models = ["vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]
    densnet_models = ["densenet121", "densenet169", "densenet201", "densenet161"]
    all_models = resnet_models + vgg_models + densnet_models
    for model_name in all_models:
        print(f"Run model: {model_name}")
        option.update({"name": model_name, "arch;args;pretrained_model": model_name})
        logging.shutdown()
        run("pretrained_model.json", option, model_name)


def run_test(conf, test_model):
    # setup data_loader instances
    data_loader = conf.init_obj("data_loader", module_data)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(conf["n_gpu"])
    test_model = test_model.to(device)
    if len(device_ids) > 1:
        test_model = torch.nn.DataParallel(test_model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, conf["loss"])
    metrics = [getattr(module_metric, met) for met in conf["metrics"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, test_model.parameters())
    optimizer = conf.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = conf.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    # define trainer of model
    test_trainer = Trainer(test_model, criterion, metrics, optimizer,
                      config=conf,
                      device=device,
                      data_loader=data_loader.train_loader,
                      valid_data_loader=data_loader.valid_loader,
                      lr_scheduler=lr_scheduler)
    return test_trainer._valid_epoch(0)


def evaluate():
    # define evaluation result path
    result_path = "saved/result"
    os.makedirs(result_path, exist_ok=True)
    evaluation_df = pd.DataFrame(
        columns=["model", "val_auc", "val_acc", "test_auc", "test_acc", "hyper_parameters", "parameters", "FLOPS",
                 "Time Consuming"])
    # load the statistic information
    for path in os.scandir("saved/models/googlenet"):
        hyper_parameters = path.name
        ckpt_path = os.path.join(path.path, "model_best.pth")
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cuda'))
        log_dic = checkpoint["log"]
        checkpoint["config"]["name"] = "test"
        config = ConfigParser(checkpoint["config"],
                              modification={"data_loader;args;valid_df_path": "dataset/test_df.csv"})
        model = config.init_obj("arch", module_arch)
        model.load_state_dict(checkpoint["state_dict"])
        start_time = time.time()
        test_log = run_test(config, model)
        time_consume = round(time.time() - start_time, 2)

        img_size, bs, lr = hyper_parameters.split("_")
        flops, params = profile(model, inputs=(
        torch.randn(int(bs), 3, int(img_size), int(img_size)).to(torch.device("cuda")),))
        flops, params = clever_format([flops, params], "%.3f")
        line = pd.Series({
            "model": "GoogLeNet", "val_auc": log_dic["val_auc"], "val_acc": log_dic["val_accuracy"],
            "hyper_parameters": hyper_parameters, "parameters": params, "FLOPS": flops,
            "test_auc": test_log["auc"], "test_acc": test_log["accuracy"], "Time Consuming": f"{time_consume} s"
        })
        evaluation_df = evaluation_df.append(line, ignore_index=True)
        evaluation_df.to_csv(os.path.join(result_path, "hyper_tuning.csv"), index=False)

    resnet_models = ["resnet34", "resnet50", "resnet101", "resnet152"]
    vgg_models = ["vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]
    densnet_models = ["densenet121", "densenet169", "densenet201", "densenet161"]
    all_models = resnet_models + vgg_models + densnet_models
    pretrained_compare_df = pd.DataFrame(
        columns=["model", "val_auc", "val_acc", "test_auc", "test_acc", "parameters", "FLOPS", "Time Consuming"])
    for model_name in all_models:
        log_dir = os.path.join("saved/models", model_name)
        ckpt_path = os.path.join(log_dir, os.listdir(log_dir)[-1], "model_best.pth")
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        log_dic = checkpoint["log"]

        checkpoint["config"]["name"] = "test"
        config = ConfigParser(checkpoint["config"],
                              modification={"data_loader;args;valid_df_path": "dataset/test_df.csv"})
        model = config.init_obj("arch", module_arch)
        model.load_state_dict(checkpoint["state_dict"])
        start_time = time.time()
        test_log = run_test(config, model)
        time_consume = round(time.time() - start_time, 2)

        img_size, bs = 256, 32
        model = model.cpu()
        flops, params = profile(model, inputs=(torch.randn(int(bs), 3, int(img_size), int(img_size)),))
        flops, params = clever_format([flops, params], "%.3f")
        line = pd.Series({
            "model": model_name, "val_auc": log_dic["val_auc"], "val_acc": log_dic["val_accuracy"],
            "parameters": params, "FLOPS": flops, "test_auc": test_log["auc"], "test_acc": test_log["accuracy"],
            "Time Consuming": f"{time_consume}s"
        })
        torch.cuda.empty_cache()
        pretrained_compare_df = pretrained_compare_df.append(line, ignore_index=True)
    pretrained_compare_df.to_csv(os.path.join(result_path, "pretrained_model_compare.csv"), index=False)


def experiment():
    trainer_hyper_tuning()
    model_hyper_tuning()
    evaluate()


if __name__ == "__main__":
    experiment()
