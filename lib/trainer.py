import os
import cv2
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import models
import lib.utils
from lib.utils.logs import MyLog
from lib.utils.distributed_utils import is_main_process
from lib.models.metric import Evaluator
from dataset.dataset_tools import our_class
import dataset
from tqdm import tqdm

# from dataset import ImageRawDataset, PartialCompEvalDataset, PartialCompDataset
import pdb
import cProfile


class Trainer(object):

    def __init__(self, args):
        self.args = args
        self.flogger = args.flogger
        if is_main_process():
            # logger
            if args.trainer["tensorboard"]:
                try:
                    from tensorboardX import SummaryWriter
                except:
                    raise Exception('Please switch off "tensorboard" ' "in your config file if you do not " "want to use it, otherwise install it.")
                self.tb_logger = SummaryWriter("{}/events".format(args.exp_path))
            else:
                self.tb_logger = None
            self.logger = MyLog()

        self.start_epoch = 1
        self.epoch = 0
        # create model
        self.model: models.PartialCompletionMask = models.__dict__[args.model["algo"]](args)
        # optionally resume from a checkpoint

        if args.resume:
            self.model.load_pretrain(args.load_pretrain)
            self.start_epoch = args.load_epoch

        # lr scheduler & datasets
        trainval_class = dataset.__dict__[args.data["trainval_dataset"]]  # 预处理
        # _dataset = trainval_class(args.data, "train")
        # train_num = int(len(_dataset)*0.8)
        # train_dataset = Subset(_dataset, range(0, train_num))
        # val_dataset = Subset(_dataset, range(train_num, len(_dataset)))
        # train_dataset = Subset(train_dataset, range(0, 1000))
        # val_dataset = Subset(val_dataset, range(0, 1000))


        if not args.evaluate:  # train
            train_dataset = trainval_class(args.data, "train")
            val_dataset = trainval_class(args.data, "val")  # 加载测试集
            self.lr_scheduler = lr_scheduler.MultiStepLR(self.model.optim, milestones=args.trainer["milestones"], gamma=args.trainer["gamma"], last_epoch = -1)
            if args.resume:
                for i in range(self.start_epoch):
                    self.lr_scheduler.step()
            if args.distributed:
                train_sampler = DistributedSampler(train_dataset)
                val_sampler = DistributedSampler(val_dataset, shuffle=False)
                self.model = nn.DataParallel(self.model)
            else:
                train_sampler = SequentialSampler(train_dataset)
                val_sampler = SequentialSampler(val_dataset)
            self.train_loader = DataLoader(train_dataset, batch_size=args.data["batch_size"], num_workers=args.data["workers"], pin_memory=True, sampler=train_sampler)
            self.val_loader = DataLoader(val_dataset, batch_size=args.data["batch_size_val"], num_workers=args.data["workers"], pin_memory=True, sampler=val_sampler)
            print(f"train_dataset:{len(self.train_loader)}")
            print(f"val_dataset:{len(self.val_loader)}")
            self.curr_step = self.start_epoch * len(self.train_loader)
        else:  # test
            self.model.load_pretrain(args.load_pretrain)
            test_dataset = trainval_class(args.data, "val")
            if args.distributed:
                test_sampler = DistributedSampler(test_dataset, shuffle=False)
                self.model = nn.DataParallel(self.model)
            else:
                test_sampler = SequentialSampler(test_dataset)
            self.val_loader = DataLoader(test_dataset, batch_size=args.data["batch_size_val"], num_workers=args.data["workers"], pin_memory=True, sampler=test_sampler)
            self.curr_step = 1



    def run(self):

        # test dataset
        if self.args.evaluate:
            self.validate("off_val")
            return

        # train
        self.train()

    def train(self):
        for epoch in range(self.start_epoch, self.args.trainer["epoch"] + 1):
            self.epoch = epoch
            self.model.switch_to("train")
            if is_main_process():
                pbar = tqdm(total=len(self.train_loader), bar_format="{l_bar}{bar:30}{r_bar}")
            for i, inputs in enumerate(self.train_loader):
                curr_lr = self.model.optim.param_groups[0]["lr"]
                self.curr_step = i + epoch * len(self.train_loader)

                self.model.set_input(*inputs)
                loss_dict = self.model.step()
                loss_str = ""
                for k in loss_dict.keys():
                    loss_dict[k] = utils.reduce_tensors(loss_dict[k]).item()
                    loss_str += f"{k}: {loss_dict[k]:.4g}, "
                # logging
                if is_main_process():
                    if self.tb_logger is not None:
                        self.tb_logger.add_scalar("lr", curr_lr, self.curr_step)
                    for k in loss_dict.keys():
                        if self.tb_logger is not None:
                            self.tb_logger.add_scalar(f"train_{k}", loss_dict[k], self.curr_step)

                        self.flogger.info(f"Iter: [{self.curr_step}]\t" + loss_str + f"lr {curr_lr:.2g}")
                    pbar.update(1)
                    pbar.set_description(f"Train [{epoch}]")
                    pbar.set_postfix_str(f" <{loss_str}, lr={curr_lr:.3e}>")

            if is_main_process():
                pbar.close()
            self.lr_scheduler.step()
            # change loss
            if is_main_process() and epoch == self.args.trainer["change_loss"]:
                self.model.change_loss()

            # validate
            if epoch % self.args.trainer["val_freq"] == 0 or epoch == self.args.trainer["epoch"] or epoch == 1:
                self.validate("on_val")

            # cur_metric = metrics_dict["mIOU"]

            # save
            if is_main_process() and (epoch % self.args.trainer["save_freq"] == 0 or epoch == self.args.trainer["epoch"]):
                # if cur_metric > best_metric:
                #     best_metric = cur_metric
                #     self.model.save_state(f"{self.args.exp_path}/checkpoints/best.pth", epoch)
                self.model.save_state(f"{self.args.exp_path}/checkpoints/ckp{epoch}.pth", epoch)

    def validate(self, phase):

        self.model.switch_to("eval")
        if is_main_process():
            pbar = tqdm(total=len(self.val_loader), bar_format="{l_bar}{bar:20}{r_bar}")
        avg_dict = {}
        metric = Evaluator(self.args.model["backbone_param"]["N_CLASSES"])
        metric1 = Evaluator(self.args.model["backbone_param"]["N_CLASSES"])
        for i, inputs in enumerate(self.val_loader):

            self.model.set_input(*inputs)
            tensor_dict, loss_dict = self.model.forward_only()
            loss_str = ""
            for k in loss_dict.keys():
                loss_dict[k] = utils.reduce_tensors(loss_dict[k]).item()
                if not k in avg_dict:
                    avg_dict[k] = 0
                avg_dict[k] += loss_dict[k]
                loss_str += f"{k}: {loss_dict[k]:.3g}|avg{avg_dict[k]/(i+1):.3g}, "
            # tb visualize
            disp_end = 1
            if is_main_process():
                if i > -1:
                    _img: np.ndarray = utils.visualize_tensor(tensor_dict, self.args.data["data_mean"], self.args.data["data_std"])
                    if self.tb_logger is not None and i == 1:
                        self.tb_logger.add_image("Image_" + phase, _img, self.curr_step)
                    image = _img.transpose(1, 2, 0)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"{self.args.exp_path}/images/{phase}_{self.curr_step}_{i}.png", image)
                pbar.update(1)
                pbar.set_description(f"Val")
                pbar.set_postfix_str(f" <{loss_str}>")
            vis_combo = tensor_dict["mask_tensors"].cpu().clone().numpy()
            inv_pred = tensor_dict["pred_mask"].cpu().clone().numpy()
            inv_target = tensor_dict["target"].cpu().clone().numpy()
            for c in range(vis_combo.shape[0]):
                mask = vis_combo[c]
                inv_pred[c][mask == 1] = 0
                inv_target[c][mask == 1] = 0
            
            tens1 = tensor_dict["target"].cpu().clone().numpy()
            tens2 = tensor_dict["pred_mask"].cpu().clone().numpy()
            
            metric.add_batch(tens1, tens2)
            metric1.add_batch(inv_target, inv_pred)

        if is_main_process():
            pbar.close()
        metric.plot_confusion_matrix(save_dir=self.args.exp_path, names=our_class.keys())
        result = {
            "mIOU": metric.Mean_Intersection_over_Union,
            "Inv-mIOU": metric1.Mean_Intersection_over_Union,
        }
        # logging
        if is_main_process():
            loss_str = ""
            result.update(avg_dict)
            for k in result.keys():
                if self.tb_logger is not None and phase == "on_val":
                    self.tb_logger.add_scalar(f"val_{k}", result[k], self.curr_step)
                loss_str += f"{k}: {result[k]:.3g} \t"

            log_str = "Validation Iter: [{0}]\t".format(self.epoch) + loss_str
            self.logger.info('\n'+log_str)
            self.flogger.info(log_str + str(result))
        return result

        