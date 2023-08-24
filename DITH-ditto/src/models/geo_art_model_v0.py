import os, glob
from pathlib import Path
from copy import deepcopy
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import AverageMeter, Precision, Recall

from src.models.modules import create_network
from src.models.modules.losses_dense_joint import PrismaticLoss, RevoluteLoss
from src.third_party.ConvONets.conv_onet.generation_two_stage import Generator3D
from src.utils import utils
from src.utils.joint_estimation import (
    aggregate_dense_prediction_r,
    eval_joint_p,
    eval_joint_r,
)

log = utils.get_logger(__name__)
# different head for occupancy and segmentation
# predict dense joint
class GeoArtModelV0(pl.LightningModule):
    def __init__(self, opt, network):
        super().__init__()
        self.opt = opt
        for k, v in opt.hparams.items():
            self.hparams[k] = v
        self.save_hyperparameters(self.hparams)
        self.model = create_network(network)
        self.cri_cls = nn.BCEWithLogitsLoss()
        self.cri_joint_p = PrismaticLoss(self.hparams)
        self.cri_joint_r = RevoluteLoss(self.hparams)

        self.seg_pr_meter = Precision(average="micro")
        self.seg_rc_meter = Recall(average="micro")
        self.seg_iou_meter = AverageMeter()

        self.revoluted_axis_ori_meter = AverageMeter()
        self.revoluted_degree_meter = AverageMeter()
        self.revoluted_p2l_ori_meter = AverageMeter()
        self.revoluted_p2l_dist_meter = AverageMeter()
        self.revoluted_displacement_meter = AverageMeter()

        self.prismatic_axis_ori_meter = AverageMeter()
        self.prismatic_offset_meter = AverageMeter()

    def forward(self, *args):
        return self.model(*args)

    def training_step(self, data, batch_idx):
        (
            logits_seg,
            logits_joint_type,
            joint_param_revolute,
            joint_param_prismatic,
        ) = self(data["pc_start"], data["pc_end"], data["p_seg"])
        joint_label = data["joint_type"].unsqueeze(-1).repeat(1, data["p_seg"].size(1))
        loss_seg = self.cri_cls(logits_seg, data["seg_label"].float()) + 1e-6
        loss_joint_cls = self.cri_cls(logits_joint_type, joint_label.float())
        joint_p_axis = joint_param_prismatic[:, :, :3]
        joint_p_t = joint_param_prismatic[:, :, 3]

        joint_r_axis = joint_param_revolute[:, :, :3]
        joint_r_t = joint_param_revolute[:, :, 3]
        joint_r_p2l_vec = joint_param_revolute[:, :, 4:7]
        joint_r_p2l_dist = joint_param_revolute[:, :, 7]

        gt_t = data["state_end"] - data["state_start"]
        loss_prismatic, _ = self.cri_joint_p(
            data["seg_label"].float(),
            joint_p_axis,
            joint_p_t,
            data["screw_axis"],
            gt_t,
        )
        loss_revolute, _ = self.cri_joint_r(
            data["p_seg"][:, :, :3],
            data["seg_label"].float(),
            joint_r_axis,
            joint_r_t,
            joint_r_p2l_vec,
            joint_r_p2l_dist,
            data["screw_axis"],
            gt_t,
            data["p2l_vec"],
            data["p2l_dist"],
        )

        if data["joint_type"].sum() == 0:
            # revolute only
            loss_joint_param = loss_revolute.mean()
        elif data["joint_type"].mean() == 1:
            # prismatic only
            loss_joint_param = loss_prismatic.mean()
        else:
            mask_reg = F.one_hot(data["joint_type"].long(), num_classes=2)
            loss_joint_param = (
                torch.stack((loss_revolute, loss_prismatic), dim=1) * mask_reg
            )
            loss_joint_param = loss_joint_param.sum(-1).mean()

        loss_seg = torch.nan_to_num(loss_seg)
        loss_joint_cls = torch.nan_to_num(loss_joint_cls)
        loss_joint_param = torch.nan_to_num(loss_joint_param)

        loss = (
            self.hparams.loss_weight_seg * loss_seg
            + self.hparams.loss_weight_joint_type * loss_joint_cls
            + self.hparams.loss_weight_joint_param * loss_joint_param
        )
        self.log("train/loss_seg", loss_seg)
        self.log("train/loss_joint_cls", loss_joint_cls)
        self.log("train/loss_joint_param", loss_joint_param)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, data, batch_idx):            
        (
            logits_seg,
            logits_joint_type,
            joint_param_revolute,
            joint_param_prismatic,
        ) = self(data["pc_start"], data["pc_end"], data["p_seg"])
        joint_label = data["joint_type"].unsqueeze(-1).repeat(1, data["p_seg"].size(1))
        loss_seg = self.cri_cls(logits_seg, data["seg_label"].float())
        loss_joint_cls = self.cri_cls(logits_joint_type, joint_label.float())
        joint_p_axis = joint_param_prismatic[:, :, :3]
        joint_p_t = joint_param_prismatic[:, :, 3]

        joint_r_axis = joint_param_revolute[:, :, :3]
        joint_r_t = joint_param_revolute[:, :, 3]
        joint_r_p2l_vec = joint_param_revolute[:, :, 4:7]
        joint_r_p2l_dist = joint_param_revolute[:, :, 7]

        gt_t = data["state_end"] - data["state_start"]

        loss_prismatic, prismatic_result_dict = self.cri_joint_p(
            data["seg_label"].float(),
            joint_p_axis,
            joint_p_t,
            data["screw_axis"],
            gt_t,
        )
        loss_revolute, revolute_result_dict = self.cri_joint_r(
            data["p_seg"][:, :, :3],
            data["seg_label"].float(),
            joint_r_axis,
            joint_r_t,
            joint_r_p2l_vec,
            joint_r_p2l_dist,
            data["screw_axis"],
            gt_t,
            data["p2l_vec"],
            data["p2l_dist"],
        )

        mask_reg = F.one_hot(data["joint_type"].long(), num_classes=2)
        loss_joint_param = (
            torch.stack((loss_revolute, loss_prismatic), dim=1) * mask_reg
        )
        loss_joint_param = loss_joint_param.sum(-1).mean()

        loss = (
            self.hparams.loss_weight_seg * loss_seg
            + self.hparams.loss_weight_joint_type * loss_joint_cls
            + self.hparams.loss_weight_joint_param * loss_joint_param
        )
        self.log("val/loss_seg", loss_seg)
        self.log("val/loss_joint_cls", loss_joint_cls)
        self.log("val/loss_joint_param", loss_joint_param)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        prob_seg = torch.sigmoid(logits_seg)
        self.seg_pr_meter.update(prob_seg, data["seg_label"].long())
        self.seg_rc_meter.update(prob_seg, data["seg_label"].long())

        seg_and = torch.logical_and(
            (prob_seg > self.hparams.test_seg_th), data["seg_label"].bool()
        )
        seg_or = torch.logical_or(
            (prob_seg > self.hparams.test_seg_th), data["seg_label"].bool()
        )
        seg_iou = seg_and.float().sum(-1) / (seg_or.float().sum(-1) + 1e-12)
        self.seg_iou_meter.update(seg_iou)

        if data["joint_type"].item() == 0:  # revoluted
            self.revoluted_axis_ori_meter.update(revolute_result_dict["axis_ori"])
            if self.hparams["r_cos_ambiguity"]:
                config_error = torch.minimum(
                    (gt_t - joint_r_t).abs(), (gt_t + joint_r_t).abs()
                )
            else:
                config_error = (gt_t - joint_r_t).abs()
            self.revoluted_degree_meter.update((config_error).abs())
            self.revoluted_p2l_ori_meter.update(revolute_result_dict["p2l_ori"])
            self.revoluted_p2l_dist_meter.update(revolute_result_dict["p2l_dist"])
            self.revoluted_displacement_meter.update(
                revolute_result_dict["displacement"]
            )

        elif data["joint_type"].item() == 1:  # prismatic
            self.prismatic_axis_ori_meter.update(prismatic_result_dict["axis_ori"])
            if self.hparams["p_cos_ambiguity"]:
                config_error = torch.minimum(
                    (gt_t - joint_p_t).abs(), (gt_t + joint_p_t).abs()
                )
            else:
                config_error = (gt_t - joint_p_t).abs()
            self.revoluted_degree_meter.update((config_error).abs())
            self.prismatic_offset_meter.update((gt_t - joint_p_t).abs())
        return loss

    def log_meter(self, meter, name):
        val = meter.compute()
        meter.reset()
        self.log(f"val/{name}", val)

    def validation_epoch_end(self, val_step_outputs):
        self.log_meter(self.seg_pr_meter, "seg_precision")
        self.log_meter(self.seg_rc_meter, "seg_recall")
        self.log_meter(self.seg_iou_meter, "seg_iou")

        self.log_meter(self.revoluted_axis_ori_meter, "revoluted_axis_ori")
        self.log_meter(self.revoluted_degree_meter, "revoluted_degree")
        self.log_meter(self.revoluted_p2l_ori_meter, "revoluted_p2l_ori")
        self.log_meter(self.revoluted_p2l_dist_meter, "revoluted_p2l_dist")
        self.log_meter(self.revoluted_displacement_meter, "revoluted_displacement")

        self.log_meter(self.prismatic_axis_ori_meter, "prismatic_axis_ori")
        self.log_meter(self.prismatic_offset_meter, "prismatic_offset")

    def test_step(self, data, batch_idx, save_pred=False, save_fig=True):
        
        print(data["data_path"])
        
        def normalize(tensor: torch.Tensor, dim: int) -> torch.Tensor:
            return tensor / ((tensor ** 2).sum(dim, keepdim=True).sqrt() + 1.0e-5)  
              
        # only support batch size 1
        assert data["pc_start"].size(0) == 1
              
        if not hasattr(self, "generator"):
            self.generator = Generator3D(
                self.model,
                device=self.device,
                threshold=self.hparams.test_occ_th,
                seg_threshold=self.hparams.test_seg_th,
                input_type="pointcloud",
                refinement_step=0,
                padding=0.1,
                resolution0=self.hparams.test_res,
            )
        result = {}

        # evaluate mesh
        c = self.model.encode_inputs(data["pc_start"], data["pc_end"])
        (
            logits_seg,
            _, # logits_joint_type,
            _, # joint_param_revolute,
            _, # joint_param_prismatic,
        ) = self(data["pc_start"], data["pc_end"], data["p_seg"])
        
        prob_seg = torch.sigmoid(logits_seg)
        mobile_points_all = data["p_seg"][prob_seg > self.hparams.test_seg_th].unsqueeze(0)
        print(mobile_points_all.shape)
        
        if mobile_points_all.size(1) == 0:
            return result       
        
        (
            logits_joint_type,
            joint_param_revolute,
            joint_param_prismatic,
        ) = self.model.decode_joints(mobile_points_all, c)
        
        # seg evaluation
        prob_seg = torch.sigmoid(logits_seg)
        seg_and = torch.logical_and(
            (prob_seg > self.hparams.test_seg_th), data["seg_label"].bool()
        )
        seg_or = torch.logical_or(
            (prob_seg > self.hparams.test_seg_th), data["seg_label"].bool()
        )
        seg_iou = seg_and.float().sum(-1) / (seg_or.float().sum(-1) + 1e-12)  
        result['segmentation'] = {'iou': seg_iou}
        
        # articulation evaluation
        joint_type_prob = logits_joint_type.sigmoid().mean()
        correct = (joint_type_prob > 0.5).long().item() == data["joint_type"][0].long().item()
        
        # revolute
        if data["joint_type"][0].item() == 0:
            gt_t = (data["state_end"] - data["state_start"]).cpu()[0].numpy()
            gt_axis = data["screw_axis"].cpu()[0].numpy()
            gt_pivot_point = data["p_seg"][:, :, :3] + data["p2l_vec"] * data[
                "p2l_dist"
            ].unsqueeze(-1)
            gt_pivot_point = gt_pivot_point[0].mean(0).cpu().numpy()
            # axis voting
            joint_r_axis = (
                normalize(joint_param_revolute[:, :, :3], -1)[0].cpu().numpy()
            )
            joint_r_t = joint_param_revolute[:, :, 3][0].cpu().numpy()
            joint_r_p2l_vec = (
                normalize(joint_param_revolute[:, :, 4:7], -1)[0].cpu().numpy()
            )
            joint_r_p2l_dist = joint_param_revolute[:, :, 7][0].cpu().numpy()
            p_seg = mobile_points_all[0].cpu().numpy()

            pivot_point = p_seg[:, :3] + joint_r_p2l_vec * joint_r_p2l_dist[:, np.newaxis]
            (
                joint_axis_pred,
                pivot_point_pred,
                config_pred,
            ) = aggregate_dense_prediction_r(
                joint_r_axis, pivot_point, joint_r_t, method="mean"
            )
            axis_ori_err, axis_displacement, config_err = eval_joint_r(
                (joint_axis_pred, pivot_point_pred, config_pred),
                (gt_axis, gt_pivot_point, gt_t),
            )
            result["articulation"] = {
                "revolute": {
                    "axis_orientation": axis_ori_err,
                    "axis_displacement": axis_displacement,
                    "config_err": config_err,
                },
                "prismatic": None,
                "joint_type": {"accuracy": correct},
            }
        # prismatic
        else:
            gt_t = (data["state_end"] - data["state_start"]).cpu()[0].numpy()
            gt_axis = data["screw_axis"].cpu()[0].numpy()
            gt_pivot_point = np.zeros(3)
            pivot_point_pred = np.zeros(3)
            # axis voting
            joint_p_axis = (
                normalize(joint_param_prismatic[:, :, :3], -1)[0].cpu().numpy()
            )
            joint_axis_pred = joint_p_axis.mean(0)
            joint_p_t = joint_param_prismatic[:, :, 3][0].cpu().numpy()
            config_pred = joint_p_t.mean()
            axis_ori_err, config_err = eval_joint_p(
                (joint_axis_pred, config_pred), (gt_axis, gt_t)
            )
            result["articulation"] = {
                "prismatic": {
                    "axis_orientation": axis_ori_err,
                    "config_err": config_err,
                },
                "revolute": None,
                "joint_type": {"accuracy": correct},
            }
        
        if save_pred:
            from pathlib import Path
            output = {k: v[0].clone().cpu().numpy() for k,v in data.items() if k not in ['data_path']}
            output["data_path"] = data["data_path"][0]
            output["seg_label_pred"] = (torch.sigmoid(logits_seg)[0] > 0.5).long().cpu().numpy()
            output["joint_axis_pred"] = joint_axis_pred
            output["pivot_point_pred"] = pivot_point_pred
            output["config_pred"] = config_pred
            output["joint_type_pred"] = (joint_type_prob > 0.5).long().item()
            
            output_path = '%s.npz'                
            fname = Path(data['data_path'][0]).stem 
            np.savez_compressed(output_path % fname, **output)
            
            
        if save_fig:
            output_path = '%s.png'
            
            fname = Path(data['data_path'][0]).stem
            
            pts = data["p_seg"].cpu().numpy()[0]            
            prob_seg = torch.sigmoid(logits_seg)[0]
            seg_label_pred = (prob_seg > 0.5).cpu().numpy()
            
            # prediction
            screw_point = pivot_point_pred
            screw_axis = joint_axis_pred
            
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=3, c=seg_label_pred)
            ax.quiver(screw_point[0], screw_point[2], screw_point[1], screw_axis[0], screw_axis[2], screw_axis[1], length=1.0, normalize=True)
            plt.savefig(output_path % (fname+'_pred'))
            plt.close()  
            
            # input
            pc_start = data["pc_start"][0].cpu().numpy()
            pc_start_color = data["pc_start_color"][0].cpu().numpy()
            pc_end = data["pc_end"][0].cpu().numpy()
            pc_end_color = data["pc_end_color"][0].cpu().numpy()
            aff_start = pc_start[:, -1]
            aff_end = pc_end[:, -1]
            
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(pc_start[:, 0], pc_start[:, 2], pc_start[:, 1], s=3, c=aff_start)
            plt.savefig(output_path % (fname+'_before'))
            plt.close()
        
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(pc_end[:, 0], pc_end[:, 2], pc_end[:, 1], s=3, c=aff_end)
            plt.savefig(output_path % (fname+'_after'))
            plt.close()
        
        return result

    def test_epoch_end(self, outputs) -> None:
        # outputs = self.all_gather(outputs)
        results_all = {
            "segmentation": {
                "iou": [],
            },
            "articulation": {
                "revolute": {
                    "axis_orientation": [],
                    "axis_displacement": [],
                    "config_err": [],
                },
                "prismatic": {"axis_orientation": [], "config_err": []},
                "joint_type": {"accuracy": []},
            },
        }
        for result in outputs:
            if "segmentation" in result:
                for k, v in result["segmentation"].items():
                    if isinstance(v, torch.Tensor):
                        v = v.cpu().numpy()
                    results_all["segmentation"][k].append(v)
            if "articulation" not in result:
                continue
            for k, v in result["articulation"].items():
                if v is None:
                    continue
                for k2, v2 in v.items():
                    if isinstance(v2, torch.Tensor):
                        v2 = v2.cpu().numpy()
                    results_all["articulation"][k][k2].append(v2)

        results_mean = deepcopy(results_all)
        for k, v in results_all["segmentation"].items():
            tmp = np.array(v).reshape(-1)
            tmp = np.mean([x for x in tmp if not np.isnan(x)])
            results_mean["segmentation"][k] = float(tmp)

        for k, v in results_all["articulation"].items():
            for k2, v2 in v.items():
                tmp = np.array(v2).reshape(-1)
                tmp = np.mean([x for x in tmp if not np.isnan(x)])
                results_mean["articulation"][k][k2] = float(tmp)

        if self.trainer.is_global_zero:
            pprint(results_mean)
            utils.save_results(results_mean)
            log.info(f"Saved results to {os.getcwd()}")
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, self.hparams.lr_decay_gamma
        )
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": self.hparams.lr_decay_freq,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_dict}
