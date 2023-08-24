import os
from typing import Any, List

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from pytorch_lightning import LightningModule
from torchmetrics import CatMetric, MaxMetric, Accuracy, Precision, Recall

from src.models.components.pointnet import PointNetSemSeg
from src.models.components.pointnet_utils import FeatureTransformRegularizer
from src.models.components.dice_loss import SoftDiceLossV2


class PointNetModule(LightningModule):
    def __init__(
        self,
        data_dir: str = '',
        num_class: int = 2,
        lr: float = 0.001,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.num_class = self.hparams["num_class"]
        self.num_votes = 5

        self.model = PointNetSemSeg(hparams=self.hparams)

        # loss function
        if self.num_class > 2:
            self.criterion_seg = torch.nn.NLLLoss()
        else:
            self.criterion_seg = torch.nn.BCELoss()
        self.criterion_feat = FeatureTransformRegularizer()
        self.criterion_dice = SoftDiceLossV2()
        self.mat_diff_loss_scale = 0.001
        self.threshold = 0.5

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        #
        self.val_acc = Accuracy()
        self.val_precision = Precision()
        self.val_recall = Recall()
        self.val_logits = CatMetric()
        self.val_targets = CatMetric()
        #
        self.test_acc = Accuracy()
        self.test_precision = Precision()
        self.test_recall = Recall()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_precision_best = MaxMetric()
        self.val_recall_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        
        # input
        pos, x, y, fname, selected_idx = batch        
        pos_x = torch.cat([pos, x], dim=2)
        pos_x_transpose = torch.transpose(pos_x, 1, 2)        
        
        # forward
        logits, output, trans_feat = self.forward(pos_x_transpose)
        
        # reshape
        if self.num_class > 2:
            logits = logits.reshape(-1, self.num_class)
            output = output.reshape(-1, self.num_class)
            y = y.reshape(-1)
        else:
            logits = logits.reshape(-1, 1)
            output = output.reshape(-1, 1)
        y = y.reshape(-1)
        
        # clean label
        logits_clean = logits[y!=-1]
        output_clean = output[y!=-1]
        y_clean = y[y!=-1]
        
        # reshape again
        if self.num_class > 2:
            logits_clean = logits_clean.reshape(-1, self.num_class)
            output_clean = output_clean.reshape(-1, self.num_class)
        else:
            logits_clean = logits_clean.reshape(-1)
            output_clean = output_clean.reshape(-1)
        y_clean = y_clean.reshape(-1).float()
        
        # compute loss
        loss_seg = self.criterion_seg(logits_clean, y_clean)
        loss_feat = self.mat_diff_loss_scale * self.criterion_feat(trans_feat)
        loss_dice = self.criterion_dice(output_clean.reshape(-1, 1), y_clean.reshape(-1, 1))
        loss = loss_seg + loss_feat + 1.0 * loss_dice
        
        # compute preds        
        del output, output_clean
        if self.num_class > 2:
            preds = torch.argmax(logits, dim=1)
        else:
            preds = torch.tensor(logits > self.threshold, dtype=torch.float64) # threshold 0.5
        preds = preds.reshape(-1)
        
        # return dict
        output_dict = {
            # input
            'fname': fname,
            'selected_idx': selected_idx,
            'pos': pos,
            'x': x,
            'targets': y,
            # pred
            'logits': logits,
            'preds': preds,
            # metric
            'loss': loss,
            'loss_seg': loss_seg,
            'loss_feat': loss_feat,
            'loss_dice': loss_dice,
        }
        
        return output_dict

    def training_step(self, batch: Any, batch_idx: int):
        output = self.step(batch)        
        
        loss = output['loss']
        loss_seg = output['loss_seg']
        loss_feat = output['loss_feat']
        loss_dice = output['loss_dice']
        preds = output['preds']
        targets = output['targets']
        
        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss_seg", loss_seg, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss_feat", loss_feat, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss_dice", loss_dice, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        output = self.step(batch)        
        
        loss = output['loss']
        logits = output['logits']
        preds = output['preds']
        targets = output['targets']
        
        # remove label -1
        logits = logits[targets!=-1]
        preds = preds[targets!=-1]
        targets = targets[targets!=-1]
        
        if len(targets) <= 0:
            return {}
        
        else:
            # log val results
            self.val_logits.update(logits)
            self.val_targets.update(targets)

            # log val metrics
            acc = self.val_acc(preds, targets)
            precision = self.val_precision(preds, targets)
            recall = self.val_recall(preds, targets)
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/precision", precision, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/recall", recall, on_step=False, on_epoch=True, prog_bar=True)        

            return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch        
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()
        self.val_acc_best.update(acc)
        self.val_precision_best.update(precision)
        self.val_recall_best.update(recall)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.log("val/precision_best", self.val_precision_best.compute(), on_epoch=True, prog_bar=True)
        self.log("val/recall_best", self.val_recall_best.compute(), on_epoch=True, prog_bar=True)
        
    def test_step(self, batch: Any, batch_idx: int):
        output = self.step(batch)
        
        # save npy and fig
        self.test_save_result(output, save_fig=False, save_npy=True)
        
        # get value
        loss = output['loss']
        logits = output['logits']
        preds = output['preds']
        targets = output['targets']
        
        # clean label
        logits_clean = preds[targets!=-1]
        preds_clean = preds[targets!=-1]
        targets_clean = targets[targets!=-1]
        
        # log test metrics
        acc = self.test_acc(preds_clean, targets_clean)
        precision = self.test_precision(preds_clean, targets_clean)
        recall = self.test_recall(preds_clean, targets_clean)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/precision", precision, on_step=False, on_epoch=True)
        self.log("test/recall", recall, on_step=False, on_epoch=True)      

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass
        
    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        #
        self.val_acc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_logits.reset()
        self.val_targets.reset()
        #
        self.test_acc.reset()
        self.test_precision.reset()
        self.test_recall.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr
        )

    def test_save_result(self, output, save_npy=False, save_fig=True):
        fname_np = output['fname']
        selected_idx_np = output['selected_idx'].clone().detach().cpu().numpy()
        points_np = output['pos'].clone().detach().cpu().numpy()
        colors_np = output['x'].clone().detach().cpu().numpy()
        labels_np = output['targets'].clone().detach().cpu().numpy()
        logits_np = output['logits'].clone().detach().cpu().numpy()
        preds_np = output['preds'].clone().detach().cpu().numpy()
        
        bn = len(output['pos'])
        
        if save_fig:
            for fname, pt, color, label, pred in zip(fname_np, points_np, colors_np, labels_np.reshape(bn,-1,1), preds_np.reshape(bn,-1,1)): 
                 
                print(fname)
                fname = fname.replace('.npz', '')  
                if not os.path.exists("./vis/"):
                    os.makedirs("./vis/")
                
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.scatter(pt[:, 0], pt[:, 2], pt[:, 1], s=3, c=color)
                plt.savefig("./vis/%s_color.png" % fname)
                plt.close()
                
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.scatter(pt[:, 0], pt[:, 2], pt[:, 1], s=3, c=label[:, 0])
                plt.savefig("./vis/%s_label.png" % fname)
                plt.close()
                
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.scatter(pt[:, 0], pt[:, 2], pt[:, 1], s=3, c=pred[:, 0])
                plt.savefig("./vis/%s_pred.png" % fname)
                plt.close()

        if save_npy:
            for fname, selected_idx, pt, color, label, logit, pred in zip(fname_np, selected_idx_np, points_np, colors_np, labels_np.reshape(bn,-1,1), logits_np.reshape(bn,-1, 1), preds_np.reshape(bn,-1,1)):
                
                print(fname)
                fname = fname.replace('.npz', '')
                npy = np.load(os.path.join(self.hparams["data_dir"], '%s.npz' % fname), allow_pickle=True)
                
                saved_dict = {k: npy[k] for k in npy}
                
                n_valid_points = len(selected_idx)
                
                saved_dict['idx_sampled'] = selected_idx
                saved_dict['pts_sampled'] = saved_dict['pts'][selected_idx]
                saved_dict['color_sampled'] = saved_dict['color'][selected_idx]
                saved_dict['affordance_sampled'] = saved_dict['affordance'][selected_idx]                
                saved_dict['pts_sampled_normalized'] = pt[:n_valid_points]
                saved_dict['logit_sampled'] = logit[:n_valid_points]
                saved_dict['pred_sampled'] = pred[:n_valid_points]
                
                if not os.path.exists("./results/"):
                    os.makedirs("./results/")
                    
                save_path = './results/%s.npz' % fname
                np.savez_compressed(save_path, **saved_dict)