import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import logging

import torch
import torch.nn.functional as F
from typing import Any, Optional

from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAccuracy, F1Score
from torchmetrics.regression import MeanAbsoluteError
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint

from modeling.emotic import EmoticNet
from arguments import parse
from data.dataloader import setup_dataset


logger = logging.getLogger(__name__)



class LitEmotic(LightningModule):
    def __init__(self, model=None, lr:float=0.01):
        super().__init__()    
        self.model = model
        self.lr = lr
        
        # define val metric
        self.mean_valid_loss = MeanMetric()
        self.val_acc = BinaryAccuracy()
        self.val_mae = MeanAbsoluteError()
        self.val_f1 = F1Score(task="multiclass", num_classes=8)

    def forward(self, x):        
        logits = self.model(x)

        return logits
        
    def training_step(self, batch, batch_idx):
        inputs, target = batch[0], batch[1]
        pred_age, pred_gender, pred_emotion = self(inputs) 
        
        pred_age, pred_gender = pred_age.squeeze(), pred_gender.squeeze()
    
        age_loss = F.l1_loss(pred_age, target[0])
        gender_loss = F.binary_cross_entropy(pred_gender, target[1])
        emotion_loss = F.cross_entropy(pred_emotion, target[2].long())
        loss = (age_loss + gender_loss + emotion_loss) / 3 
        
        self.log("train/loss", loss.detach(), on_epoch=True, prog_bar=True, logger=True,  sync_dist=True)
  
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch[0], batch[1]
        with torch.no_grad():
            pred_age, pred_gender, pred_emotion = self(inputs) 
            pred_age, pred_gender, pred_emotion = pred_age.squeeze(), pred_gender.squeeze(), pred_emotion.squeeze()
        
        # Loss
        age_loss = F.l1_loss(pred_age, target[0])
        gender_loss = F.binary_cross_entropy(pred_gender, target[1])
        emotion_loss = F.cross_entropy(pred_emotion, target[2].long())
        loss = (age_loss + gender_loss + emotion_loss) / 3 
        self.mean_valid_loss.update(loss, weight=inputs.shape[0])
        
        # Metric
        pred_emotion = torch.argmax(pred_emotion, dim=1)
        self.val_f1.update(pred_emotion, target[2])
        self.val_mae.update(pred_age, target[0])
        self.val_acc.update(pred_gender, target[1])
        
        return loss
    
    def on_validation_epoch_end(self):
        # compute metrics
        val_age = self.val_mae.compute()
        val_gender = self.val_acc.compute()
        val_emotion = self.val_f1.compute()
        
        # log metrics
        self.log("valid/loss", self.mean_valid_loss, prog_bar=True, sync_dist=True)
        self.log("valid/acc_age", val_age, prog_bar=True, sync_dist=True)
        self.log("valid/acc_gender", val_gender, prog_bar=True, sync_dist=True)
        self.log("valid/acc_emotion", val_emotion, prog_bar=True, sync_dist=True)
        
        # reset all metrics
        self.val_mae.reset()
        self.val_acc.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        inputs, target = batch[0], batch[1]
        with torch.no_grad():
            pred_age, pred_gender, pred_emotion = self(inputs) 
            pred_age, pred_gender, pred_emotion = pred_age.squeeze(), pred_gender.squeeze(), pred_emotion.squeeze()
        
        # Metric
        pred_emotion = torch.argmax(pred_emotion, dim=1)
        self.val_f1.update(pred_emotion, target[2])
        self.val_mae.update(pred_age, target[0])
        self.val_acc.update(pred_gender, target[1])
        
    
    def on_test_epoch_end(self) -> None:
        val_age = self.val_mae.compute()
        val_gender = self.val_acc.compute()
        val_emotion = self.val_f1.compute()
        
        # log metrics
        self.log("test/acc_age", val_age, sync_dist=True)
        self.log("test/acc_gender", val_gender, sync_dist=True)
        self.log("test/acc_emotion", val_emotion, sync_dist=True)
        
        # reset all metrics
        self.val_mae.reset()
        self.val_acc.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer =  torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=5e-4)
        
        return [optimizer]

    
    def save_checkpoint(self, filepath, weights_only:bool=False, storage_options:Optional[Any]=None) -> None:
        checkpoint = self._checkpoint_connector.dump_checkpoint(weights_only)
        self.strategy.save_checkpoint(checkpoint, filepath, storage_options=storage_options)
        self.strategy.barrier("Trainer.save_checkpoint")


def main(args):
    # setup config
    ## Training hyp
    batch_size, saved_ckpt_path, epochs = args.batch_size, args.output_path, args.epochs
    
    model = EmoticNet(backone='mobinetv2')
    
    # create Model wrapper
    litmodel = LitEmotic(model, lr=opt.lr)
    
    # create dataset
    if args.do_train:
        training_dataset, train_dataloader = setup_dataset(
            datapath=args.train_path,
            batch_size=batch_size,
            num_workers=args.num_proc,
            train_mode=True, 
            emotion_path=args.emotion_labels_train_path,
        )
        
    if args.do_eval:
        val_dataset, val_dataloader = setup_dataset(
            datapath=args.test_path,
            batch_size=batch_size,
            num_workers=args.num_proc,
            train_mode=False,
            emotion_path=args.emotion_labels_test_path
        )
        
        
    # create callback functions
    model_checkpoint = ModelCheckpoint(save_top_k=3,
                        monitor="valid/loss",
                        mode="min", dirpath="output/",
                        filename="sample-{epoch:02d}",
                        save_weights_only=True)

    # create Trainer
    trainer = Trainer(
        max_epochs=epochs, accelerator='cuda', devices=[0,1], callbacks=[model_checkpoint], strategy='ddp_find_unused_parameters_true'
    )
    
    if args.do_train:
        logger.info("*** Start training ***")
        trainer.fit(
            model=litmodel, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader if args.do_eval else None
        )
        
        # Saves only on the main process    
        saved_ckpt_path = f'{saved_ckpt_path}/checkpoint'
        os.makedirs(saved_ckpt_path, exist_ok=True)
        saved_ckpt_path = f'{saved_ckpt_path}/best.pt'
        trainer.save_checkpoint(saved_ckpt_path)
        
    if args.do_eval:
        logger.info("\n\n*** Evaluate ***")
        trainer.devices = 0
        trainer.test(litmodel, dataloaders=val_dataloader, ckpt_path="best")
        
        
if __name__ == '__main__':
    opt = parse()
    
    print("\nHyperparameters\n", opt, "\n")
    
    # trainer
    logger.info('*** Training mode ***')
    main(opt)
    