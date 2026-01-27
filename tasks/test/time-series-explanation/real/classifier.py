import torch as th
import os
import sys; sys.path.append(os.path.dirname(__file__))
from torchmetrics import Accuracy, Precision, Recall, AUROC, F1Score
from typing import Callable, Union

from synthetic.hmm.classifier import StateClassifier
from models.transformer import TransformerClassifier
from models.cnn import CNN

from tint.models import Net


class MimicClassifierNet(Net):
    def __init__(
        self,
        feature_size: int,
        n_state: int,
        n_timesteps: int,
        hidden_size: int,
        rnn: str = "GRU",
        dropout: float = 0.5,
        regres: bool = True,
        bidirectional: bool = False,
        loss: Union[str, Callable] = "mse",
        optim: str = "adam",
        lr: float = 0.001,
        lr_scheduler: Union[dict, str] = None,
        lr_scheduler_args: dict = None,
        l2: float = 0.0,
        model_type: str = "state",
        pam: bool = False,
    ):
        self.multiclass = (n_state > 2)
        if model_type == "state":
            classifier = StateClassifier(
                feature_size=feature_size,
                n_state=n_state,
                hidden_size=hidden_size,
                rnn=rnn,
                dropout=dropout,
                regres=regres,
                bidirectional=bidirectional,
            )

        # elif model_type == "mtand":
        #     classifier = mTANDClassifier(
        #         feature_size=feature_size,
        #         n_state=n_state,
        #         n_timesteps=n_timesteps,
        #         hidden_size=hidden_size,
        #         rnn=rnn,
        #         dropout=dropout,
        #         bidirectional=bidirectional,
        #     )
            
        # elif model_type == "seft":
        #     classifier = SeFTClassifier(
        #         feature_size=feature_size,
        #         n_state=n_state,
        #         n_timesteps=n_timesteps,
        #         hidden_size=hidden_size
        #     )
            
        elif model_type == "transformer":
            mimic_config = {
                'd_inp': feature_size,
                # 'd_model': 36,
                'nhead': 1,
                # 'nhid': 2 * 36,
                'nlayers': 1,
                'enc_dropout': 0.3,
                'max_len': n_timesteps,
                'd_static': 0,
                'MAX': n_timesteps,
                'aggreg': 'mean',
                'n_classes': n_state,
                'static': False,
            }
                          
            classifier = TransformerClassifier(**mimic_config)
            
        elif model_type == "cnn":
            classifier = CNN(d_inp=feature_size, 
                             n_classes=n_state, 
                             dim=128)

        super().__init__(
            layers=classifier,
            loss=loss,
            optim=optim,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            l2=l2,
        )
        self.save_hyperparameters()

        for stage in ["train", "val", "test"]:
            if self.multiclass:
                setattr(self, stage + "_acc", Accuracy(task="multiclass", num_classes=n_state))
                setattr(self, stage + "_pre", Precision(task="multiclass", num_classes=n_state))
                setattr(self, stage + "_rec", Recall(task="multiclass", num_classes=n_state))
                setattr(self, stage + "_f1", F1Score(task="multiclass", num_classes=n_state))
            else:
                setattr(self, stage + "_acc", Accuracy(task="binary"))
                setattr(self, stage + "_pre", Precision(task="binary"))
                setattr(self, stage + "_rec", Recall(task="binary"))
                setattr(self, stage + "_auroc", AUROC(task="binary"))

    def forward(self, *args, **kwargs) -> th.Tensor:
        return self.net(*args, **kwargs)
    
        # when execute deeplift, self.net(*args, **kwargs).softmax(-1)

    def step(self, batch, batch_idx, stage):
        x, y, mask = batch
        y_hat = self(x, mask=mask)
        
        loss = self.loss(y_hat, y)

        if self.multiclass:
            for metric in ["acc", "pre", "rec", "f1"]:
                getattr(self, stage + "_" + metric)(y_hat, y.long())
                self.log(stage + "_" + metric, getattr(self, stage + "_" + metric))
        else:
            for metric in ["acc", "pre", "rec", "auroc"]:
                getattr(self, stage + "_" + metric)(y_hat[:, 1], y.long())
                self.log(stage + "_" + metric, getattr(self, stage + "_" + metric))

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, mask = batch
        return self(x.float(), mask=mask)
    
    def predict(self, *args, **kwargs) -> th.Tensor:
        # print(self.net(*args, **kwargs))
        # print(self.net(*args, **kwargs).shape)
        # raise RuntimeError
        return self.net(*args, **kwargs).softmax(-1)