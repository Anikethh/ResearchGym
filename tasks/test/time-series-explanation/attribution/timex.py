import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import time

from txai.utils.predictors.loss import Poly1CrossEntropyLoss, GSATLoss_Extended, ConnectLoss_Extended
from txai.utils.predictors.loss_smoother_stats import *

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.data.preprocess import process_Boiler_OLD, process_Epilepsy
from txai.utils.predictors.eval import eval_mv4
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.loss_cl import *
from txai.utils.predictors.select_models import simloss_on_val_wboth
        
class TimeXExplainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_features: int,
        num_classes: int,
        max_len: int,
        data_name: str = "default",
        split: int = 0,
        is_timex: bool = True,
    ):
        """
        :param model: Your trained PyTorch model used for inference.
        :param device: The torch device (cpu or cuda).
        :param num_features: Number of input features, e.g. embedding dimension.
        :param num_classes: Number of output classes for classification.
        :param data_name: Optional string naming the dataset, e.g. 'mimic'.
        """
        self.model = model
        self.device = device
        self.num_features = num_features
        self.num_classes = num_classes
        self.max_len = max_len
        self.data_name = data_name
        self.is_timex = is_timex
        self.split = split

        self.timex_model = None

    def train_timex(self, x_train, y_train, x_test, y_test, encoder_path, skip_training):
        from torch.utils.data import DataLoader, TensorDataset
        timesteps=(
            torch.linspace(0, 1, x_train.shape[1], device=x_train.device)
            .unsqueeze(0)
            .repeat(x_train.shape[0], 1)
        )
        x_train = x_train.transpose(0, 1)
        timesteps = timesteps.transpose(0, 1)
        
        timesteps_test = torch.linspace(0, 1, x_test.shape[1], device=x_test.device).unsqueeze(0).repeat(x_test.shape[0], 1).transpose(0,1)
        x_test = x_test.transpose(0, 1)
        
        if self.is_timex:
            from txai.models.bc_model4 import TimeXModel, AblationParameters, transformer_default_args
            from txai.trainers.train_mv4_consistency import train_mv6_consistency
        else:
            from txai.models.bc_model import TimeXModel, AblationParameters, transformer_default_args
            from txai.trainers.train_mv6_consistency import train_mv6_consistency
        
        tencoder_path = encoder_path

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        clf_criterion = Poly1CrossEntropyLoss(
            num_classes = self.num_classes,
            epsilon = 1.0,
            weight = None,
            reduction = 'mean'
        )

        sim_criterion_label = LabelConsistencyLoss()
        sim_criterion_cons = EmbedConsistencyLoss(normalize_distance = True)
        
        sim_criterion = [sim_criterion_cons, sim_criterion_label]

        selection_criterion = simloss_on_val_wboth(sim_criterion, lam = 1.0)
        
        targs = transformer_default_args
        if "state" in tencoder_path:
            archtype = "state"
        elif "transformer" in tencoder_path:
            archtype = "transformer"
        elif "cnn" in tencoder_path:
            archtype = "cnn"
        
        all_indices = np.arange(x_train.shape[1])

        np.random.seed(42)
        np.random.shuffle(all_indices)

        split_idx = int(0.9 * len(all_indices))
        train_indices = all_indices[:split_idx]
        val_indices   = all_indices[split_idx:]
        
        x_val = x_train[:, val_indices]
        timesteps_val = timesteps[:, val_indices]
        y_val = y_train[val_indices]
        
        x_train = x_train[:, train_indices]
        timesteps = timesteps[:, train_indices]
        y_train = y_train[train_indices]

        trainB = (x_train, timesteps, y_train)
        
        # Output of above are chunks
        train_dataset = DatasetwInds(*trainB)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        val = (x_val, timesteps_val, y_val)
        test = (x_test, timesteps_test, y_test)

        mu = trainB[0].mean(dim=1)
        std = trainB[0].std(unbiased = True, dim = 1)

        abl_params = AblationParameters(
            equal_g_gt = False,
            g_pret_equals_g = False, 
            label_based_on_mask = True,
            ptype_assimilation = True, 
            side_assimilation = True,
            use_ste = True,
            archtype = archtype
        )

        loss_weight_dict = {
            'gsat': 1.0,
            'connect': 2.0
        }

        # targs['trans_dim_feedforward'] = 16
        # targs['trans_dropout'] = 0.25
        targs['norm_embedding'] = False
        targs['MAX'] = self.max_len

        model = TimeXModel(
            d_inp = self.num_features,
            max_len = self.max_len,
            n_classes = self.num_classes,
            n_prototypes = 50,
            gsat_r = 0.5,
            transformer_args = targs,
            ablation_parameters = abl_params,
            loss_weight_dict = loss_weight_dict,
            masktoken_stats = (mu, std)
        )
        orig_state_dict = torch.load(tencoder_path)
        
        state_dict = {}
        
        if "state" in tencoder_path:
            for k, v in orig_state_dict.items():
                if "net.regressor" in k:
                    name = k.replace("net.regressor", "mlp")
                    state_dict[name] = v
                if "net.rnn" in k:
                    name = k.replace("net.rnn", "encoder")
                    state_dict[name] = v
        elif "transformer" in tencoder_path:
            for k, v in orig_state_dict.items():
                if "net." in k:
                    name = k.replace("net.", "")
                    state_dict[name] = v
        elif "cnn" in tencoder_path:
            for k, v in orig_state_dict.items():
                if "net." in k:
                    name = k.replace("net.", "")
                    state_dict[name] = v
            
        model.encoder_main.load_state_dict(state_dict, strict=True)
        model.to(device)

        if self.is_timex:
            model.init_prototypes(train = trainB)

            #if not args.ge_rand_init: # Copies if not running this ablation
            model.encoder_t.load_state_dict(state_dict)

        for param in model.encoder_main.parameters():
            param.requires_grad = False

        if self.is_timex:
            optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 0.001)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-4, weight_decay = 0.001)

     
        spath = f'./model/{self.data_name}/{archtype}_timex_split_{self.split}'
        if self.is_timex == False:
            spath += "_timexplus"

        start_time = time.time()

        if skip_training:
            pass
        else:
            best_model = train_mv6_consistency(
                model,
                optimizer = optimizer,
                train_loader = train_loader,
                clf_criterion = clf_criterion,
                sim_criterion = sim_criterion,
                beta_exp = 2.0,
                beta_sim = 1.0,
                val_tuple = val, 
                num_epochs = 50,
                save_path = spath,
                train_tuple = trainB,
                early_stopping = True,
                selection_criterion = selection_criterion,
                label_matching = True,
                embedding_matching = True,
                use_scheduler = True
            )

        end_time = time.time()

        print('Time {}'.format(end_time - start_time))

        sdict, config = torch.load(spath)

        model.load_state_dict(sdict)
        self.timex_model = model

        f1, _ = eval_mv4(test, self.timex_model, masked = True)
        print('Test F1: {:.4f}'.format(f1))

    def attribute(self, x_batch: torch.Tensor, additional_forward_args=None):
        self.timex_model.eval()
        
        if additional_forward_args[1] is None:
            time_batch = (
                torch.linspace(0, 1, x_batch.shape[1], device=x_batch.device)
                .unsqueeze(0)
                .repeat(x_batch.shape[0], 1)
            )
        else:
            time_batch = additional_forward_args[1]
        with torch.no_grad():
            out = self.timex_model.get_saliency_explanation(x_batch, time_batch, captum_input = True)
        
        attr_results = out['mask_in']
        # print(attr_results)

        return attr_results