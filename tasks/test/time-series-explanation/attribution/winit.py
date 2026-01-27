import copy
import numpy as np

class BaseExplainer:
    def train_generators(self, num_epochs) :
        gen_result = self.explainer.train_generators(
            self.train_loader, self.valid_loader, num_epochs
        )
        self.explainer.test_generators(self.test_loader)
        
        return None
    
    def load_generators(self):
        self.explainer.load_generators()
        self.explainer.test_generators(self.test_loader)

class FIT(BaseExplainer):
    def __init__(
        self,
        model,
        device,
        datamodule,
        data_name,
        feature_size,
        path,
        cv
    ):
        from winit.explainer.fitexplainers import FITExplainer
        self.explainer = FITExplainer(
            device,
            feature_size,
            data_name,
            path,
        )
        
        self.explainer.set_model(model, False)
        
        self.datamodule = copy.deepcopy(datamodule)
        self.datamodule.setup()
        self.datamodule.batch_size = 100
        
        self.train_loader = self.datamodule.train_dataloader()
        self.valid_loader = self.datamodule.val_dataloader()
        self.test_loader = self.datamodule.test_dataloader()
        
    def attribute(self, x):
        return self.explainer.attribute(x)
        
class WinIT(BaseExplainer):
    def __init__(
        self,
        model,
        device,
        datamodule,
        data_name,
        feature_size,
        path,
        cv
    ):
        self.datamodule = copy.deepcopy(datamodule)
        self.datamodule.setup()
        self.datamodule.batch_size = 100
        
        self.train_loader = self.datamodule.train_dataloader()
        self.valid_loader = self.datamodule.val_dataloader()
        self.test_loader = self.datamodule.test_dataloader()
        
        from winit.explainer.winitexplainers import WinITExplainer
        self.explainer = WinITExplainer(
            device,
            feature_size,
            data_name,
            path,
            self.train_loader,
            random_state=42
        )
        
        self.explainer.set_model(model, True)
        
    def attribute(self, x):
        scores = self.explainer.attribute(x)
        num_samples, num_features, num_times, window_size = scores.shape

        aggregated_scores = np.zeros((num_samples, num_features, num_times))
        for t in range(num_times):
            relevant_windows = np.arange(t, min(t + window_size, num_times))
            relevant_obs = -relevant_windows + t - 1
            relevant_scores = scores[:, :, relevant_windows, relevant_obs]
            relevant_scores = np.nan_to_num(relevant_scores)

            aggregated_scores[:, :, t] = relevant_scores.mean(axis=-1)
        # scores# (bs, fts, ts, window_size)
        return aggregated_scores.reshape(-1, num_times, num_features)

    
    
