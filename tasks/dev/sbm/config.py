
import argparse
EXPR_PATH = "/experiment_folder/"
dataset_paths = {
    "waterbirds": "/data_folder/waterbird_complete95_forest2water2",
    "celeba": "/data_folder/celeba/",
    "multinli": "/data_folder/multinli",
    "civilcomments": "/data_folder/civilcomments",
    "imagenet-9": "/data_folder/imagenet",
    "imagenet-a": "/data_folder/imagenet-a",
    "imagenet-bg": "/data_folder/bg_challenge",
    "chexpert": "/data_folder/CheXpert-v1.0-small",
    "nico": "/data_folder/NICO/NICO/multi_classification",
}
NICO_DATA_FOLDER = "/data_folder/NICO/NICO/multi_classification"
NICO_CXT_DIC_PATH = "/data_folder/NICO/Context_name2label.json"
NICO_CLASS_DIC_PATH = "/data_folder/NICO/Animal_name2label.json"

def parse_bool(v):
    if v.lower()=='true':
        return True
    elif v.lower()=='false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Adapted from https://sumit-ghosh.com/articles/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value_str = value.split('=')
            if value_str.replace('-','').isnumeric():
                processed_val = int(value_str)
            elif value_str.replace('-','').replace('.','').isnumeric():
                processed_val = float(value_str)
            elif '[' in value_str:
                eles = value_str.replace('[', '').replace(']', '').split(',')
                processed_val = [int(e.strip()) for e in eles]
            elif value_str in ['True', 'true']:
                processed_val = True
            elif value_str in ['False', 'false']:
                processed_val = False
            else:
                processed_val = value_str
            getattr(namespace, self.dest)[key] = processed_val

def data_args(parser):
    parser.add_argument(
            "--dataset",
            default="waterbirds",
            type=str,
            help="select dataset",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--num_workers",
        default=12,
        type=int,
        help="number of workers",
    )
    parser.add_argument(
        "--resolution",
        default=224,
        type=int,
        help="Number of centers",
    )

    parser.add_argument(
        "--use_shortcutwalk_dataset",
        default=False,
        type=parse_bool,
        help="specify whether to use the shortcutwalk dataset",
    )

def afr_args(parser):
    parser.add_argument(
        "--balance_classes",
        default=False,
        type=parse_bool,
        help="balance classes when computing the weights",
    )
    parser.add_argument(
        "--group_uniform",
        default=False,
        type=parse_bool,
        help="balance groups when computing the weights",
    )

    parser.add_argument(
        "--afr_gamma",
        default=0,
        type=float,
        help="gamma for computing the weights",
    )

    parser.add_argument(
        "--afr_reg_coeff",
        default=0,
        type=float,
        help="regularization strength for the l1 norm",
    )
    
    parser.add_argument(
        "--afr_epochs",
        default=20,
        type=int,
        help="number of epochs for last layer retraining",
    )



def training_args(parser):

    parser.add_argument(
        "--last_layer",
        default=True,
        type=parse_bool,
        help="specify whether to only train the last layer",
    )
    
    parser.add_argument(
        "--save_folder",
        default="",
        type=str,
        help="Folder that saves the experimental results",
    )
    parser.add_argument(
        "--erm_model",
        default='',
        type=str,
        help="ERM trained model",
    )

    parser.add_argument(
        "--train_split",
        default="train",
        type=str,
        help="Specify which split of the dataset is used for training. Default is train. When split_train < 1, this can be train_subset1",
    )

    parser.add_argument(
        "--test_split",
        default="test",
        nargs='+', 
        type=str,
        help="Specify which split of the dataset is used for testing. Default is test. This could vary with the dataset.",
    )

    parser.add_argument(
        "--mode",
        default="debug",
        type=str,
        help="training mode",
    )
    parser.add_argument(
        "--pretrained",
        default=True,
        type=parse_bool,
        help="load pretrained weights",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam", "adamw", "bert_adamw"],
        help="select optimizer",
    )
    parser.add_argument('--optimizer_kwargs', nargs='*', action=ParseKwargs, default={"momentum":0.9, "weight_decay":1.e-4, "lr":1.e-3})

    parser.add_argument(
        "--optimizer_backbone",
        type=str,
        default="sgd",
        choices=["sgd", "adam", "adamw", "bert_adamw"],
        help="select optimizer",
    )
    parser.add_argument('--optimizer_backbone_kwargs', nargs='*', action=ParseKwargs, default={"momentum":0.9, "weight_decay":1.e-4, "lr":1.e-3})
    
    parser.add_argument(
        "--epoch",
        default=100,
        type=int,
        help="number of epochs to train the main model",
    )
    

    parser.add_argument(
        "--scheduler",
        default="none",
        type=str,
        help="choose a learning rate scheduler",
    )

    parser.add_argument('--scheduler_kwargs', nargs='*', action=ParseKwargs, default={})

    parser.add_argument(
        "--scheduler_cls",
        default="none",
        type=str,
        help="choose a learning rate scheduler",
    )
    parser.add_argument('--scheduler_cls_kwargs', nargs='*', action=ParseKwargs, default={})

    parser.add_argument(
        "--optimizer_cls",
        type=str,
        default="sgd",
        choices=["sgd", "adam", "adamw", "bert_adamw"],
        help="select optimizer",
    )
    parser.add_argument('--optimizer_cls_kwargs', nargs='*', action=ParseKwargs, default={"momentum":0.9, "weight_decay":1.e-4, "lr":1.e-4})

    parser.add_argument(
        "--optimizer_vec",
        type=str,
        default="sgd",
        choices=["sgd", "adam", "adamw", "bert_adamw"],
        help="select optimizer",
    )
    parser.add_argument('--optimizer_vec_kwargs', nargs='*', action=ParseKwargs, default={"momentum":0.9, "weight_decay":1.e-4, "lr":1.e-4})

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="gpu index",
    )

    parser.add_argument(
        "--check_point",
        type=str,
        default='',
        help="path to a saved model check point",
    )

    parser.add_argument(
        "--eval_freq",
        type=int,
        default=1,
        help="frequency of evaluating the model during training",
    )

    parser.add_argument(
        "--save_freq",
        type=int,
        default=0,
        help="Frequency of saving the model during training. If save_freq <=0, then intermediate checkpoints are not saved",
    )

    parser.add_argument(
        "--split_train",
        default=1.0,
        type=float,
        help="the ratio for splitting the training dataset into two parts; when the ratio is 1.0, no splitting",
    )

    parser.add_argument(
        "--split_val",
        default=1.0,
        type=float,
        help="the ratio for splitting the validation dataset into two parts; when the ratio is 1.0, no splitting",
    )

    # parser.add_argument(
    #     "--selection_metrics",
    #     nargs='+', 
    #     type=str,
    #     help="specify model selection metrics, can be val_acc, val_worst_group_acc, val_worst_class_acc, val_class_diff",
    # )
    parser.add_argument(
        "--num_batches",
        default=200,
        type=int,
        help="select the top scoring features",
    )

    parser.add_argument(
        "--class_balanced",
        default=False,
        type=parse_bool,
        help="use class balanced sampling during training",
    )

    parser.add_argument(
        "--group_balanced",
        default=False,
        type=parse_bool,
        help="use group balanced sampling during training",
    )

def dfr_args(parser):
    parser.add_argument(
        "--dfr_reg",
        default=1.0,
        type=float,
        help="regularization strength for the l1 norm",
    )
    parser.add_argument(
        "--process_embeddings",
        default=False,
        type=parse_bool,
        help="choose whether to normalize embeddings per dimension",
    )
    parser.add_argument(
        "--dfr_epochs",
        default=20,
        type=int,
        help="number of epochs for last layer retraining",
    )
def cnc_args(parser):
    parser.add_argument(
        "--cluster_method",
        default="gmm",
        type=str,
        help="clustering method",
    )
    parser.add_argument(
        "--num_positive",
        default=10,
        type=int,
        help="number of positive samples",
    )
    parser.add_argument(
        "--num_negative",
        default=10,
        type=int,
        help="number of negative samples",
    )
    parser.add_argument(
        "--num_anchor",
        default=10,
        type=int,
        help="number of anchors",
    )
    parser.add_argument(
        "--replicate",
        default=42,
        type=int,
        help="",
    )
    parser.add_argument(
        "--balance_targets",
        default=False,
        type=parse_bool,
        help="",
    )
    parser.add_argument('--target_sample_ratio', type=float, default=1)
    parser.add_argument('--anc_loss_temp', type=float, default=0.5)
    parser.add_argument('--pos_loss_temp', type=float, default=0.5)
    parser.add_argument('--neg_loss_temp', type=float, default=0.5)
    parser.add_argument('--weight_anc_by_loss',
                        default=False, type=parse_bool)
    parser.add_argument('--weight_pos_by_loss',
                        default=False, type=parse_bool)
    parser.add_argument('--weight_neg_by_loss',
                        default=False, type=parse_bool)
    parser.add_argument('--num_negative_easy', type=int, default=0)
    parser.add_argument('--batch_factor', type=int, default=32)    
    parser.add_argument('--supervised_linear_scale_up', default=False,
                        type=parse_bool)
    parser.add_argument('--contrastive_weight', type=float, default=0.5)
    parser.add_argument('--supervised_update_delay', type=int, default=0)
    parser.add_argument('--cnc_epoch', type=int, default=100)
    parser.add_argument('--finetune_epochs', type=int, default=0)
    parser.add_argument('--hard_negative_factor', type=float, default=0)
    parser.add_argument('--single_pos', default=False, type=parse_bool)
    parser.add_argument('--clip_grad_norm', default=False,
                        type=parse_bool)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
                        

def get_args():
    parser = argparse.ArgumentParser(description="spurious correlation")
    data_args(parser)
    training_args(parser)
    dfr_args(parser)
    afr_args(parser)

    parser.add_argument(
        "--algorithm",
        default="erm",
        type=str,
        help="training algorithm",
    )

    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="choose the backbone network",
    )
    
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="load a saved model",
    )
    parser.add_argument(
        "--tag",
        default="",
        type=str,
        help="additional information",
    )
    parser.add_argument(
        "--test_erm",
        default="",
        type=str,
        help="additional information",
    )
    parser.add_argument(
        "--use_val",
        default="",
        action='store_true',
        help="use validation set for training",
    )
    parser.add_argument(
        "--add_train",
        default="",
        action='store_true',
        help="use validation and training sets for training",
    )
    
    # NeuronTune-specific flags removed in skeleton

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--jtt_lambda", type=float, default=100)
    parser.add_argument("--first_stage_epochs", type=int, default=1)
    

    parser.add_argument(
        "--result_path",
        default="",
        type=str,
        help="specify where to store the evaluation results",
    )
    args = parser.parse_args()
    return args
