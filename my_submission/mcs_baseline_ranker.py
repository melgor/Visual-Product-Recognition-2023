import numpy as np
import open_clip
import torch
import yaml
from loguru import logger
from sklearn.preprocessing import normalize
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from MCS2023_baseline.data_utils.augmentations import get_val_aug_query, get_val_aug_gallery
from MCS2023_baseline.data_utils.dataset import SubmissionDataset
from MCS2023_baseline.utils import convert_dict_to_tuple
from my_submission.itbn import db_augmentation, calculate_sim_matrix, db_augmentation_both, db_augmentation_both_simbased

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


class Head(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Head, self).__init__()

    def forward(self, x):
        return torch.nn.functional.normalize(x)


class ModelToUse(torch.nn.Module):
    def __init__(self, vit_backbone):
        super(ModelToUse, self).__init__()
        self.model = vit_backbone
        self.head = Head(1024)

    def forward(self, images):
        x = self.model(images)
        return self.head(x)


class MCS_BaseLine_Ranker:
    def __init__(self, dataset_path, gallery_csv_path, queries_csv_path):
        """
        Initialize your model here
        Inputs:
            dataset_path
            gallery_csv_path
            queries_csv_path
        """
        # Try not to change
        self.dataset_path = dataset_path
        self.gallery_csv_path = gallery_csv_path
        self.queries_csv_path = queries_csv_path
        self.max_predictions = 1000

        checkpoint_path1 = 'MCS2023_baseline/experiments/vit/convnext_final_bigdatast.pt'
        self.batch_size = 32
        self.input_size = 272

        self.exp_cfg = 'config/baseline_mcs.yml'
        self.inference_cfg = 'config/inference_config.yml'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(self.exp_cfg) as f:
            data = yaml.safe_load(f)
        self.exp_cfg = convert_dict_to_tuple(data)

        with open(self.inference_cfg) as f:
            data = yaml.safe_load(f)
        self.inference_cfg = convert_dict_to_tuple(data)

        logger.info('Creating model and loading checkpoint')
        self.model_scripted = self.get_model_raw(checkpoint_path1, self.device)
        logger.info('Weights are loaded!')

    def get_model_raw(self, model_path: str, device_type: torch.device):
        vit_backbone, model_transforms, _ = open_clip.create_model_and_transforms("convnext_xxlarge")
        model = ModelToUse(vit_backbone.visual).half()
        checkpoint = torch.jit.load(model_path).half().state_dict()
        print(model.load_state_dict(checkpoint, strict=True))
        model = model.eval().to(device=device_type).to(memory_format=torch.channels_last)
        return model

    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)

    def predict_product_ranks(self):
        """
        This function should return a numpy array of shape `(num_queries, 1000)`.
        For ach query image your model will need to predict
        a set of 1000 unique gallery indexes, in order of best match first.

        Outputs:
            class_ranks - A 2D numpy array where the axes correspond to:
                          axis 0 - Batch size
                          axis 1 - An ordered rank list of matched image indexes, most confident prediction first
                            - maximum length of this should be 1000
                            - predictions above this limit will be dropped
                            - duplicates will be dropped such that the lowest index entry is preserved
        """
        gallery_dataset = SubmissionDataset(
            root=self.dataset_path, annotation_file=self.gallery_csv_path,
            transforms=get_val_aug_gallery(self.input_size)
        )

        query_dataset = SubmissionDataset(
            root=self.dataset_path, annotation_file=self.queries_csv_path,
            transforms=get_val_aug_query(self.input_size), with_bbox=True
        )

        datasets = ConcatDataset([query_dataset, gallery_dataset])
        combine_loader = torch.utils.data.DataLoader(
            datasets, batch_size=self.batch_size,
            shuffle=False, pin_memory=True, num_workers=self.inference_cfg.num_workers
        )

        logger.info('Calculating embeddings')
        embeddings = []
        # with torch.no_grad(), torch.cuda.amp.autocast():
        with torch.no_grad():
            logger.info('Start')
            for i, images in tqdm(enumerate(combine_loader), total=len(combine_loader)):
                images = images.to(self.device).to(memory_format=torch.channels_last).half()
                outputs = self.model_scripted(images).cpu().float().numpy()
                embeddings.append(outputs)

        embeddings = np.concatenate(embeddings)
        embeddings = np.nan_to_num(embeddings, posinf=0, neginf=0)
        query_embeddings = embeddings[:len(query_dataset)]
        gallery_embeddings = embeddings[len(query_dataset):]

        logger.info('Normalizing and calculating distances')
        gallery_embeddings = normalize(gallery_embeddings)
        query_embeddings = normalize(query_embeddings)

        logger.info(f"whitening, query:{query_embeddings.shape[0]} gallery:{gallery_embeddings.shape[0]}")

        # swap in DBA as query > gallery
        reference_vecs, query_vecs = db_augmentation(gallery_embeddings, query_embeddings, top_k=5)
        similarities = calculate_sim_matrix(query_vecs, reference_vecs)
        topk_final = min(similarities.shape[1], self.max_predictions)
        class_ranks = torch.topk(torch.from_numpy(similarities), topk_final, dim=1)[1].numpy()
        logger.info("Finished")
        return class_ranks
