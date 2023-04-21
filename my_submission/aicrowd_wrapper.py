## DO NOT CHANGE THIS FILE
## Any changes made to this file will be discarded at the server evaluation

import os
import numpy as np
from PIL import Image
import pandas as pd

from my_submission.user_config import My_Ranker

def read_image_batch(base_path, paths):
    return [read_image(os.path.join(base_path, p)) for p in paths]

def read_image(path):
    return np.array(Image.open(path))

class AIcrowdWrapper:
    """
        Entrypoint for the evaluator to connect to the user's agent
        Abstracts some operations that are done on client side
            - Reading sound files from shared disk
            - Checking predictions for basic issues
            - Writing predictions to shared disk
    """
    def __init__(self, dataset_dir='./public_dataset/'):            
        self.dataset_dir = os.getenv("AICROWD_DATASET_DIR", dataset_dir)
        assert os.path.exists(self.dataset_dir), f'{self.dataset_dir} - No such directory'

        self.gallery_csv_path = os.path.join(self.dataset_dir, 'gallery.csv')
        self.queries_csv_path = os.path.join(self.dataset_dir, 'queries.csv')
        
    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs """
        raise NameError(msg)

    def predict_product_ranks(self):
        model = My_Ranker(self.dataset_dir, self.gallery_csv_path, self.queries_csv_path)
        class_ranks = model.predict_product_ranks()
        return class_ranks