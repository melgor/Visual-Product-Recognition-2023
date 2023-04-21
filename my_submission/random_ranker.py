import numpy as np
import pandas as pd


class Random_Ranker:
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

        # Add your code below

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

        gallery_df = pd.read_csv(self.gallery_csv_path)
        gallery_index = list(gallery_df.seller_img_id)
        replace = len(gallery_index) < self.max_predictions
        class_ranks = np.array(
                        [np.random.choice(gallery_index, size=self.max_predictions, replace=replace)
                            for _ in range(len(gallery_df))]
                        )
        return class_ranks