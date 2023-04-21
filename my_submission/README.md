## How to write your own models?

We recommend that you place the code for all your agents in the `my_submisison` directory (though it is not mandatory). We have added random ranker example in `random_ranker.py`

**Add your model name in** [`user_config.py`](user_config.py)

## Ranker model format
You will have access to a set of gallery images, query images, and csv files containing all the names and indexes for the images.

You will have to implement a class containing the function `predict_product_ranks`. This function should return a numpy array of shape `(num_queries, 1000)`. For ach query image your model will need to predict a set of 1000 unique gallery indexes, in order of best match first.

## What's used by the evaluator
The evaluator uses `My_Ranker` from `user_config.py` as its entrypoint. Specify the class name for your model here.

## What's AIcrowd Wrapper

Don't change this file, it is used for relaying outputs and do basic checks on evaluation server. The AIcrowdWrapper is the actual class that is called by the evaluator.
