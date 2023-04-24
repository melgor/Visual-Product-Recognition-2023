# Model training procedure

This document will explain how to train model using this codebase. The codebase was modified many times during the competition then it is no so clean as it should be :(

Here I will describe 2 main points:
1. Data preparation
2. Model training
3. Model preparation for submission

# Data preparation
## Product10k and dev-set
In both cases there was no change in the files from this dataset. Just place them in folder like `dataset`.
My folder structure for these datasets in following:
```
Product10K
- train
- train.csv
- test
- test_kaggletest.csv

development_test_data
- gallery
- gallery.csv
- queries
- queries.csv
```

## Amazon Review
Note: to get requested files, you need to use [gitlab repo](https://gitlab.aicrowd.com/bartosz_ludwiczuk/visual-product-recognition-2023-starter-kit/-/tree/master) with Git LFS installed. The whole repo require > 20 GB. 

I decided to use this [dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) as from the review dataset we can get product images and user images. Connecting both of them allow to create pairs.
The main problem of using such dataset is that not all images from user are correct one (in sense of Visual-Search system). Like they present the detail of product (like 5x zoom) or product was damage.

To filter out this kind of images I decided to implement following pipeline:
- extract the embedding for each image (for both, product and user image)
- create all possible pairs product vs user and calculate cosine similarity between them
- take only these images that mean cosine similarity value > TH (I've chosen 0.6 and 0.5)

Such pipeline enabled my to create dataset of ~400k images for 0.6(`dataset/amazon_review_mean_06_full_images.csv`) and 800k for 0.5 (`dataset/amazon_review_mean_05_full_images.csv`).

Run:
```commandline
cat dataset/images_mean_50k_800.parta* > images_mean_50k_800.tar
tar -xf images_mean_50k_800.tar
```
to get prepared dataset.

Folder structure:
```
Amazon-review
- amazon_review_mean_05_full_images.csv
- amazon_review_mean_06_full_images.csv
- images_mean_50k
```

### How I created dataset
In `dataset/dataset_creation_steps` there is 4 scripts which I used to prepare dataset. 

1. Download the data from page. I downloaded these categories
- AMAZON_FASHION
- All_Beauty
- Appliances
- Clothing_Shoes_and_Jewelry
- Electronics
- Home_and_Kitchen
- Office_Products
- Sports_and_Outdoors
- Toys_and_Games
2. Create a parquet files which will store only product which have reviews with images
```bash
python get_pairs.py <downloaded_single_json>
python get_pairs.py <Office_Products.json>
```
3. Extract the features for each image. I model from Phase-2 (more on later). I didn't save images as here I'm processing them 10x than I will need in final step. Script will save snapshot every 1k iterations.
```bash
python extract_features.py <parquet_file>
python extract_features.py Toys_and_Games.parquet
```
4. Now extract the images/product which are ok for our case. So calculate cosine similarity between images in same product and take only these one which have score above 0.5/0.6
```bash
python analyse_pairs.py <path_to_parquet> "<path_to_foolder_with_embeddings>/*.embeddings"
python analyse_pairs.py Clothing_Shoes_and_Jewelry/Clothing_Shoes_and_Jewelry_pairs.parquet "Clothing_Shoes_and_Jewelry/Clothing_Shoes_and_Jewelry_pairs.parquet_*.embeddings"
```
5. Download the images
```bash
python download_images.py  
```
6. Merge all departments to single file. It will save the last CSV file which can be used for training
```bash
python prepare_training_set.py
```

These scripts were extracted from Notebooks so they are kind of chaos. If anything does not work, just ping me. 



## Config setup
In each config there is dataset section:
```
dataset:
    product10k:
        train_valid_root: '<path_to_Product10K_dir>
        train_prefix: 'train/'
        train_list: 'train.csv'
        val_prefix: 'test/'
        val_list: 'test_kaggletest.csv'

    dev_set:
        eval_root: "<path_to_development_test_data>"
        gallery: 'gallery.csv'
        queries: 'queries.csv'

    seed: 42
    num_of_classes: 9700
    input_size: 224
    batch_size: 48
    augmentations: 'default'
    augmentations_valid: 'default'
    num_workers: 8
```
You need to set `dataset.product10k.train_valid_root` and `dataset.dev_set.eval_root` (also in case of XBM Amazon review dataset need to be setup)


# Model Training

Before describing the model training recipes, here are the requirements for training the model. All training were run on GCP.

## Setup

1. Environment
```commandline
conda create -n aicrowd python=3.8
pip install -r requirements.txt
```
2. Machine
The training require GPU with 40GB of memory. I run it on single A100.

## Training
Model was training in couple of phases, as I was refining the training recipes.
To speed-up first step, I used lower resolution (224). Later on I was using 256.

Main inspiration:
- 4th place from Universal Image Embedding competition [code](https://github.com/IvanAer/G-Universal-CLIP)
  - it uses AutoAugument as Data Augmentation technique
  - warmup + cosine LR Scheduler
  - ArcMarginProduct_subcenter with adjustable margins per class
- [XBM](https://github.com/msight-tech/research-xbm)
  - use contrastive learning loss + memory-bank to make more pair comparisons
  - this works great if dataset have > ~10k classes. 

Phase 1:
Train model on Product10k, on 224 resolution. 
```bash
python -m visual_search.main_arcface --config config/arcface_1st.yml --name arcface_1st
```

Phase 2:
Still only Product10k, on 256 resolution. Before staring training you need to paste path of model from Phase 1 to config/arcface_2nd.yml (line `pretrain_model`). The same thing apply to all next steps
```bash
python -m visual_search.main_arcface --config config/arcface_2nd.yml --name arcface_2nd
```

Phase 3:
Here I switched from ArcFace to XBM. The main reason is #classes in Amazon-Review dataset (~100k), making it infeasible to store 100k x emb-dim parameters in memory.
Amazon Dataset was filtered in two ways, using cosine similarity threshold 0.5 and 0.6. Here I'm using 0.6 threshold (dataset path is in the config).
When I compare ArcFace and XBM on same dataset, ArcFace was slightly better on LB and XBM was better on local validation set.
```bash
python -m visual_search.main_arcface --config config/xbm_1st.yml --name xbm_1st
```

Phase 4:
XBM but now with dataset with threshold=0.5. It is 2x bigger than 0.6.
```bash
python -m visual_search.main_arcface --config config/xbm_2nd.yml --name xbm_2nd
```

Phase 5:
Same like in Phase 4, but I just lower down the LR.
```bash
python -m visual_search.main_arcface --config config/xbm_3st.yml --name xbm_3st
```

## Things that I tried and failed for training:
- using GEM pooling in CLIP models
- get better LR/WD parameters as based on Google Universal Embedding competition it was crucial for the winners. I just get ~0.3 with better values but also lot of time
- different types of augmentations (RandAug or standard-one)


# Model preparation for submission
For submission, I was uploading the traced model (`torch.jit.trace`) as in the begging of the competition it was a convenient way to do so. However then I discovered that `trace-model` take a lot of time to warm-up. 
Finally, I'm uploading traced model but then I read it and use just its weights with normal torch model.

```bash
python my_submission/to_torchscript.py <path_from_phase_5>
```
Then the output path need to be placed in `my_submission/mcs_baseline_ranker.py` line 53.
And then everything should work. 

## Some notes on optimization part
My aim was running model on 272 resolution. To achieve that I switched from autocast to pure fp16 precision.
Also I run CNN with Channel-Last option which give ~15% speed-boost.
Secondary, in the post-processing part I switched from full-sort to partial-sort (I needed just first 1k elements).





