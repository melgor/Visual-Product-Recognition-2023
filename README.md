
![Banner image](https://images.aicrowd.com/uploads/ckeditor/pictures/1049/content_b8c5690e284c32f89810.jpg)

# **[Visual Product Recognition Challenge 2023](https://www.aicrowd.com/challenges/visual-product-recognition-challenge-2023)** - Starter Kit
[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/fNRrSvZkry)

This repository is the Visual Product Recognition Challenge 2023 **Starter kit**! It contains:
*  **Documentation** on how to submit your models to the leaderboard
*  **The procedure** for best practices and information on how we evaluate your agent, etc.
*  **Starter code** for you to get started!

Quick Links:

* [Visual Product Recognition Challenge 2023 - Competition Page](https://www.aicrowd.com/challenges/visual-product-recognition-challenge-2023)
* [Discussion Forum](https://www.aicrowd.com/challenges/visual-product-recognition-challenge-2023discussion)
* [Starer-Kit](https://gitlab.aicrowd.com/aicrowd/challenges/visual-product-matching-2023/visual-product-recognition-2023-starter-kit)


# Table of Contents
1. [About the Visual Product Recognition Challenge](#about-the-visual-product-recognition-challenge)
2. [Evaluation](#evaluation)
3. [Baselines](#baselines) 
4. [How to test and debug locally](#how-to-test-and-debug-locally)
5. [How to submit](#how-to-submit)
6. [Dataset](#dataset)
7. [Setting up your codebase](#setting-up-your-codebase)
8. [FAQs](#faqs)

# About the Visual Product Recognition Challenge

Enabling quick and precise search among millions of items on marketplaces is a key feature for e-commerce. The use of common text-based search engines often requires several iterations and can render unsuccessful unless exact product names are known. Image-based search provides a powerful alternative and can be particularly handy when a customer observes the desired product in real life, in movies or online media.

Recent progress in computer vision now provides rich and precise descriptors for visual content. The goal of this challenge is to benchmark and advance existing computer vision methods for the task of image-based product search. Our evaluation targets a real-case scenario where we use over 40k images for 9k products from real marketplaces. Example products include sandals and sunglasses, and their successful matching requires overcoming visual variations in terms of resolution, quality, sharpness.


## Problem Statement 

In this challenge we separate product images into user and seller photos. User photos are typically snapshots of products taken with a phone camera in cluttered scenes. Such images differ substantially from seller photos that are intended to represent products on marketplaces. We provide object bounding boxes to indicate desired products on user photos and use such images and boxes as search queries. Given a search query, the goal of the algorithm is to find correct product matches in the database of seller photos.

The contest is being held as part of Machines Can See Summit 2023.

Your contribution to this challenge can help push pattern recognition research and develop a novel algorithm for product verification. 

# Evaluation

Participants' submissions will be evaluated by mAP (mean Average Precision) score for the retrieval task.

$`AP@n = {1 \over GTP}\sum_k^n{P@k \times rel@k}`$

where GTP refers to the total number of ground truth positives, n refers to the total number of products you are interested in, P@k refers to the precision@k and rel@k is a relevance function. The relevance function is an indicator function which equals 1 if the product at rank k is relevant and equals to 0 otherwise. K = 1000 in our case.

Calculating AP for a given query, Q, with a GTP=3

# Baselines

**Baseline for MCS2023: Visual Product Recognition Challenge** 

Check the baseline training code in the `MCS2023_baseline` directory and the inference code at `my_submission/mcs_baseline_ranker.py`

**To submit the baseline** - Change the ranker class in `my_submission/user_config.py`

![Visual Products](MCS2023_baseline/figures/pipeline.jpg?raw=true "Baseline pipeline")  

This `MCS2023_baseline` with a baseline solution for the MCS2023: Visual Product Recognition Challenge. 

In this competition, participants need to train a model to search for similar 
products on the marketplaces based on a user's photo.

The idea of the basic solution is to train a classifier of different products, 
remove the classification layer and use embeddings to solve the retrieval problem.

# How to Test and Debug Locally

The best way to test your models is to run your submission locally.

You can do this by simply running  `python local_evaluation.py`. **Note that your local setup and the server evalution runtime may vary.** Make sure you mention setup your runtime according to the section: [How do I specify my dependencies?](#how-do-i-specify-my-dependencies)

# How to Submit

You can use the submission script `source submit.sh <submission name>`

More information on submissions can be found in [SUBMISSION.md](/docs/submission.md).

#### A high level description of the Challenge Procedure:
1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/visual-product-recognition-challenge-2023).
2. **Fork** this repo and start developing your solution.
3. **Train** your models on publicly available dataset, and ensure `local_evaluation.py` works.
4. **Submit** your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com)
for evaluation (full instructions below). The automated evaluation setup
will evaluate the submissions against the test data to compute and report the metrics on the leaderboard
of the competition.


# Dataset

Download the [public dataset](https://www.aicrowd.com/challenges/visual-product-recognition-challenge-2023/dataset_files)) for this competition using the link below, you'll need to accept the rules of the competition to access the data. This is intended for local validation and not sufficient for training a competent model.

# Setting Up Your Codebase

AIcrowd provides great flexibility in the details of your submission!  
Find the answers to FAQs about submission structure below, followed by 
the guide for setting up this starter kit and linking it to the AIcrowd 
GitLab.

## FAQs

* How do I submit a model?
  * More information on submissions can be found at our [submission.md](/docs/submission.md). In short, you should push you code to the AIcrowd's gitlab with a specific git tag and the evaluation will be triggered automatically.

### How do I specify my dependencies?

We accept submissions with custom runtimes, so you can choose your 
favorite! The configuration files typically include `requirements.txt` 
(pypi packages), `apt.txt` (apt packages) or even your own `Dockerfile`.

You can check detailed information about this in [runtime.md](/docs/runtime.md).

### What should my code structure look like?

Please follow the example structure as it is in the starter kit for the code structure.
The different files and directories have following meaning:


```
.
├── aicrowd.json                # Add any descriptions about your model and gpu flag
├── apt.txt                     # Linux packages to be installed inside docker image
├── requirements.txt            # Python packages to be installed
├── local_evaluation.py         # Use this to check your model evaluation flow locally
├── MCS2023_baseline            # Baseline for training on Products10k dataset
└── my_submission               # Place your models and related code here
    ├── <Your model files>      # Add any models here for easy organization
    ├── aicrowd_wrapper.py      # Keep this file unchanged
    └── user_config.py          # IMPORTANT: Add your model name here
```

### How can I get going with an existing baseline?

See [baselines section](#baselines)

### How can I get going with a completely new model?

Train your model as you like, and when you’re ready to submit, implement the inference class and import it to `my_submission/user_config.py`. Refer to [`my_submission/README.md`](my_submission/README.md) for a detailed explanation.

Once you are ready, test your implementation `python local_evaluation.py`

### How do I actually make a submission?

The submission is made by adding everything including the model to git,
tagging the submission with a git tag that starts with `submission-`, and 
pushing to AIcrowd's GitLab. The rest is done for you!

For large model weight files, you'll need to use `git-lfs`

More details are available at [docs/submission.md](/docs/submission.md).

### How to use GPU?

To use GPU in your submissions, set the gpu flag in `aicrowd.json`. 

```
    "gpu": true,
```

### Are there any hardware or time constraints?

Your submission will need to complete predictions on **all the query images** under **10 minutes**. Make sure you take advantage of all the cores by parallelizing your code if needed. 

The machine where the submission will run will have following specifications:
* 4 vCPUs
* 16GB RAM
* (Optional) 1 NVIDIA T4 GPU with 16 GB VRAM - This needs setting `"gpu": true` in `aicrowd.json`

# 📎 Important links
- 💪 Challenge Page: https://www.aicrowd.com/challenges/visual-product-recognition-challenge-2023
- 🗣️ Discussion Forum: https://www.aicrowd.com/challenges/visual-product-recognition-challenge-2023discussion
- 🏆 Leaderboard: https://www.aicrowd.com/challenges/visual-product-recognition-challenge-2023/leaderboards

**Best of Luck** 🎉 🎉
