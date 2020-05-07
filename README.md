# BrainNet-ML-ToolBox: A Python Machine Learning Toolbox for Brain Network Classification
==============================================================

**BrainNet-ML-ToolBox** library supports the combination of models and score from
key ML libraries such as `scikit-learn <https://scikit-learn.org/stable/index.html>`_, and `xgboost <https://xgboost.ai/>`_, for data preprocessing,  dimensionality reduction, and classification. This toolbox has been put together and polished by Goktug Guvercin (goktug150140@gmail.com).

# Introduction

This repo is a machine learning (ML) toolbox including 20 different pipelines for brain network classification.

Autism spectrum disorder (ASD) affects the brain connectivity at different levels. Nonetheless, non-invasively distinguishing such effects using magnetic resonance imaging (MRI) remains very challenging to machine learning diagnostic frameworks due to ASD heterogeneity. So far, existing network neuroscience works mainly focused on functional (derived from functional MRI) and structural (derived from diffusion MRI) brain connectivity, which might not capture relational morphological changes between brain regions. Indeed, machine learning (ML) studies for ASD diagnosis using morphological brain networks derived from conventional T1-weighted MRI are very scarce.

To fill this gap, we leverage crowdsourcing by organizing a **Kaggle competition** to build a pool of machine learning pipelines for neurological disorder diagnosis with application ASD diagnosis using cortical morphological networks derived from T1-weighted MRI. The general aim of this challenge was to encourage the competitors to come up with a machine learning pipelines that can differentiate normal controls from autistic subjects using cortical morphological networks. The competitors were allowed to use built-in machine learning methods to design their brain network classification frameworks. **In this repository, we include the source codes of the top 20 teams in the competition.**

During the competition, participants were provided with a training dataset and only allowed to check their performance on a public test data. The final evaluation was performed on both public and hidden test datasets based on accuracy, sensitivity, and specificity metrics. Teams were ranked using each performance metric separately and the final ranking was determined based on the mean of all rankings. **The first-ranked team (Team-1) achieved 70% accuracy, 72.5% sensitivity, and 67.5% specificity, where the second-ranked team (Team-2) achieved 63.8%, 62.5%, 65% respectively.**

![BrainNet pipeline](https://github.com/basiralab/BrainNet-ML-ToolBox/blob/master/Fig1.png)
![BrainNet pipeline](https://github.com/basiralab/BrainNet-ML-ToolBox/blob/master/Fig2.png)

# Installation

The source codes have been tested with Python 3.6.2 version through PyCharm IDE on OSX operating system. There is no need of GPU to run the codes.

Required Python Modules:

* csv
* numpy
* pandas
* xgboost
* mlxtend
* statistics
* warnings
* matplotlib
* scikit-learn

# Dataset format:

The brain network dataset in the training stage comprised 120 samples, each represented by 595 morphological connectivity features. The testing set comprised  80 samples. If you intend to run source codes for your own dataset, some operations inside the codes such as constant feature elimination and loading data from CSV files can be modified accordingly. 

# Please cite the following paper when using BrainNet-ML-ToolBox:

Paper link on arXiv:
https://arxiv.org/pdf/2004.13321v1.pdf


