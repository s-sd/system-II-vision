# Automating fast-and-slow thinking enhances machine vision

This repository contains code to implement a dual-process method for machine cognition, which consists of two modules: 

1) **The System I module**: for fast decisions in familiar scenarios, trained adversarially across multiple tasks
2) **The System II module**: for slow and effortful reasoning to refine solutions in novel tasks, using reinforcement learning self-play to propose, consider and implement decision strateiges akin to human reasoning.

Please refer to the example training script in `training_example.py` for the task of MNIST digit segmentation from noisy images, implemented using our proposed approach.

Our approach (summarised below) outeperforms commonly used deep learning methods:

![Screenshot from 2025-02-10 11-22-11](https://github.com/user-attachments/assets/9fb4e29f-7ec3-41a2-a7af-4f02cf3a53b1)

It also demonstrates the salient features of System II congition in humans, when evaluated for the challenging task of cancer segmentation on medical images, which can enable non-invasive cancer diagnoses but often requires extensive expertise and is plagued by limited data availability:

![cancer_quant](https://github.com/user-attachments/assets/fcd2eb8d-2df2-43fe-8970-acb6041a4928)
