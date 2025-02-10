# Automating fast-and-slow thinking enhances machine vision

Shaheer U. Saeed [1,2∗], Yipei Wang [1,2], Veeru Kasivisvanathan [3], Brian R. Davidson [3], Matthew J. Clarkson [1,2], Yipeng Hu [1,2], Daniel C. Alexander [1,4]

[1] UCL Hawkes Institute, University College London, UK <br/>
[2] Department of Medical Physics and Biomedical Engineering, University College London, UK <br/>
[3] Division of Surgery and Interventional Sciences, University College London, UK <br/>
[4] Department of Computer Science, University College London, UK <br/>
[∗] Correspondence e-mail: shaheer.saeed.17@ucl.ac.uk <br/>

### Citation

```
To-be-added
```

### Description

This repository contains code to implement a dual-process method for machine cognition, which consists of two modules: 

1) **The System I module**: for fast decisions in familiar scenarios, trained adversarially across multiple tasks
2) **The System II module**: for slow and effortful reasoning to refine solutions in novel tasks, using reinforcement learning self-play to propose, consider and implement decision strateiges, akin to human reasoning

<br/>

### How-to

#### Dependencies
```
pip install tensorflow==2.13 gym==0.26.2 matplotlib numpy
```

#### Code
```
cd system-II-vision
python3 training_example.py
```

<br/>


### Results for computer-vision tasks

Please refer to the example training script in `training_example.py` for the task of MNIST digit segmentation from noisy images, implemented using our proposed approach.

Our approach (summarised below) outeperforms commonly used deep learning methods:

![Screenshot from 2025-02-10 11-22-11](https://github.com/user-attachments/assets/9fb4e29f-7ec3-41a2-a7af-4f02cf3a53b1)

<br/>

### Results for cancer segmentation on medical images

It also demonstrates the salient features of System II congition in humans, when evaluated for the challenging task of cancer segmentation on medical images, which can enable non-invasive cancer diagnoses but often requires extensive expertise and is plagued by limited data availability:

![cancer_quant](https://github.com/user-attachments/assets/fcd2eb8d-2df2-43fe-8970-acb6041a4928)
