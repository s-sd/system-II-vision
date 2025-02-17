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
python3.8 -m pip install tensorflow==2.13 gym==0.26.2 matplotlib numpy
```

#### Code
```
cd system-II-vision
python3 training_example.py
```

<br/>


### Datasets used in our experiments

In addition to using the [MNIST digit dataset](https://www.tensorflow.org/datasets/catalog/mnist) and the [ImageNet-LUSS dataset](https://github.com/LUSSeg/ImageNet-S) for our experiments, we also use medical imaging datasets as outlined below:

| Dataset Link           | Organ    | Modality | Role |
| :---------------- | :------: | :----:   | :----: |
| [Medical Segmentation Decathalon](http://medicaldecathlon.com/)  |   Spleen   | CT | Training |
| [Medical Segmentation Decathalon](http://medicaldecathlon.com/)  |   Liver Vessels   | CT | Training |
| [Multi-Atlas Labeling Beyond the Cranial Vault](https://www.synapse.org/Synapse:syn3193805/wiki/89480)  |   Gallbladder   | CT | Training |
| [Multi-Atlas Labeling Beyond the Cranial Vault](https://www.synapse.org/Synapse:syn3193805/wiki/89480)  |   Adrenal Gland   | CT | Training |
| [Multi-Atlas Labeling Beyond the Cranial Vault](https://www.synapse.org/Synapse:syn3193805/wiki/89480)  |   Major Vessels   | CT | Training |
| [Multi-Atlas Labeling Beyond the Cranial Vault](https://www.synapse.org/Synapse:syn3193805/wiki/89480)  |   Stomach   | CT | Training |
| [CT-ORG](https://www.cancerimagingarchive.net/collection/ct-org/)  |   Kidneys   | CT | Training |
| [CT-ORG](https://www.cancerimagingarchive.net/collection/ct-org/)  |   Bladder   | CT | Training |
| [CHAOS](https://chaos.grand-challenge.org/)  |   Liver   | MR | Training |
| [CHAOS](https://chaos.grand-challenge.org/)  |   Kidneys   | MR | Training |
| [CHAOS](https://chaos.grand-challenge.org/)  |   Spleen   | MR | Training |
| [AMOS](https://amos22.grand-challenge.org/)  |   Bladder   | MR | Training |
| [AMOS](https://amos22.grand-challenge.org/)  |   Gallbladder   | MR | Training |
| [AMOS](https://amos22.grand-challenge.org/)  |   Prostate   | MR | Training |
| [AMOS](https://amos22.grand-challenge.org/)  |   Major Vessels   | MR | Training |
| [Medical Segmentation Decathalon](http://medicaldecathlon.com/)  |   Prostate   | MR | Training |
| [PROMIS](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(16)32401-1/fulltext)  |   Prostate Tumour   | MR | Evaluation |
| [Medical Segmentation Decathalon](http://medicaldecathlon.com/)  |   Liver Tumour   | CT | Evaluation |
| [Medical Segmentation Decathalon](http://medicaldecathlon.com/)  |   Pancreas Tumour   | CT | Evaluation |
| [Medical Segmentation Decathalon](http://medicaldecathlon.com/)  |   Colon Tumour   | CT | Evaluation |
| [KITS](https://kits-challenge.org/kits23/)  |   Kidney Tumour   | CT | Evaluation |


<br/>

### Computer-vision experiments

Please refer to the example training script in `training_example.py` for the task of MNIST digit segmentation from noisy images, implemented using our proposed approach. The System I module is trained across digits 0-4 (see [`ln 88`](training_example.py#L88) in `training_example.py`) to predict initial solutions, as outline din the first pane of the figure below. This System I module is then adapted using 4 samples from each of the target digits 6-9 (see [`ln 222`](training_example.py#L222) in `training_example.py`). The System II module is then used to predict segmentations for the the target digits (see [`ln 258`](training_example.py#L258) in `training_example.py`), as outlined in the second pane in the figure below.

![Screenshot from 2025-02-10 11-22-11](https://github.com/user-attachments/assets/9fb4e29f-7ec3-41a2-a7af-4f02cf3a53b1)

<br/>

### Cancer segmentation on medical image experiments

It also demonstrates the salient features of System II congition in humans, when evaluated for the challenging task of cancer segmentation on medical images, which can enable non-invasive cancer diagnoses but often requires extensive expertise and is plagued by limited data availability:

![Screenshot from 2025-02-10 13-05-30](https://github.com/user-attachments/assets/a909a443-a69e-4cb9-8249-a4fd7e7a4f3e)

