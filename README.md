<img width="1170" height="570" alt="图片2" src="https://github.com/user-attachments/assets/5542b750-e05f-4d28-86b1-6afc1ccab64e" /># DADCUNet: Dual-Attention Deformable Convolution UNet for Building Extraction

## Results
DADCUNet outperforms existing methods such as TransUnet, UNetX, and DeepLabV3+ across multiple datasets, particularly in complex urban environments. The model achieved significant improvements in:
- **Accuracy**: +1.75% over TransUnet
- **F1 Score**: +1.8 over TransUnet
- **mIoU**: +3.6 over TransUnet

### Table 1: Results on the Shanghai Dataset

| Model      | Accuracy | F1 Score | mIoU   |
|------------|----------|----------|--------|
| unet+      | 0.903419 | 0.901416 | 0.746649 |
| SWIN       | 0.904613 | 0.903195 | 0.751363 |
| unetx      | 0.912110 | 0.911162 | 0.769354 |
| deeplabv3+ | 0.920852 | 0.918726 | 0.784148 |
| TransUNet  | 0.932983 | 0.932559 | 0.832544 |
| DADCUNet   | 0.945117 | 0.944509 | 0.844489 |

### Table 2: Results on the Wuhan Dataset

| Model      | Accuracy | F1 Score | mIoU   |
|------------|----------|----------|--------|
| unetx      | 0.878965 | 0.876206 | 0.661317 |
| unet+      | 0.897243 | 0.892224 | 0.691939 |
| TransUNet  | 0.901731 | 0.897117 | 0.703300 |
| SWIN       | 0.902534 | 0.899313 | 0.705624 |
| deeplabv3+ | 0.913923 | 0.912183 | 0.742725 |
| DADCUNet   | 0.923238 | 0.921149 | 0.763816 |

### Table 3: Ablation Study Results

| Model Variant | Dataset | Accuracy | Δ% | F1 Score | Δ% | mIoU   | Δ% |
|----------------|---------|----------|----|----------|----|--------|----|
| w/o DA         | Shanghai| 0.934477 | -1.1| 0.934570 | -1.1| 0.821635 | -2.7 |
| w/o DCFF       | Shanghai| 0.936388 | -0.9| 0.935477 | -1.0| 0.821759 | -2.7 |
| Full           | Shanghai| 0.945117 | -  | 0.944509 | -  | 0.844489 | -  |
| w/o DA         | Wuhan   | 0.918216 | -0.5| 0.916107 | -0.5| 0.751541 | -1.6 |
| w/o DCFF       | Wuhan   | 0.919191 | -0.4| 0.917784 | -0.4| 0.757131 | -0.9 |
| Full           | Wuhan   | 0.923238 | -  | 0.921149 | -  | 0.763816 | -  |

<img width="965" height="918" alt="图片1" src="https://github.com/user-attachments/assets/f89c2540-0045-4594-8b84-e6b9658dcd00" />

<img width="1144" height="309" alt="图片3" src="https://github.com/user-attachments/assets/6bb81006-34e6-4105-ad1d-1207d5bc46f9" />


## Overview
DADCUNet is a deep learning architecture designed for high-precision building extraction from complex urban remote sensing images. This method introduces a novel combination of deformable convolution, dual-attention mechanisms, and multi-scale feature fusion to improve building extraction performance in challenging urban environments. By utilizing key modules like the Auxiliary Encoder, Main Encoder, and Decoder, DADCUNet captures fine details, addresses scene diversity, high-density complexity, and interference from urban roads and vehicles, providing enhanced robustness for building extraction tasks.

## Key Features
- **Deformable-Convolution Fusion Feature (DCFF) Module**: This module leverages deformable convolution to adaptively capture irregular building shapes and handle spatial variations.
- **Dual-Attention Mechanism (DANet)**: This mechanism applies both position and channel attention to improve feature representation and capture global context.
- **Improved R Block**: Replaces traditional Transformer structures to enhance the model's learning capability with GateMLP and Dual-Attention techniques.
- **Auxiliary Encoder & Main Encoder**: These encoders perform multi-scale feature extraction, fusion, and complementarity of features to improve model accuracy.
- **Decoder**: Refines the extracted features using attention mechanisms to model spatial and channel dependencies.

## Dataset
The model was evaluated using a high-resolution urban building dataset from cities like Beijing, Shanghai, Shenzhen, and Wuhan, consisting of 7,260 image patches with 63,886 labeled buildings. This dataset includes MS COCO 2017 formatted annotation files and binary building masks.You can download the dataset in https://www.scidb.cn/en/detail?dataSetId=806674532768153600 with download link https://china.scidb.cn/download?fileId=605eb02f7da87f2745b580b3&traceId=effefb13-57d8-45c6-9f87-62d24fabc8dc.

@article{Wu2021ADO,
  title={A dataset of building instances of typical cities in China},
  author={Kaishun Wu and Daoyuan Zheng and Yanling Chen and Linyun Zeng and Jiahui Zhang and Shenghua Chai and Wenjie Wu and Yongliang Yang and Shengwen Li and Yuanyuan Liu and Fang Fang},
  journal={China Scientific Data},
  year={2021},
  url={https://api.semanticscholar.org/CorpusID:273593711}
}

## Installation

### Requirements
- Python 3.x
- PyTorch
- NumPy
- OpenCV
- Matplotlib

### Installation Steps
1. Clone this repository:

git clone https://github.com/zbole/DADCUNet.git
cd DADCUNet/ourmodel

2. Install required dependencies:

3. Ensure that you have the dataset (building patches) ready in the required format.

4. Run the model:

## Evaluation Metrics
The model is evaluated based on the following standard metrics:
- **Accuracy**: Proportion of correctly predicted pixels.
- **F1 Score**: Harmonic mean of precision and recall, suitable for imbalanced classes.
- **mIoU (Mean Intersection over Union)**: Measures the intersection and union ratio for segmentation performance.

## Results
DADCUNet outperforms existing methods such as TransUnet, UNetX, and DeepLabV3+ across multiple datasets, particularly in complex urban environments. The model achieved significant improvements in:
- **Accuracy**: +1.75% over TransUnet
- **F1 Score**: +1.8 over TransUnet
- **mIoU**: +3.6 over TransUnet

## Conclusion
DADCUNet provides a powerful solution for automatic building extraction in remote sensing images, with superior performance in urban planning, post-disaster assessment, and environmental monitoring tasks. The introduction of deformable convolution and dual-attention mechanisms enables the model to effectively handle complex building structures, urban clutter, and dynamic environments, setting a new state-of-the-art in the field.

## Citation
If you use this code or dataset in your work, please cite the following paper:
@article{yourpaper2025,
title={DADCUNet: Dual-Attention Deformable Convolution UNet for Building Extraction},
author={Bole Zhang},
journal={NotAvailableNow},
year={2025},
url={NotAvailableNow}
}
