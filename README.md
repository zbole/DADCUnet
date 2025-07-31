# DADCUNet: Dual-Attention Deformable Convolution UNet for Building Extraction

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
