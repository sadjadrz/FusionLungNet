FusionLungNet: Multi-scale Fusion Convolution with Refinement Network for Lung CT Image Segmentation
---
> This repository is the official PyTorch implementation of the paper "[FusionLungNet: Multi-scale Fusion Convolution with Refinement Network for Lung CT Image Segmentation](https://arxiv.org/pdf/2410.15812)"


> [Sadjad Rezvani](https://scholar.google.com/citations?user=jxn15pUAAAAJ&hl=en&oi=sra), [Mansoor Fateh](https://scholar.google.com/citations?user=ZHezeMIAAAAJ&hl=en&oi=ao), [Yeganeh Jalali](https://scholar.google.com/citations?user=v2yd2SUAAAAJ) and [Amirreza Fateh](https://scholar.google.com/citations?user=wjNokn4AAAAJ&hl=en&oi=ao)

> New lung segmentation methods face difficulties in identifying long-range relationships between image components, reliance on convolution operations that may not capture all critical features, and the complex structures of the lungs. Furthermore, semantic gaps between feature maps can hinder the integration of relevant information, reducing model accuracy. Skip connections can also limit the decoder's access to complete information, resulting in partial information loss during encoding. To overcome these challenges, we propose a hybrid approach using the FusionLungNet network, which has a multi-level structure with key components, including the ResNet-50 encoder, Channel-wise Aggregation Attention (CAA) module, Multi-scale Feature Fusion (MFF) block, Self-Refinement (SR) module, and multiple decoders. The refinement sub-network uses convolutional neural networks for image post-processing to improve quality. Our method employs a combination of loss functions, including SSIM, IOU, and focal loss, to optimize image reconstruction quality. We created and publicly released a new dataset for lung segmentation called [LungSegDB](https://github.com/sadjadrz/Lung-segmentation-dataset), including 1800 CT images from  LIDC-IDRI dataset (dataset version 1) and 700 images from the Chest CT Cancer Images from kaggle dataset (dataset version 2).

### Network Architecture
![image](https://github.com/user-attachments/assets/21a53fc1-5333-4ce3-bcf4-870e414ffe02)

## Contact
For any questions or inquiries, please contact us at sadjadRezvani@gmail.com.

Thank you for your interest in FusionLungNet!

---

_This page will be updated with more information as soon as it is available._

