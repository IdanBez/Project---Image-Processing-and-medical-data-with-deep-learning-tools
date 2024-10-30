# Project---Image-Processing-and-medical-data-with-deep-learning-tools

## Abstract

This project explores medical image segmentation and proposes a framework called BiomedCLIP-decoder segmentation. The work addresses the ongoing challenges in medical image analysis where traditional segmentation methods require extensive labeled datasets and often lack transferability across different contexts. The project leverages recent developments in Vision-Language Models (VLMs) which have shown promise in natural image processing. By incorporating text prompts as additional input for segmentation tasks, this approach aims to enable more flexible and adaptable segmentation capabilities, particularly useful when dealing with limited datasets. The proposed BiomedCLIP-decoder framework builds upon existing models like CLIP and SegmentAnything-Model, testing its effectiveness across six different medical imaging datasets. The experimentation shows that while Vision-Language Segmentation Models (VLMs) can perform comparably to traditional image-only models, there's still room for improvement in how they utilize text prompts, as image features continue to dominate the segmentation process. Despite these limitations, the project demonstrates promising results in handling diverse imaging modalities and shows potential for better adaptation to different domains. The findings suggest that while this approach offers a promising direction for medical image segmentation, further development is needed to fully leverage the capabilities of language prompts in medical image analysis.

## Pretrained Models

### BiomedCLIP and CLIPSeg
pretrained weights are readily available in the Hugging Face Model Hub
- BiomedCLIP: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- CLIPSeg: https://huggingface.co/CIDAS/clipseg-rd64-refined

### CRIS
We have used the Pretrained Model from:
- https://polimi365-my.sharepoint.com/:f:/g/personal/10524166_polimi_it/Ej-lkQiFHU1ArDG68PP-u3kBJL_UBvvn1scRU7Ps5fiIOw?e=KzFowg

## Dataset Preparation

The project utilized six different 2D medical imaging datasets, covering various medical modalities, organs, and pathologies. These datasets include both radiology and non-radiology images and were used for both binary and multi-class segmentation tasks. Each dataset was separately used for finetuning the model. To make the evaluation process efficient, the project implemented an automated prompt generation system that creates simple, standardized prompts containing basic class information and image modality type.
For instance, prompts followed patterns like:
"Segment the liver/kidney/spleen/pancreas in the abdominal region in the CT image" or "Identify and segment the pneumothorax region in the chest X-ray"

#### You can find the scripts used to create the datasets and prompts in the utils directory

#### DATASET References:
- https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks
- https://github.com/JunMa11/AbdomenCT-1K
- https://datasets.simula.no/kvasir-seg/
- https://drive.google.com/drive/folders/10QXjxBJqCf7PAXqbDvoceWmZ-qF07tFi
- http://vi.cvc.uab.es/colon-qa/cvccolondb/
- https://www.kaggle.com/datasets/balraj98/cvcclinicdb

## Methods

We initially aimed to modify the CRIS model by replacing its Vision and Text Encoders with BiomedCLIP encoders. However, due to architectural compatibility issues, the team developed two alternative approaches: finetuning the existing CRIS model and creating a new architecture by adding a decoder to BiomedCLIP, which was pretrained on medical image-text pairs. This resulted in the BiomedCLIPSeg-D model, which uses a pretrained CLIPSeg decoder. The model processes triplets consisting of a medical image, a segmentation mask, and a text prompt. While CLIPSeg can work with both CNN and Vision Transformer (ViT) backbones, CRIS is limited to CNN-based CLIP backbones. The new BiomedCLIPSeg-based models utilize transformer-based backbones for their encoders.

## Test and Finetuning

To perform tests or finetune, you can use the provided test/finetune scripts. These scripts will start the fine-tuning or test process.


