# Project---Image-Processing-and-medical-data-with-deep-learning-tools

## Abstract

 Medical image segmentation of anatomical structures and pathology is crucial in modern clinical diagnosis, disease study, and treatment planning. 
Traditional methods require large amounts of labeled data and are limited in their transferability across domains often perform well in a single domain but fail when applied to new contexts. There's a strong need for generalizable segmentation techniques that can adapt to new environments without retraining.
Building upon recent advancements in foundation Vision-Language Models (VLMs) from natural image-text pairs, several studies have proposed adapting them to Vision-Language Segmentation Models (VLSMs) that allow using language text as an additional input to segmentation models. Introducing auxiliary information via text with human-in-the-loop prompting during inference opens up unique opportunities, and potentially more robust segmentation models against out-of-distribution data. Although transfer learning from natural to medical images has been explored for image only segmentation models, the joint representation of vision-language in segmentation problems remains underexplored. 
In this work, we propose a novel framework, called BiomedCLIP-decoder segmentation which is inspired by CRIS model to generate segmentation of clinical scans using text prompts. We focused on transferring VLSMs to 2D medical images, using carefully curated 6 datasets encompassing diverse modalities and insightful language prompts and experiments. Our findings demonstrate that although VLSMs show competitive performance compared to image-only models for segmentation after finetuning in limited medical image datasets, not all VLSMs utilize the additional information from language prompts, with image features playing a dominant role.
Our results suggest that novel approaches are required to enable VLSMs to leverage the various auxiliary information available through language prompts.
 VLSMs exhibit enhanced performance in handling datasets with diverse modalities and show potential robustness to domain shifts compared to conventional segmentation models. 

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


