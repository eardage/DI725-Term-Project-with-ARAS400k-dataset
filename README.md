# DI725 Term Project - Blind Faith in Text

Evaluating text bias in RS multi-modal embeddings.
For detailed information about the project:
[report](DI725_Term_Project_Phase_2.pdf).

## Dataset
This project uses the **ARAS400k** dataset, a remote sensing 
segmentation and image caption dataset accepted for presentation 
at CVPR 2026 - 3rd Synthetic Data for Computer Vision Workshop.
Prepared by Ümit Mert Çağlar and Alptekin Temizel 
(METU Informatics Institute).

- Dataset: https://zenodo.org/records/18890661
- Paper: https://arxiv.org/abs/2603.09625
- Codebase: [https://lnkd.in/dFdgShJz](https://github.com/caglarmert/ARAS400k)

## How to Run
1. Upload dataset to Kaggle
2. Run notebook cells sequentially
3. Results saved to phase2_results.png

## Models
- CLIP (openai/clip-vit-base-patch32)
- RemoteCLIP (chendelong/RemoteCLIP)

## Results Summary
- CLIP caption similarity: 0.2788 (correct) vs 0.2681 (misleading)
- RemoteCLIP caption similarity: 0.2430 (correct) vs 0.2242 (misleading)
- RemoteCLIP shows 75% larger sensitivity to caption corruption than CLIP
