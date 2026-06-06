# DI725 Term Project - Blind Faith in Text

Evaluating multi-modal embedding robustness in remote sensing land cover analysis.
This project investigates whether **fusion strategy** moderates the
"blind faith in text" phenomenon: when an image caption is misleading,
does the model trust the text over the visual evidence?

For full details, see the [report](DI725-TP-PHASE3-REPORT.pdf).

## Research Question
Does caption accuracy affect image-text alignment in RS multi-modal
embeddings, and does fusion strategy moderate this effect?

## Approach
Two architectures are compared, both built on **frozen RemoteCLIP**
(ViT-B/32) encoders with an identical classification head, isolating
fusion strategy as the single variable:

- **Dual encoder (late fusion):** the pooled image embedding goes
  directly to the head. Text never enters the image representation.
- **Fusion encoder (cross-attention):** image patch sequences attend
  over text token sequences before classification, so text can modulate
  the image representation.

Both are trained only on matching captions, then evaluated under three
text conditions following Deng et al. (CVPR 2025): **matching**,
**corrupted** (wrong-class caption), and **irrelevant** (domain-unrelated
text).

## Results Summary
Macro F1 on the 3,000-image test set:

| Architecture | Matching | Corrupted | Irrelevant |
|---|---|---|---|
| Dual (late fusion) | 0.60 | 0.60 | 0.60 |
| Fusion (cross-attention) | 0.93 | 0.05 | 0.38 |

The dual encoder is invariant to caption corruption by construction.
The fusion encoder peaks with matching captions but collapses below
random chance under corruption, while partially recovering under
irrelevant text. This ordering shows the fusion model does not ignore
the image, but actively prefers confident yet incorrect text over visual
evidence: cross-attention fusion creates a text-specific vulnerability
that late fusion is structurally immune to.

## Dataset
This project uses the **ARAS400k** dataset, a remote sensing
segmentation and image caption dataset accepted for presentation at
CVPR 2026 - 3rd Synthetic Data for Computer Vision Workshop.
Prepared by Ümit Mert Çağlar and Alptekin Temizel
(METU Informatics Institute).

- Dataset: https://zenodo.org/records/18890661
- Paper: https://arxiv.org/abs/2603.09625
- Codebase: https://github.com/caglarmert/ARAS400k

## Models
- RemoteCLIP (chendelong/RemoteCLIP, ViT-B/32) - frozen encoders
- CLIP (openai/clip-vit-base-patch32) - reference baseline (earlier phases)

## How to Run
1. Place the ARAS400k dataset to your directory.
2. Run the notebook cells sequentially. The pipeline:
   investigation -> feature caching -> cache verification ->
   architectures -> training and three-condition evaluation.
3. Frozen encoder embeddings are cached once; training reads the cache
   and runs in seconds.

## Notes
- The dataset and the cached embeddings (`.pt`) are not included in this
  repository due to size; see the dataset link above.
- Macro F1 is the primary metric due to severe class imbalance
  (grass 47%, several classes under 2%).
