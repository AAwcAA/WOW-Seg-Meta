
<div align="center">
<h1>
WOW-Seg: A Word-free Open World Segmentation Model
</h1>

</div>

<div align="center">

[Danyang Li](), [Tianhao Wu](), [Bin Lin](), [Zhenyuan Chen](), [Yang Zhang](), [Yuxuan Li]() <br>
[Ming-Ming Cheng](), [Xiang Li]() <br>
NKU, SICAU, PKU

</div>

<p align="center">
  <a href="https://openreview.net/pdf?id=AyJPSnE1bq"><b>📕 Paper</b></a> |
  <a href="https://huggingface.co/AAwcAA/WOW-Seg"><b>📥 Model Download</b></a> |
  <a href="https://huggingface.co/datasets/AAwcAA/RR-7K"><b>🤗 Dataset</b></a> |
  <a href="#license"><b>📜 License</b></a> |
  <a href="#citation"><b>📖 Citation (BibTeX)</b></a> <br>
</p>

<p align="center">
    <img src="assets/1.png" width="95%"> <br>
</p>

## News

**2026.01.26**: 🎉 Our new work, WOW-Seg is accepted by ICLR 2026.

**2026.02.28**: 🎉 Model weights and datasets are released. Please refer to [Model](https://huggingface.co/AAwcAA/WOW-Seg) and [Datasets](https://huggingface.co/datasets/AAwcAA/RR-7K).

## Introduction

**WOW-Seg** is a Word-free Open World Segmentation model for segmenting and recognizing objects from open-set categories. Specifically, WOW-Seg introduces a novel visual prompt module, Mask2Token, which transforms image masks into visual tokens and ensures their alignment with the VLLM feature space. Moreover, we introduce the Cascade Attention Mask to decouple information across different instances. This approach mitigates inter-instance interference, leading to a significant improvement in model performance. We further construct an open world region recognition test [**benchmark**](XXX): the Region Recognition Dataset(RR-7K). With 7,662 classes, it represents the most extensive category-rich region recognition dataset to date. 


<p align="center">
    <img src="assets/2.png" width="95%"> <br>
</p>

## Installation

1. Clone this repository and navigate to the base folder
```bash
git clone https://github.com/AAwcAA/WOW-Seg-Meta.git
cd WOW-Seg-Meta
```

2. Install packages
```bash
### packages for base
conda create -n wow-seg python=3.10 -y
conda activate wow-seg
```

3. Install Flash-Attention
```bash
pip install flash-attn --no-build-isolation
### (If the method mentioned above don’t work for you, try the following one)
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
```

## Dataset

Please refer to [this link](https://huggingface.co/datasets/Perceive-Anything/PAM-data) to download RR-7K.

## License

This code repository is licensed under [Apache 2.0](./LICENSE).

## Acknowledgement
We would like to thank the following projects for their contributions to this work:

- [SAM](https://github.com/facebookresearch/segment-anything)
- [SAM 2](https://github.com/facebookresearch/sam2)

## Citation

If you find WOW-Seg useful for your research and applications, or use our dataset in your research, please use the following BibTeX entry.

```bibtex
@inproceedings{
    li2026wowseg,
    title={{WOW}-Seg: A Word-free Open World Segmentation Model},
    author={Danyang Li and Tianhao Wu and Bin Lin and Zhenyuan Chen and Yang Zhang and Yuxuan Li and Ming-Ming Cheng and Xiang Li},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=AyJPSnE1bq}
}
```
