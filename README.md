
<div align="center">
<h1>
WOW-SEG: A WORD-FREE OPEN WORLD SEGMENTATION MODEL
</h1>

</div>

<div align="center">

[Danyang Li](), [Tianhao Wu](), [Bin Lin](), [Zhenyuan Chen](), [Yang Zhang](), [Yuxuan Li]() <br>
[Ming-Ming Cheng](), [Xiang Li]() <br>
NankaiU, XXX, XXX, PekingU

</div>

<p align="center">
  <a href="XXXXX"><b>🌐 Project Website</b></a> |
  <a href="XXXXX"><b>📕 Paper</b></a> |
  <a href="XXXXX"><b>📥 Model Download</b></a> |
  <a href="XXXXX"><b>🤗 Dataset</b></a> |
  <a href="#quick-start"><b>⚡Quick Start</b></a> <br>
  <a href="#license"><b>📜 License</b></a> |
  <a href="#citation"><b>📖 Citation (BibTeX)</b></a> <br>
</p>

<p align="center">
    <img src="assets/" width="95%"> <br>
    <img src="assets/" width="95%"> <br>
</p>

## News

**2026.XX.XX**: 🚀 XXXXXXX.

**2026.XX.XX**: 🎉 WOW-Seg is accepted by ICLR 2026.

**2026.XX.XX**: Model weights (XB) and training datasets are released. Please refer to [Model](XXX) and [Datasets](XXX).

**2026.XX.XX**: WOW-Seg is released, a Word-free Open World Segmentation model for segmenting and recognizing objects from open-set categories. See [paper](XXXX)

## Introduction

**WOW-Seg** is a Word-free Open World Segmentation model for segmenting and recognizing objects from open-set categories. Specifically, WOW-Seg introduces a novel visual prompt module, Mask2Token, which transforms image masks into visual tokens and ensures their alignment with the VLLM feature space. Moreover, we introduce the Cascade Attention Mask to decouple information across different instances. This approach mitigates inter-instance interference, leading to a significant improvement in model performance. We further construct an open world region recognition test [**benchmark**](XXX): the Region Recognition Dataset(RR-7K). With 7,662 classes, it represents the most extensive category-rich region recognition dataset to date. 



<p align="center">
    <img src="assets/" width="95%"> <br>
    <img src="assets/" width="95%"> <br>
</p>

## Installation

1. Clone this repository and navigate to the base folder
```bash
git clone https://github.com/XXXX
cd WOW-Seg
```

2. Install packages
```bash
### packages for base
conda create -n WOW-Seg python=X.XX -y
conda activate WOW-Seg


```

3. Install Flash-Attention
```bash
pip install flash-attn --no-build-isolation
### (If the method mentioned above don’t work for you, try the following one)
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
```

4. Download the XXXX checkpoint:
```bash
cd llava/model/multimodal_encoder
bash download_ckpts.sh
```

## Dataset

Please refer to [this link](https://huggingface.co/datasets/Perceive-Anything/PAM-data) to download our refined and augmented data annotations.

**Note:** We do not directly provide the source images. However, for each dataset, we will provide the relevant download links or official website addresses to guide users on how to download them. [DATA_README](data/README.md)

## License

This code repository is licensed under [Apache 2.0](./LICENSE).


## Acknowledgement
We would like to thank the following projects for their contributions to this work:

- [XXXX](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [XXXX](https://github.com/facebookresearch/segment-anything)
- [XXXX](https://github.com/facebookresearch/sam2)

## Citation

If you find PAM useful for your research and applications, or use our dataset in your research, please use the following BibTeX entry.

```bibtex
@misc{li2026wowseg,
      title={WOW-SEG: A WORD-FREE OPEN WORLD SEGMENTATION MODEL}, 
      author={Danyang Li and Tianhao Wu and Bin Lin and Zhenyuan Chen and Yang Zhang and Yuxuan Li and Ming-Ming Cheng and Xiang Li},
      year={2026},
      eprint={XXXXXX},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={XXXXXXX}, 
}
```
