# PGAR
The repo of the ECCV 2020 paper: [Progressively Guided Alternate Refinement Network for RGB-D Salient Object Detection](https://arxiv.org/abs/2008.07064).

Network overview
---
![image](https://github.com/ShuhanChen/PGAR_ECCV20/blob/master/Figures/arch.png)


Usage
---
Depth format: nearer pixels appear brighter and vice versa. Download our pre-trained model and put it into ``$models/``. Then, run
```
Testing:  python3 test.py
```

Pre-computed saliency maps
---
We provide saliency maps of our PGAR on 7 datasets: [Baidu](https://pan.baidu.com/s/1QoipsTNUVORYPQ6rW2mCeQ)(3jzx) and [Google](https://drive.google.com/file/d/1TADquVq-m4jwgmlIemyY0Ck7McsNbxup/view?usp=sharing).

Citation
---
```
@inproceedings{chen2020eccv, 
  author={Chen, Shuhan and Fu, Yun}, 
  booktitle={European Conference on Computer Vision}, 
  title={Progressively Guided Alternate Refinement Network for RGB-D Salient Object Detection}, 
  year={2020}
} 
```
