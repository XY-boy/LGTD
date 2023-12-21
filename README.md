# LGTD (IEEE TCSVT 2023)
### 📖[**Paper**](https://ieeexplore.ieee.org/document/10239514) | 🖼️[**PDF**](./img/XY-IF.pdf)

PyTorch codes for "[Local-Global Temporal Difference Learning for Satellite Video Super-Resolution](https://ieeexplore.ieee.org/document/10239514)", **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)**, 2023.

Authors: [Yi Xiao](https://xy-boy.github.io/), [Qiangqiang Yuan*](http://qqyuan.users.sgg.whu.edu.cn/), Kui Jiang, Xianyu Jin, Jiang He, [Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html), and [Chia-Wen Lin](https://www.ee.nthu.edu.tw/cwlin/), and <br>
Wuhan University
### Abstract
> As a new earth observation tool, satellite video has been widely used in remote-sensing field for dynamic analysis. Video super-resolution (VSR) technique has thus attracted increasing attention due to its improvement to spatial resolution of satellite video. However, the difficulty of remote-sensing image alignment and the low efficiency of spatial–temporal information fusion make poor generalization of the conventional VSR methods applied to satellite videos. In this article, a novel fusion strategy of temporal grouping projection and an accurate alignment module are proposed for satellite VSR. First, we propose a deformable convolution alignment module with a multiscale residual block to alleviate the alignment difficulties caused by scarce motion and various scales of moving objects in remote-sensing images. Second, a temporal grouping projection fusion strategy is proposed, which can reduce the complexity of projection and make the spatial features of reference frames play a continuous guiding role in spatial–temporal information fusion. Finally, a temporal attention module is designed to adaptively learn the different contributions of temporal information extracted from each group. Extensive experiments on Jilin-1 satellite video demonstrate that our method is superior to current state-of-the-art VSR methods.
### Network  
 ![image](/img/network.png)
## 🧩Install
```
git clone https://github.com/XY-boy/MSDTGP.git
```
## Environment
 * CUDA 10.0
 * pytorch 1.x
 * build [DCNv2](https://github.com/CharlesShang/DCNv2)
 
 ## Dataset Preparation
 Please download our dataset in 
 * Baidu Netdisk [Jilin-189](https://pan.baidu.com/s/1Y1-mS5gf7m8xSTJQPn4WZw) Code:31ct
 * Zenodo: <a href="https://doi.org/10.5281/zenodo.6969604"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6969604.svg" alt="DOI"></a>
 
You can also train your dataset following the directory sturture below!
 
### Data directory structure
trainset--  
&emsp;|&ensp;train--  
&emsp;&emsp;|&ensp;LR4x---  
&emsp;&emsp;&emsp;| 000.png  
&emsp;&emsp;&emsp;| ···.png  
&emsp;&emsp;&emsp;| 099.png  
&emsp;&emsp;|&ensp;GT---   
&emsp;&emsp;|&ensp;Bicubic4x--- 

testset--  
&emsp;|&ensp;eval--  
&emsp;&emsp;|&ensp;LR4x---  
&emsp;&emsp;&emsp;| 000.png  
&emsp;&emsp;&emsp;| ···.png  
&emsp;&emsp;&emsp;| 099.png  
&emsp;&emsp;|&ensp;GT---   
&emsp;&emsp;|&ensp;Bicubic4x--- 
 ## Training
```
python main.py
```

## Test
```
python eval.py
```
### Quantitative results
 ![image](/img/res1png.png)
 
 ### Qualitative results
 ![image](/img/res2.png)
 #### More details can be found in our paper!

## Contact
If you have any questions or suggestions, feel free to contact me. 😊  
Email: xiao_yi@whu.edu.cn; xy574475@gmail.com

## Citation
If you find our work helpful in your research, please consider citing it. Thank you! 😊😊
```
@ARTICLE{xiao2022msdtgp,  
author={Xiao, Yi and Su, Xin and Yuan, Qiangqiang and Liu, Denghong and Shen, Huanfeng and Zhang, Liangpei},  
journal={IEEE Transactions on Geoscience and Remote Sensing},  
title={Satellite Video Super-Resolution via Multiscale Deformable Convolution Alignment and Temporal Grouping Projection},   
year={2022},  
volume={60},  
number={},  
pages={1-19},  
doi={10.1109/TGRS.2021.3107352}
}
```

## Acknowledgement
Our work is built upon [RBPN](https://github.com/alterzero/RBPN-PyTorch) and [TDAN](https://github.com/YapengTian/TDAN-VSR-CVPR-2020
🚀🚀Coming soon !!

# Compared with State-of-the-arts
 ![image](/fig/res2.png)
 
 # Quantitative results
 ![image](/fig/res1.png)

 ## Contact
If you have any questions or suggestions, feel free to contact me. 😊  
Email: xiao_yi@whu.edu.cn; xy574475@gmail.com
