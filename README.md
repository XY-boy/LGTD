# LGTD (IEEE TCSVT 2023)
### 📖[**Paper**](https://ieeexplore.ieee.org/document/10239514) | 🖼️[**PDF**](./img/lgtd.pdf) | 🎁[**Dataset**](https://zenodo.org/record/6969604)

PyTorch codes for "[Local-Global Temporal Difference Learning for Satellite Video Super-Resolution](https://ieeexplore.ieee.org/document/10239514)", **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)**, 2023.

- Authors: [Yi Xiao](https://xy-boy.github.io/), [Qiangqiang Yuan*](http://qqyuan.users.sgg.whu.edu.cn/), [Kui Jiang](https://scholar.google.com.hk/citations?user=AbOLE9QAAAAJ&hl=zh-CN), Xianyu Jin, [Jiang He](https://jianghe96.github.io/), [Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html), and [Chia-Wen Lin](https://www.ee.nthu.edu.tw/cwlin/)<br>
- Wuhan University, Harbin Institute of Technology, and National Tsinghua University 

### Abstract
> Optical-flow-based and kernel-based approaches have been extensively explored for temporal compensation in satellite Video Super-Resolution (VSR). However, these techniques are less generalized in large-scale or complex scenarios, especially in satellite videos. In this paper, we propose to exploit the well-defined temporal difference for efficient and effective temporal compensation. To fully utilize the local and global temporal information within frames, we systematically modeled the short-term and long-term temporal discrepancies since we observed that these discrepancies offer distinct and mutually complementary properties. Specifically, we devise a Short-term Temporal Difference Module (S-TDM) to extract local motion representations from RGB difference maps between adjacent frames, which yields more clues for accurate texture representation. To explore the global dependency in the entire frame sequence, a Long-term Temporal Difference Module (L-TDM) is proposed, where the differences between forward and backward segments are incorporated and activated to guide the modulation of the temporal feature, leading to a holistic global compensation. Moreover, we further propose a Difference Compensation Unit (DCU) to enrich the interaction between the spatial distribution of the target frame and temporal compensated results, which helps maintain spatial consistency while refining the features to avoid misalignment. Rigorous objective and subjective evaluations conducted across five mainstream video satellites demonstrate that our method performs favorably against state-of-the-art approaches.
> 
### Network  
 ![image](/fig/network.png)
## 🧩Install
```
git clone https://github.com/XY-boy/LGTD.git
```
## Environment
 * CUDA 11.1
 * PyTorch 1.9.1
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
### Qualitative results
 ![image](/fig/res2.png)
 
### Quantitative results
 ![image](/fig/res1.png)
 
#### More details can be found in our paper!

## Contact
If you have any questions or suggestions, feel free to contact me. 😊  
Email: xiao_yi@whu.edu.cn; xy574475@gmail.com

## Citation
If you find our work helpful in your research, please consider citing it. Thank you! 😊😊
```
@ARTICLE{xiao2023lgtd,
  author={Xiao, Yi and Yuan, Qiangqiang and Jiang, Kui and Jin, Xianyu and He, Jiang and Zhang, Liangpei and Lin, Chia-wen},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Local-Global Temporal Difference Learning for Satellite Video Super-Resolution}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TCSVT.2023.3312321}
}
```
