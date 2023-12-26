# Semi-supervised Domain Adaptation via Prototype-based Multi-level Learning
[[Project]](https://bupt-ai-cz.github.io/ProML/) [[Paper]](https://arxiv.org/abs/2305.02693v2) [[Supplementary Version]](https://arxiv.org/abs/2305.02693v3)

# Overview
![](/assets/framework.png)

# Motivation
To avoid misunderstandings, let us elaborate further on our motivation and give a [Supplementary Version](https://github.com/bupt-ai-cz/ProML/blob/main/assets/IJCAI2023Extend.pdf).
![](/assets/motivation.png)


## 1. Requirements
```shell
pip install -r requirements.txt
```

## 2. Data Preparation
For DomainNet, please follow [MME](https://github.com/VisionLearningGroup/SSDA_MME) to prepare the data. The expected dataset path pattern is like `your-domainnet-data-root/domain-name/class-name/images.png`.

For Office-Home, please download the [resized images](https://drive.google.com/file/d/1OkkrggGq35QSZNPuYhmrdmMZXtkqnBqO/view?usp=sharing) and extract, you will get a .pkl and a .npy file, then specify their paths in `loader/office_home.py`.

## 3. Training

```shell
python -u train.py --dataset visda --base_path ./data/txt/visda/ --data_root /root/SSDA/data/visda/ --source clipart --target sketch --num 1 --log_dir ./logs --num_classes 12 --threshold2 0.4 --T 0.05
```

## 4. Acknowledgement

The code is partly based on [MME](https://github.com/VisionLearningGroup/SSDA_MME) and [MCL](https://github.com/chester256/MCL). Thank them for their great work.


## 5. Citation
```shell
@misc{huang2023semisupervised,
      title={Semi-supervised Domain Adaptation via Prototype-based Multi-level Learning}, 
      author={Xinyang Huang and Chuang Zhu and Wenkai Chen},
      year={2023},
      eprint={2305.02693},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 6. Contact

Xinyang Huang ([hsinyanghuang7@gmail.com](hsinyanghuang7@gmail.com))

If you have any questions, you can contact us directly.
