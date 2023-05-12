# Semi-supervised Domain Adaptation via Prototype-based Multi-level Learning
[[Project]](https://bupt-ai-cz.github.io/ProML/) [[Paper]](https://arxiv.org/abs/2305.02693)

# Overview
![](flamework.png)
## 1. Requirements
```shell
pip install -r requirements.txt
```

## 2. Data Preparation
As the same as the [MCL](https://github.com/chester256/MCL).

## 3. Training

```shell
python -u train.py --dataset visda --base_path ./data/txt/visda/ --data_root /root/SSDA/data/visda/ --source clipart --target sketch --num 1 --log_dir ./logs --num_classes 12 --threshold2 0.4 --T 0.05
```

## 4. Acknowledgement

The code is partly based on [MME](https://github.com/VisionLearningGroup/SSDA_MME) and [MCL](https://github.com/chester256/MCL).


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

Chuang Zhu ([czhu@bupt.edu.cn](czhu@bupt.edu.cn))

If you have any questions, you can contact us directly.
