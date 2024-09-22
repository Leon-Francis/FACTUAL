# Introduction

This repository open-sources the code and datas used in our [paper](https://arxiv.org/abs/2402.14296)「**Mitigating Biases of Large Language Models in Stance Detection with Counterfactual Augmented Calibration**」

<img src="figures\framework.svg" width = "100%" />

Please cite our paper and kindly give a star for this repository if you use our code or data.

# Requirements

Seeing in requirement.txt

You could using `pip install -r requirement.txt` to install the required packages.

# Usage

## Dataset

Download your needed model weights into `model_state` or remove all `model_state/` dir prefix in all config files in `configs` to automatically download the model weights.

Download the [Sem16](https://alt.qcri.org/semeval2016/task6), [P-stance](https://github.com/chuchun8/PStance), and [VAST](https://github.com/emilyallaway/zero-shot-stance) or other stance detection dataset, place them into `dataset/<dataset name>`

Process the datasets into the following format:

```
# Each file is a csv file, containing at least the three keys 'Tweet', 'Target', 'Stance'
- datasets
  - <dataset name>
    - in-target
      - <target name>
        - train.csv
        - valid.csv
        - test.csv
      - <target name>
        - ...
    - zero-shot
      - <target name>
        - train.csv
        - valid.csv
        - test.csv
      - <target name>
        - ...
  - <dataset name>
    - ...
```

The way of how I process the datasets is shown in `datasets/preprocess_datasets.py`

## Stance Detection on Social Media with Background Knowledge

```bash
sh scripts/run_FACTUAL.sh
```

Take in-target stance detection on p-stance for example

```bash
>>> sh scripts/run_FACTUAL.sh
>>> input training dataset: [sem16, p_stance, vast]: p_stance
>>> input train dataset mode: [in_target, zero_shot]: in_target
>>> input model framework: [rationale, cad]: cad
>>> input llm name: [gpt, llama]: gpt
>>> input model name: [bert_base, roberta_base, bertweet_base, robert_base_sentiment, kebert]: roberta_base
>>> input running mode: [sweep, wandb, normal]: normal
>>> input training cuda idx: Your Cuda index
```

# Citation

The BibTex of the citation is as follows:

```bibtex
@misc{li2024mitigatingbiaseslargelanguage,
      title={Mitigating Biases of Large Language Models in Stance Detection with Calibration}, 
      author={Ang Li and Jingqian Zhao and Bin Liang and Lin Gui and Hui Wang and Xi Zeng and Xingwei Liang and Kam-Fai Wong and Ruifeng Xu},
      year={2024},
      eprint={2402.14296},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.14296}, 
}
```

# Contact

angli@stu.hit.edu.cn

If you find our paper or codes useful, please give us a kind star. ❤️