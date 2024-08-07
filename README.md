# Prompt Optimization with Human Feedback
This is the official implementation of the paper [Prompt Optimization with Human Feedback](https://arxiv.org/abs/2405.17346).

# Oral Presentation at ICML 2024 Workshop on Models of Human Feedback for AI Alignment

[Video](https://www.bilibili.com/video/BV13daQeuEgb)

This repo is based on the codebase of [INSTINCT](https://github.com/xqlin98/INSTINCT).
## Prepare the data
Please download the data from: https://github.com/xqlin98/INSTINCT/tree/main/Induction/experiments/data/instruction_induction and put it under ./experiments/data/instruction_induction

## Prepare the environment
We use conda to manage our environment. Please install our environment using the following command:
`conda env create -f environment.yml`

## Prepare your OpenAI API key
Add your OpenAI key to `Induction/key`

## Find our running scripts
Our running scripts are in
`experiments/run_dbandits_po.sh`, `experiments/run_dbandits_image_gen.sh` and `experiments/run_dbandits_response.sh`. To run the script
```
cd Induction
bash experiments/run_dbandits_po.sh
```

## Citation
If you find this repo/paper helpful, please consider citing our paper:
```
@article{lin2024prompt,
  title={Prompt Optimization with Human Feedback},
  author={Lin, Xiaoqiang and Dai, Zhongxiang and Verma, Arun and Ng, See-Kiong and Jaillet, Patrick and Low, Bryan Kian Hsiang},
  journal={arXiv preprint arXiv:2405.17346},
  year={2024}
}
```
