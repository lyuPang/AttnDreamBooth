# AttnDreamBooth

Official Implementation of **"AttnDreamBooth: Towards Text-Aligned Personalized Text-to-Image"** by Lianyu Pang, Jian Yin, Baoquan Zhao, Qing Li and Xudong Mao.

## Abstract
>Recent advances in text-to-image models have enabled high-quality personalized image synthesis of user-provided concepts with flexible textual control. In this work, we analyze the limitations of two primary techniques in text-to-image personalization: Textual Inversion and DreamBooth. When integrating the learned concept into new prompts, Textual Inversion tends to overfit the concept, while DreamBooth often overlooks it. We attribute these issues to the incorrect learning of the embedding alignment for the concept. We introduce AttnDreamBooth, a novel approach that addresses these issues by separately learning the embedding alignment, the attention map, and the subject identity in different training stages. We also introduce a cross-attention map regularization term to enhance the learning of the attention map. Our method demonstrates significant improvements in identity preservation and text alignment compared to the baseline methods. Code will be made publicly available.

<img src='assets/teaser.jpg'>
<!-- <a href="https://arxiv.org/abs/2312.15905"><img src="https://img.shields.io/badge/arXiv-2312.15905-b31b1b.svg" height=20.5></a> -->

## Setup
Our code is primarily based on [Diffusers-DreamBooth](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) and relies on the [diffusers](https://github.com/huggingface/diffusers) library.
### Set up the Environment
To set up the environment, run the following commands:
```bash
conda env create -f environment.yaml
conda activate AttnDreamBooth
```
### Initialize Accelerate
Initialize an [Accelerate](https://github.com/huggingface/accelerate/) environment with:
```bash
accelerate config
```

### Logging into Huggingface
To use `stabilityai/stable-diffusion-2-1-base` model, you may have to log into Huggingface as follows

+ Use `huggingface-cli` to log in via the Terminal
+ Input your token extracted from [Token](https://huggingface.co/settings/tokens)

## Download
### Image Dataset
Our datasets were originally collected and are provided by [Textual Inversion](https://github.com/rinongal/textual_inversion) and [DreamBooth](https://github.com/google/dreambooth). You can find all datasets used in our paper from [here](https://drive.google.com/drive/folders/1lLrG95EH3pmwlYkKBRJ_q3mZtI7I1QOo?usp=sharing).

### Pretrained Checkpoints
We provide pretrained checkpoints for two objects. You can download the sample images and their corresponding pretrained checkpoints.

|Concepts|Samples|Models|
|---|---|---|
|child doll|[images](https://drive.google.com/drive/folders/1QW0bS-mT4ICn3PkIvyziQsCmYhA8ynxZ?usp=sharing)|[model](https://drive.google.com/drive/folders/1VcvjBFF_0HF1xKNtY76LtFG-qb5-uW7h?usp=sharing)|
|grey sloth|[images](https://drive.google.com/drive/folders/1J7_buXn1y6uopHZp8GTxTxTl5uu_fuYJ?usp=sharing)|[model](https://drive.google.com/drive/folders/1EoLlAzMEvIiamG9FJstWuI9GgrzQIGfH?usp=sharing)|

## Usage

### Training
You can run the `bash_script/train_attndreambooth.sh` script to train your own model. Before executing the training command, ensure that you have configured the following parameters in `train_attndreambooth.sh`:
+ Line **6**: `output_dir`. This is the directory where the fine-tuned model will be saved.
+ Line **8**: `instance_dir`. This is the directory containing the images of the target concept.
+ Line **10**: `category`. This is the category of the target concept.

For example, to train the concept `child doll` in the [images](#pretrained-checkpoints), you need to set the parameters as follows.
```bash
output_dir="./models/"
instance_dir="./dataset/child_doll"
category="doll"
```
To run the training script, use the following command.
```bash
bash bash_script/train_attndreambooth.sh
```
**Notes**:
+ All training arguments can be found in `train_attndreambooth.sh` and are set to their defaults according to the official paper.
+ Please refer to `train_attndreambooth.sh` and `train_attndreambooth.py` for more details on all parameters.

### Inference
You can run the `bash_script/inference.sh` script to generate images. Before executing the inference command, ensure that you have configured the following parameters:
+ Line **2**: `learned_embedding_path`. This is the path to the embeddings learned in the first stage.
+ Line **4**: `checkpoint_path`. This is the path to the fine-tuned models trained in the third stage.
+ Line **6**: `category`. This is the category of the target concept.
+ Line **8**: `output_dir`. This is the directory where the generated images will be saved.

To run the inference, use the following command.
```bash
bash bash_script/inference.sh
```
**Notes**:
+ If you did not set `--only_save_checkpoints` during the training phase, you can specify `--pretrained_model_name_or_path` as the path to the full model, and then omit `--checkpoint_path`.
+ We offer learned embeddings and models for two objects [here](https://drive.google.com/drive/folders/10XFEjFm22jTHuFUx36Cq8MVXYf6ouQhv?usp=sharing) for direct experimentation.
+ For convenience, you can either specify a path of a text file with `--prompt_file`, where each line contains a prompt. For example:
```
A photo of a {}
A {} floats on the water
A {} latte art
```
+ Specify the concept using `{}`, and we will replace it with the concept’s placeholder token and the specified category.
+ The resulting images will be saved in the directory `{save_dir}/{prompt}`

+ For detailed information on all parameters, please consult `inference.py` and `inference.sh`.

## Metrics
We use the same evaluation protocol as used in [Textual Inversion](https://github.com/rinongal/textual_inversion).

## Results of Our Method

<img src='assets/results.jpg'>

## Acknowledgements
Our code mainly bases on [Diffusers-DreamBooth](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth). A huge thank you to the authors for their valuable contributions.

## References

```

```