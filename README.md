# Prompt Tuning on Vision Language Model

## Introduction
Our proposed method makes BEiT3 more capable of writing human-like sentences using Prompt Tuning on Vision Language Model.

**Keywords:** VLM, Prompt Tuning.

## Configurations

<p align="left">
 <a href=""><img src="https://img.shields.io/badge/python-3.9-aff.svg"></a>
</p>

### Run locally
- Create conda environment, note that python version should be <span style="color:#9BB8ED;">Python 3.9</span>
```
conda create --name beit3-prompt-tuning python=3.9
conda activate beit3-prompt-tuning
```

- Install required packages

```
pip install -r requirements.txt --no-cache-dir
```

## Traning/Inference
Please follow the instructions at [Instruction from BEiT3](./get_started_for_captioning.md)