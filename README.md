# FCTR
Cross-Lingual Learning vs. Low-Resource Fine-Tuning: A Case  Study with Fact-Checking in Turkish


 The rapid spread of misinformation through social media platforms has raised concerns regarding its impact on
 public opinion. While misinformation is prevalent in other languages, the majority of research in this field has
 concentrated on the English language. Hence, there is a scarcity of datasets for other languages, including
 Turkish. To address this concern, we have introduced the FCTR dataset, consisting of 3238 real-world claims. This
 dataset spans multiple domains and incorporates evidence collected from three Turkish fact-checking organizations.
 Additionally, we aim to assess the effectiveness of cross-lingual transfer learning for low-resource languages, with
 a particular focus on Turkish. We demonstrate in-context learning (zero-shot and few-shot) performance of large
 language models in this context. The experimental results indicate that the dataset has the potential to advance
 research in the Turkish language.

## Installation
You need to install the packages listed in "requirements.txt" file to execute the models.  

## Dataset

You can find the FCTR dataset at "data/fctr.csv" and the Snopes dataset at "data/snopes.csv".

The exact data splits that we emplyoed in the experiments are under "data/fctr/" and "data/snopes/" directories. 

## Execution

To fine-tune the Llama models, you should provide the following arguments:

```
python fine_tuning_llama_fc.py \
--model_name "meta-llama/Llama-2-7b-hf" \
--data "fctr500" \
--batch 4 \ 
--epoch 3 \ 
--lr 2e-5 \
--temperature 0.7 \
--skip_train False \
--kshot 0 \
--prompt alpaca \
--evidence False \
--cache_dir None \
--wandb_key "WANDB API KEY" \
--wandb_project "llama-fctr" \
```

or to perform in-context learning (zero-shot or few-shot) with the Llama models, you should provide the following arguments:

```
python fine_tuning_llama_fc.py \
--model_name "meta-llama/Llama-2-7b-hf" \
--access_token "HF ACCESS TOKEN" \
--data "fctr500" \
--temperature 0.7 \
--skip_train True \
--kshot 1 \
--prompt alpaca \
--evidence False \
--cache_dir None \
```

## Citation
Please cite the paper as follows if you find the study useful.
```
@misc{cekinel2024crosslingual,
      title={Cross-Lingual Learning vs. Low-Resource Fine-Tuning: A Case Study with Fact-Checking in Turkish}, 
      author={Recep Firat Cekinel and Pinar Karagoz and Cagri Coltekin},
      year={2024},
      eprint={2403.00411},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
