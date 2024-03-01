import re
import numpy as np
import evaluate
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import numpy as np
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    AutoModelForCausalLM,
)
from trl import SFTTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import wandb
from random import *
import argparse

torch.cuda.empty_cache()

# default params
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_SYSTEM_PROMPT = """Is the following statement \"true\" or \"false\" ?""".strip()

def generate_training_prompt(claim: str, evidence: str, label: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    if PROMPT_FORMAT == "alpaca":
        prompt =  f"""### Instruction: {system_prompt}

### Input:
{claim.strip()}{evidence.strip()}

### Response:
{label}
""".strip()

    else:
        prompt =  f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>
{claim.strip()}{evidence.strip()} [/INST] {label}
""".strip()
        
    return prompt

def generate_text(data_point):
    if EVIDENCES:
      text = generate_training_prompt(claim=data_point["claim"], evidence=". " + data_point['evidence'], label=str(data_point['label']), system_prompt=DEFAULT_SYSTEM_PROMPT)
    else:
      text = generate_training_prompt(claim=data_point["claim"], evidence="", label=str(data_point['label']), system_prompt=DEFAULT_SYSTEM_PROMPT)
    return {
        #"claim" : data_point['claim'],
        #"evidence" : data_point['evidence'],
        "labels" : data_point["label"],
        "prompt" : text
    }

def process_dataset(data: Dataset):
    return (
        data.shuffle(seed=42)
        .map(generate_text)
        .remove_columns(["claim", "evidence",  "label", "__index_level_0__"])
    )
    
def create_model_and_tokenizer():
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
        token=ACCESS_TOKEN,
        cache_dir=CACHE_DIR
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token = ACCESS_TOKEN, cache_dir=CACHE_DIR)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    #model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer


parser = argparse.ArgumentParser() 
parser.add_argument('--model_name', default='meta-llama/Llama-2-7b-hf', type=str, help='model name')
parser.add_argument('--access_token', default='HF ACCESS TOKEN', type=str, help='llama access token')
parser.add_argument('--data', default='fctr500', type=str, help='which input data')
parser.add_argument('--batch', default=4, type=int, help='batch size')
parser.add_argument('--epoch', default=3, type=int, help='number of epochs')
parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
parser.add_argument('--temperature', default=0.7, type=float, help='temperature')
parser.add_argument('--skip_train', default=False, type=bool, help='skip training')
parser.add_argument('--kshot', default=0, type=int, help='k-shot instances')
parser.add_argument('--prompt', default="alpaca", type=str, help='alpaca or hg')
parser.add_argument('--evidence', default=False, type=bool, help='claim only or with evidence')
parser.add_argument('--cache_dir', default=None, type=str, help='cache_dir')
parser.add_argument('--wandb_key', default="WANDB KEY", type=str, help='wandb api key')
parser.add_argument('--wandb_project', default="llama-fctr", type=str, help='wandb project name')
args = parser.parse_args()


MODEL_NAME = args.model_name
ACCESS_TOKEN = args.access_token
DATA = args.data
BATCH_SIZE = args.batch
N_EPOCHS = args.epoch
LR = args.lr
SKIP_TRAIN = args.skip_train
K_SHOT = args.kshot
CACHE_DIR = args.cache_dir
PROMPT_FORMAT = args.prompt
EVIDENCES = args.evidence


OUTPUT_DIR = "output/" + DATA + "_experiments_" + MODEL_NAME.split("/")[-1] + PROMPT_FORMAT
print("Input args: ", args)
print("output_dir=" + OUTPUT_DIR)

#### DATA
if DATA.startswith("fctr"): # fctr500 or fctr1000
    DEFAULT_SYSTEM_PROMPT = """Is the following statement in Turkish true or false? Answer as either \"doğru\" or \"yanlış\" """.strip()
    
    train_df = pd.read_csv("data/fctr/"+DATA+"_train.csv", sep="\t", encoding='utf-8')
    train_df = train_df[['claim', 'summary', 'label']]
    train_df = train_df.dropna()
    train_df.columns = ['claim', 'evidence', 'label']

    val_df = pd.read_csv("data/fctr/"+DATA+"_val.csv", sep="\t", encoding='utf-8')
    val_df = val_df[['claim', 'summary', 'label']]
    val_df = val_df.dropna()
    val_df.columns = ['claim', 'evidence', 'label']

    test_df = pd.read_csv("data/fctr/"+DATA+"_test.csv", sep="\t", encoding='utf-8')
    test_df = test_df[['claim', 'summary', 'label']]
    test_df = test_df.dropna()
    test_df.columns = ['claim', 'evidence', 'label']

elif DATA == "snopes":
    DEFAULT_SYSTEM_PROMPT = """Is the following statement \"true\" or \"false\" ?""".strip()
    
    train_df = pd.read_csv("data/snopes/snopes_train.csv", sep="\t", encoding="utf-8")
    train_df = train_df[["claim", "evidence", "label"]]
    b = ["true", "false"]
    train_df= train_df[train_df['label'].isin(b)]
    train_df.dropna(inplace=True)

    val_df = pd.read_csv("data/snopes/snopes_train.csv", sep="\t", encoding="utf-8")
    val_df = val_df[["claim", "evidence", "label"]]
    b = ["true", "false"]
    val_df= val_df[val_df['label'].isin(b)]
    val_df.dropna(inplace=True)

    test_df = pd.read_csv("data/snopes/snopes_train.csv", sep="\t", encoding="utf-8")
    val_df = test_df[["claim", "evidence", "label"]]
    b = ["true", "false"]
    test_df = test_df[test_df['label'].isin(b)]
    test_df.dropna(inplace=True)

elif DATA == "snopes_tr":
    DEFAULT_SYSTEM_PROMPT = """Is the following statement in Turkish true or false? Only answer as either \"doğru\" or \"yanlış\" """.strip()
    
    df = pd.read_csv("data/snopes/snopes_tr.csv", sep="\t", encoding="utf-8")
    df = df[["tr_claim", "evidence", "label"]]
    df = df.rename({'tr_claim': 'claim'}, axis=1)
    df.dropna(inplace=True)

    mask = df.applymap(type) != bool
    d = {True: 'doğru', False: 'yanlış'}
    df = df.where(mask, df.replace(d))
    df.replace({'label': {"true": "doğru", "false": "yanlış"}}, inplace=True)

    test_df = pd.read_csv("data/fctr/fctr500_test.csv", sep="\t", encoding='utf-8')
    test_df = test_df[['claim', 'label']]
    test_df = test_df.dropna()
    
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)


elif DATA.startswith("chatgpt"): #chatgpt_fctr500 or chatgpt_fctr1000
    DEFAULT_SYSTEM_PROMPT = """Is the following statement \"true\" or \"false\" ?""".strip()

    train_df = pd.read_csv("data/snopes/snopes_train.csv", sep="\t", encoding="utf-8")
    train_df = train_df[["claim", "evidence", "label"]]
    b = ["true", "false"]
    train_df= train_df[train_df['label'].isin(b)]
    train_df.dropna(inplace=True)

    val_df = pd.read_csv("data/snopes/snopes_train.csv", sep="\t", encoding="utf-8")
    val_df = val_df[["claim", "evidence", "label"]]
    b = ["true", "false"]
    val_df= val_df[val_df['label'].isin(b)]
    val_df.dropna(inplace=True)
    
    test_df = pd.read_csv("data/fctr/" + DATA +".csv", sep="\t", encoding="utf-8")
    test_df = test_df[["chatgpt_claims", "evidence", "label"]]
    test_df.columns = ['claim', 'evidence', 'label']
    test_df.replace({'label': {"doğru": "true", "yanlış": "false"}}, inplace=True)

print(train_df.shape, (val_df.shape, test_df.shape))

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df),
    "test": Dataset.from_pandas(test_df)
    })

dataset["train"] = process_dataset(dataset["train"])
dataset["validation"] = process_dataset(dataset["validation"])
#dataset["test"] = process_dataset(dataset["test"])

#### MODEL
model, tokenizer = create_model_and_tokenizer()
model.config.use_cache = False
model.config.quantization_config.to_dict()

if not SKIP_TRAIN:
    wandb.login(key=args.wandb_key)
    wandb.init(project=args.wandb_project)
    
    lora_r = 16
    lora_alpha = 64
    lora_dropout = 0.1
    lora_target_modules = [
        "q_proj",
        #"up_proj",
        "o_proj",
        "k_proj",
        #"down_proj",
        #"gate_proj",
        "v_proj",
    ]

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # prepare int-8 model for training
    model = prepare_model_for_kbit_training(model)
    # add LoRA adaptor
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_arguments = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    #gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    logging_steps=1,
    learning_rate=LR,
    fp16=True,
#    max_grad_norm=0.3,
    num_train_epochs=N_EPOCHS,
    evaluation_strategy="steps",
    eval_steps=0.2,
    warmup_ratio=0.05,
    save_strategy="steps",
    save_steps = 0.2, 
    group_by_length=True,
    output_dir=OUTPUT_DIR,
    report_to="wandb",
    run_name=DATA + "|" + MODEL_NAME,
    save_safetensors=True,
    lr_scheduler_type="linear",
    seed=42,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="prompt",
        max_seq_length=4096,
        tokenizer=tokenizer,
        args=training_arguments,
    )
    trainer.train()
    trainer.save_model()

    model = trainer.model

### INFERENCE

def generate_few_shot_prompt(k_shot=K_SHOT):
    fs_promt = ""
    if k_shot > 0:
        train_df.reset_index(inplace=True, drop=True)
        r = randint(1, 100)
        true_idx = train_df[train_df.label == "doğru"].sample(n=k_shot, random_state=r).index.values #, random_state=42
        false_idx = train_df[train_df.label == "yanlış"].sample(n=k_shot, random_state=r).index.values #, random_state=42
    
    for i in range(k_shot):
        data_point = train_df.iloc[true_idx[i]]
        fs_promt += generate_training_prompt(claim=data_point["claim"], evidence="", label=data_point["label"], system_prompt=DEFAULT_SYSTEM_PROMPT)
        fs_promt += "\n"
    
        data_point = train_df.iloc[false_idx[i]]
        fs_promt += generate_training_prompt(claim=data_point["claim"], evidence="", label=data_point["label"], system_prompt=DEFAULT_SYSTEM_PROMPT)
        fs_promt += "\n"
    return fs_promt

def generate_prompt(
    claim: str, evidence: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    if PROMPT_FORMAT == "alpaca":
        prompt =  f"""### Instruction: {system_prompt}

### Input:
{claim.strip()}{evidence.strip()}

### Response:
""".strip()

    else:
        prompt =  f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>
{claim.strip()}{evidence.strip()} [/INST]""".strip()
        
    return prompt

examples = []
for data_point in dataset["test"]:
    if EVIDENCES:
      text = generate_prompt(claim=data_point["claim"], evidence=". " + data_point["evidence"], system_prompt=DEFAULT_SYSTEM_PROMPT)
    else:
      text = generate_prompt(claim=data_point["claim"], evidence="", system_prompt=DEFAULT_SYSTEM_PROMPT)
    

    examples.append( {
            "claim" : data_point['claim'],
            "label" : data_point["label"],
            "prompt" : text
        }
    )

test_df = pd.DataFrame(examples)

if SKIP_TRAIN:
    model = PeftModel.from_pretrained(model, OUTPUT_DIR)
    model = model.merge_and_unload()
    model.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

def generateResponse(model, text: str):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    inputs_length = len(inputs["input_ids"][0])
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=10, temperature=args.temperature) # 0.0001
    return tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)


def evaluate(prompt, test_df):
    predictions = []
    references = []
    #First parameter is the replacement, second parameter is your input string
    for i in tqdm(range(test_df.shape[0])):
        row = test_df.iloc[i]
        
        response = generateResponse(model, prompt + "\n" + row.prompt)
        y_pred = response.strip().split("\n")[0]
        y_pred = y_pred.split(" ")[0]
        #y_pred = re.sub("[^a-zA-Z]+", "", y_pred)
        y_pred = re.sub(r'[^\w\s]', '', y_pred)
        try:
            predictions.append(label_dict[y_pred.lower()])
        except:
            isMatched = False
            for key in label_dict.keys():
                if key in y_pred.lower():
                    y_pred = key
                    predictions.append(label_dict[y_pred.lower()])
                    isMatched = True
                    break
            if not isMatched:
                print("Exception!!", y_pred)
                if row["label"] == "doğru":
                    y_pred = "yanlış"
                else:
                    y_pred = "doğru"
                predictions.append(label_dict[y_pred.lower()])

        references.append(label_dict[str(row['label']).lower()])



    print(confusion_matrix(references, predictions))
    print("f1-macro: ", round(f1_score(references, predictions, average='macro'), 4))
    print("f1-weighted: ", round(f1_score(references, predictions, average='weighted'), 4))
    print("f1-binary: ", round(f1_score(references, predictions, average='binary'), 4))
        
    return (round(f1_score(references, predictions, average='macro'), 4), round(f1_score(references, predictions, average='binary'), 4))


label_dict = {"true":0, "false":1, "doğru":0, "yanlış":1}
fs_prompt = ""

if K_SHOT > 0: # in-context
    for k in range(1, 6):
        scores_macro = []
        scores_binary = []
        print("k-shot: ", str(k))
        trial = 5
        for i in range(trial):
            fs_prompt = generate_few_shot_prompt(k)
            macro, binary = evaluate(fs_prompt, test_df)
            scores_macro.append(macro)
            scores_binary.append(binary)
        
        arr = np.array(scores_macro)
        arr_bi = np.array(scores_binary)
        print("avg F1-macro: ", round(np.mean(arr, axis=0),4))
        print("std dev: ", round(np.std(arr, axis=0), 4))
        print("std err: ", round((np.std(arr, axis=0) / trial**0.5),4))
        print("avg F1-binary: ", round(np.mean(arr_bi, axis=0),4))
        print("std dev binary: ", round(np.std(arr_bi, axis=0), 4))
        print("std err binary: ",  round((np.std(arr_bi, axis=0) / trial**0.5),4))
else: # inference after fine-tuning
    scores_macro = []
    scores_binary = []
    trial = 5
    for k in range(trial):
        macro, binary = evaluate(fs_prompt, test_df)
        scores_macro.append(macro)
        scores_binary.append(binary)

    arr = np.array(scores_macro)
    arr_bi = np.array(scores_binary)
    print("avg F1-macro: ", round(np.mean(arr, axis=0),4))
    print("std dev: ", round(np.std(arr, axis=0), 4))
    print("std err: ", round((np.std(arr, axis=0) / trial**0.5),4))
    print("avg F1-binary: ", round(np.mean(arr_bi, axis=0),4))
    print("std dev binary: ", round(np.std(arr_bi, axis=0), 4))
    print("std err binary: ",  round((np.std(arr_bi, axis=0) / trial**0.5),4))
