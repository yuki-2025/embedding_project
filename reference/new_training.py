from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import pandas as pd
import torch
from tqdm.notebook import tqdm
tqdm.pandas()
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import GenerationConfig
from peft import LoraConfig, get_peft_model 
from evaluate import load
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import joblib
import os
system_path = os.environ.get("PATH")


def evaluate_glue(eval_pred,glue):
        labels = eval_pred.label_ids
        predictions = eval_pred.predictions[0]
        labels[labels == -100] = tokenizer.pad_token_id
        acc = 0.0
        for i in range(len(predictions)):
            if len(predictions[i]) > len(labels[i]):
                preds = np.resize(predictions[i],len(labels[i]))
            elif len(labels[i]) > len(predictions[i]):
                preds = np.concatenate([predictions[i],np.array([tokenizer.pad_token_id]*len(labels[i]))])
            res = glue.compute(predictions=preds, references=labels[i])
            acc = acc+res["accuracy"]
        results = {"glue":acc/(i+1)}
        return results
def evaluate_rouge(eval_pred,rouge):
    labels = eval_pred.label_ids
    predictions = eval_pred.predictions[0]
    predictions[predictions == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels[labels == -100] = tokenizer.pad_token_id
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    results = rouge.compute(predictions=predictions, references=labels,use_aggregator = True)
    return results
def compute_metrics(eval_pred):
    glue_metric = load('glue', 'sst2')
    rouge_metric = load("rouge")
    rouge_res = evaluate_rouge(eval_pred,rouge_metric)
    glue_res = evaluate_glue(eval_pred,glue_metric)
    res = {}
    res.update(rouge_res)
    res.update(glue_res)
    return res
def preprocess_logits_for_metrics(logits, labels):
    #pred_ids = torch.argmax(logits[0], dim=-1)
    return logits, labels

class TrainConversationalModel:
    def __init__(self,model_id,data_path,output_dir):
        self.model_id = model_id
        self.data_path = data_path
        self.output_dir = output_dir
        print("setting_up_model")
        self.bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id,device_map = "cuda:0",quantization_config = self.bnb)
        self.tokenizer =AutoTokenizer.from_pretrained(self.model_id,return_tensors = "pt",padding = True,truncation = True,max_length = 10)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", "lm_head"
                ],
                bias="none",
                task_type="CAUSAL_LM"
            )
            
        self.model = get_peft_model(self.model, self.config)
        print("model setup complete")
        print("loading data and tokenizing")
        df = pd.read_csv(self.data_path)
        texts = df["conversation"]
        self.train,self.val = train_test_split(texts,test_size = 0.2)
        self.train = [self.tokenizer(samp) for samp in tqdm(self.train.tolist())]
        self.val = [self.tokenizer(samp) for samp in tqdm(self.val.tolist())]
        print("tokenization complete, data ready")
        print("setting up trainer")
        self.gen = GenerationConfig.from_pretrained(self.model_id,max_new_tokens = 25)
        self.args = Seq2SeqTrainingArguments(
        output_dir = self.output_dir,
        do_train= True,
        do_eval = True,
        eval_steps = 10000,
        logging_steps = 10000,
        save_steps = 10000,
        eval_strategy = "steps",
        save_strategy = "steps",
        logging_strategy = "steps",
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = 1,
        eval_accumulation_steps = 1,
        learning_rate = 2.5e-4,
        save_total_limit = 3,
        bf16 = True,
        load_best_model_at_end = True,
        num_train_epochs = 3,
        optim="paged_adamw_8bit",
        predict_with_generate = True,
        generation_config = self.gen
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.s2strainer = Seq2SeqTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            args = self.args,
            data_collator = DataCollatorForLanguageModeling(tokenizer = self.tokenizer,mlm = False),
            train_dataset = self.train,
            eval_dataset = self.val,
            compute_metrics = compute_metrics,
            preprocess_logits_for_metrics = preprocess_logits_for_metrics
        )
        print("trainer setup")

if __name__ == "__main__":
    #../Qwen/Qwen2.5-7B-Instruct
    tcm = TrainConversationalModel("/project/aaz/ashmitam/Qwen2/Qwen2Base","/project/aaz/ashmitam/ft_qwen2.5/Cleaned_CEO_Text.csv","./outputs")
    tcm.s2strainer.train()

