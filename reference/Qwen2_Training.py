 

# print("torch version = ", torch.__version__); 
# print("torch cuda = ",torch.cuda.is_available())
import os
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig,pipeline
import pandas as pd
import torch
from tqdm.notebook import tqdm
tqdm.pandas()
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import GenerationConfig
from peft import LoraConfig, get_peft_model,PeftModel,PeftModelForCausalLM
from evaluate import load
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import joblib
from sklearn.decomposition import PCA
import gc

def evaluate_glue(eval_pred,glue):
    """
    Evaluate the Glue Score.
    eval_pred: the predictions from the model.
    glue: glue score instance.
    """
    tokenizer = AutoTokenizer.from_pretrained("models/Llama-3.1-8B",return_tensors = "pt",padding = True,truncation = True,max_length = 10)
    labels = eval_pred.label_ids
    predictions = eval_pred.predictions[0]
    labels[labels == -100] = tokenizer.pad_token_id
    acc = 0.0
    for i in range(len(predictions)):
        if len(predictions[i]) > len(labels[i]):
                preds = np.resize(predictions[i],len(labels[i]))
        elif len(labels[i]) > len(predictions[i]):
                preds = np.concatenate([predictions[i],np.array([tokenizer.pad_token_id]*len(labels[i]))])
        else:
                preds = predictions[i]
        res = glue.compute(predictions=preds, references=labels[i])
        acc = acc+res["accuracy"]
    results = {"glue":acc/(i+1)}
    return results
def evaluate_rouge(eval_pred,rouge):
    """
    Evaluate the Rouge Score.
    eval_pred: the predictions from the model.
    rouge: rouge score instance.
    """
    tokenizer = AutoTokenizer.from_pretrained("models/Llama-3.1-8B",return_tensors = "pt",padding = True,truncation = True,max_length = 10)
    labels = eval_pred.label_ids
    predictions = eval_pred.predictions[0]
    predictions[predictions == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels[labels == -100] = tokenizer.pad_token_id
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    results = rouge.compute(predictions=predictions, references=labels,use_aggregator = True)
    return results
def compute_metrics(eval_pred):
    """ 
    Evaluation Function for trainer.
    eval_pred: predictions from the model.
    """
    glue_metric = load('glue', 'sst2')
    rouge_metric = load("rouge")
    rouge_res = evaluate_rouge(eval_pred,rouge_metric)
    glue_res = evaluate_glue(eval_pred,glue_metric)
    res = {}
    res.update(rouge_res)
    res.update(glue_res)
    return res
def preprocess_logits_for_metrics(logits, labels):
    """
    Preprocess Logits for the compute_metrics function
    logits: logits from the model.
    labels: ground truth for logits.
    """
    #pred_ids = torch.argmax(logits[0], dim=-1)
    return logits, labels

class TrainConversationalModel:
    def __init__(self,model_id,data_path,output_dir):
        """
        Initialization Function for the class TrainConversatonalModel, initializes data, model, tokenizer and trainer
        model_id: model path on huggingface or local directory, should contain model, tokenizer and generation_config
        data_path: data path for the texts to be loaded, should be in CSV format
        output_dir: Output Directory for Trainer
        """
        self.model_id = model_id
        self.data_path = data_path
        self.output_dir = output_dir
        print("setting_up_model")
        self.bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_id,device_map = "cuda:0",quantization_config = self.bnb)
        self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto",  # Let the library decide optimal placement
                    quantization_config=self.bnb,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True
                    # offload_folder="offload",
                )
        print("model loaded")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id,return_tensors = "pt",padding = True,truncation = True,max_length = 10)
        print("tokenizer loaded")
        # Before applying PEFT
        gc.collect()
        torch.cuda.empty_cache()
        print(f"GPU memory before PEFT: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

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
        print(f"GPU memory after PEFT: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print("model setup complete")
        print("loading data and tokenizing")
        df = pd.read_csv(self.data_path)
        texts = df["conversation"]
        self.train,self.val = train_test_split(texts,test_size = 0.2)
        self.train = [self.tokenizer(samp[:50]) for samp in tqdm(self.train.tolist())]
        self.val = [self.tokenizer(samp[:50]) for samp in tqdm(self.val.tolist())]
        print("tokenization complete, data ready")
        print("setting up trainer")
        self.gen = GenerationConfig.from_pretrained(self.model_id,max_new_tokens = 1)
        self.args = Seq2SeqTrainingArguments(
        output_dir = self.output_dir,
        do_train= True,
        do_eval = True,
         # 减少评估和保存的频率
        eval_steps = 50,
        logging_steps = 10,
        save_steps = 100,
        eval_strategy = "steps",
        save_strategy = "steps",
        logging_strategy = "steps",
            # 增加批量大小
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
         # 增加梯度累积
        gradient_accumulation_steps = 8,
        eval_accumulation_steps = 4,
        learning_rate = 2.5e-4,
        save_total_limit = 3,
        bf16 = True,
        load_best_model_at_end = True,
        num_train_epochs = 3,
        optim="paged_adamw_8bit",
        predict_with_generate = True,
        generation_config = self.gen,
        # 添加新的优化参数
        fp16_full_eval = True,        # 使用FP16进行评估
        dataloader_num_workers = 4,   # 增加数据加载的工作线程
        group_by_length = True,       # 按长度分组以减少填充
        gradient_checkpointing = True,# 启用梯度检查点
        report_to = "none",           # 禁用报告以提高速度
        no_cuda = False,              # 确保使用CUDA
        use_cpu = False,              # 不使用CPU训练
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.s2strainer = Seq2SeqTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            args = self.args,
            data_collator = DataCollatorForLanguageModeling(tokenizer = self.tokenizer,mlm = False),
            train_dataset = self.train,
            eval_dataset = self.val,
            # compute_metrics = compute_metrics,
            # preprocess_logits_for_metrics = preprocess_logits_for_metrics
        )
        print("trainer setup")
    def start_training(self):
        """
        Start Training the model.
        """
        self.s2strainer.train()
    def load_best_checkpoint(self,checkpoint):
        """
        Load best checkpoint for peft model.
        checkpoint: directory with the considered checkpoint.
        """
        self.best_checkpoint = PeftModelForCausalLM.from_pretrained(model_id = checkpoint,device_map = "auto" ,quantization_config = self.bnb,model = self.model)
        self.best_checkpoint = self.best_checkpoint.merge_and_unload()
        self.best_checkpoint.eval()
    def embed_documents(self,checkpoint):
        """
        Embed Documents using Checkpoint.
        checkpoint: directory with the considered checkpoint.
        """
        self.load_best_checkpoint(checkpoint)
        df = pd.read_csv(self.data_path)
        texts = df["conversation"]
        embeddings = []
        for text in tqdm(texts):
            with torch.no_grad():
                embedding = self.best_checkpoint(self.tokenizer(text[:10],return_tensors = "pt").input_ids, output_hidden_states=True)
                embeddings.append(embedding.hidden_states[-1].mean(1).detach().cpu().numpy().tolist())
        self.pca = self.fit_pca(pd.Series(embeddings).explode().tolist())
        self.pca_transform(pd.Series(embeddings).apply(lambda x: x[0]).tolist()).to_csv("qwen_embeddings.csv",index = False)
    def fit_pca(self,embeds):
        """
        Fit PCA to reduce embedding dimensions
        embeds: embeddings to fit the pca on.
        """
        dims = [768,512,256,128,64]
        dim_ratios = []
        for dim in dims:
            if dim < len(embeds) and dim < len(embeds[0]):
                pca = PCA(n_components = dim)
                pca.fit(embeds)
                dim_ratios.append(pca.explained_variance_ratio_[0])
        best_value = -10000
        for i in range(len(dim_ratios)):
            if dims[i] < len(embeds) and dims[i] < len(embeds[0]):
                if dim_ratios[i] - 0.95 < best_value and (dim_ratios[i] - 0.95) > 0:
                    best_value = i
        self.pca = PCA(n_components = dims[i])
        self.pca.fit(embeds)
        return self.pca
    def pca_transform(self,embeds):
        """
        transform documents with fitted PCA.
        embeds: embeddings to fit the pca on.
        """
        return pd.DataFrame(self.pca.transform(embeds))
if __name__ == "__main__":
    path = "/project/aaz/leo/"
    model_path = os.path.join(path,"models/Llama-3.1-8B")
    data_path = os.path.join(path,"dataset/Cleaned_CEO_Text.csv")
    output_dir = "./outputs"
    tcm = TrainConversationalModel(model_path,data_path,output_dir)
    # tcm.start_training()
    tcm.embed_documents(checkpoint = "./outputs/checkpoint-511")
