from accelerate import FullyShardedDataParallelPlugin, Accelerator
# from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from trl import SFTTrainer
from transformers.utils import quantization_config
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# Load dataset
print("Loading data")
token = ""
model_path = "mistralai/Mistral-7B-Instruct-v0.3"
train_data_path = "./rag_cite_data_negative_mistral.jsonl"
output_dir = "./rag_mistral_neg"
logging_dir = "./logging_test"

train_dataset = load_dataset('json', data_files=train_data_path, split='train')
train_dataset = train_dataset.shuffle()

#eval_dataset = load_dataset('json', data_files=eval_data_path, split='train')

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    padding_side="right",
    token=token
)
tokenizer.add_special_tokens({"pad_token":"[PAD]"})
# Load Base Model
print("Loading Model")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             quantization_config=bnb_config,
                                             # torch_dtype=torch.float16,
                                             # load_in_8bit=True,
                                             token=token)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id


#model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=64,
    lora_alpha=32,
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)
# model = get_peft_model(model, config)

print("Setting up trainer")
arguments = TrainingArguments(
    output_dir=output_dir,
    warmup_steps=100,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_checkpointing=True,
    #max_steps=100000,
    num_train_epochs=5,
#    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    weight_decay=0.001,
#    warmup_ratio=0.01,
    group_by_length=True,
    max_grad_norm=0.3,
    logging_steps=25,
    # fp16=True,
    optim="paged_adamw_32bit",
    logging_dir=logging_dir,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=10,
    #evaluation_strategy="steps",
    #eval_steps=1000,
)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    #eval_dataset=eval_dataset,
    args=arguments,
    max_seq_length=2048,
    #data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    tokenizer=tokenizer,
    peft_config=config,
    dataset_text_field="text",
)

trainer.train()
