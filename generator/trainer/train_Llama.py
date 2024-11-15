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
model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
train_data_path = "./rag_cite_data_llama.jsonl"

output_dir = "./rag_llama_ft"
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
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_storage=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             quantization_config=bnb_config,
                                             torch_dtype=torch.float16,
                                             token=token)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id


#model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj"
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)
model.add_adapter(config)
for param in model.parameters():
    if param.requires_grad:
        param.data = param.data.float()

print("Setting up trainer")
arguments = TrainingArguments(
    output_dir=output_dir,
    warmup_steps=100,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_checkpointing=True,
    num_train_epochs=5,
    learning_rate=2e-4,
    weight_decay=0.001,
    group_by_length=True,
    max_grad_norm=0.3,
    logging_steps=25,
    fp16=True,
    optim="paged_adamw_32bit",
    logging_dir=logging_dir,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=5,
)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=arguments,
    max_seq_length=2200,
    tokenizer=tokenizer,
    dataset_text_field="text",
)

trainer.train()
