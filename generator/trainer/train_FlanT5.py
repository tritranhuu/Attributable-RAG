import nltk
import numpy as np
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, BitsAndBytesConfig
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForSeq2SeqLM
from peft import LoraConfig, TaskType, get_peft_model
from accelerate import Accelerator


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

model_path = "google/flan-t5-xl"
train_data_path = "./rag_cite_data_T5_full.jsonl"
output_dir = "./t5_xl_full_finetune"
logging_dir = "./logging_test"

accelerator = Accelerator()
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=32,
    lora_alpha=32, lora_dropout=0.05,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_storage=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path,
                                                   quantization_config=bnb_config,
                                                   torch_dtype=torch.float16,
                                                   )
model.add_adapter(peft_config)

for param in model.parameters():
    if param.requires_grad:
        param.data = param.data.float()

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def preprocess_function(examples):
    inputs = [doc for doc in examples["inputs"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["labels"],
                       max_length=256,
                       truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_dataset = load_dataset('json', data_files=train_data_path, split='train')
train_dataset = train_dataset.shuffle()
tokenized_dataset = train_dataset.map(preprocess_function, batched=True)

print("Setting up trainer")
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    weight_decay=0.01,
    save_total_limit=5,
    num_train_epochs=5,
    warmup_steps=100,
    predict_with_generate=True,
    push_to_hub=False,
    logging_dir='./t5_logs',
    logging_steps=100,
    save_strategy="steps",
    save_steps=1000,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()
