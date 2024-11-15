import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str, default='')
parser.add_argument('output_path', type=str, default='')

if __name__ == '__main__':
    args = parser.parse_args()

    token = ""
    base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    # base_model_path = "mistralai/Mistral-7B-Instruct-v0.3"
    model_path = args.model_path
    output_path = args.output_path

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=True, token=token)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = 'right'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_storage=torch.float16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="cuda", token=token
    )

    base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, model_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.merge_and_unload()

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


