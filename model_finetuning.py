import os
HF_TOKEN = os.environ.get("HF_TOKEN")

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

max_seq_length = 2048
dtype = None 
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = HF_TOKEN,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # 0 is optimized
    bias = "none",    # "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

prompt = """Below is an instruction that describes a task, paired with an optional input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["Prompt"]
    # Create an empty input field
    inputs = [""] * len(instructions)
    outputs = examples["Response"]
    texts = []
    for instruction, inp, output in zip(instructions, inputs, outputs):
        # Format the text using the prompt template and append the EOS token.
        text = prompt.format(instruction, inp, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}



ds = load_dataset("johnjehiel/ph-llm-dataset-batch-splits")
dataset = ds.map(formatting_prompts_func, batched=True)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        
        # for full training runs
        num_train_epochs=3,
        warmup_ratio=0.03,
        gradient_checkpointing=True,              # use gradient checkpointing to save memory
        save_steps=5000,
        max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
        group_by_length=True, # group batches of similar length together
        eval_strategy="epoch",
        
        # warmup_steps = 5,
        # max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 200, # increase for training runs
        optim = "adamw_8bit",
        weight_decay = 0.001, # 0.001 for training runs
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "./outputs/1B",
        report_to = "none", 
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

model.save_pretrained("PH-LLM-Llama-3.2-1B-Instruct")
tokenizer.save_pretrained("PH-LLM-Llama-3.2-1B-Instruct")

# Push only the adapters
model.push_to_hub("johnjehiel/PH-LLM-Llama-3.2-1B-Instruct", token=HF_TOKEN)
tokenizer.push_to_hub("johnjehiel/PH-LLM-Llama-3.2-1B-Instruct", token=HF_TOKEN)

# MERGE AND PUSH ONLY AFTER THE MODEL IS FINE-TUNED WITH ALL 5 BATCHES OF TRAINING DATA
new_model_online = "johnjehiel/PH-LLM-Llama-3.2-1B-Instruct"
model.push_to_hub_merged(
    new_model_online, 
    tokenizer, 
    save_method="merged_4bit_forced",  # Use "merged_4bit" if you prefer 4-bit precision
    commit_message="Merged base model and LoRA adapters for fine-tuned Llama 3.2 1B Instruct",
    token=HF_TOKEN
)

loaded_model, loaded_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "johnjehiel/PH-LLM-Llama-3.2-1B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = HF_TOKEN
)

FastLanguageModel.for_inference(loaded_model) # Enable native 2x faster inference
inputs = loaded_tokenizer(
[
    prompt.format(
        # dataset_test["Prompt"][601], # instruction
        "provide tips to loose belly fat. List them in ordered and structured manner",
        "", # input
        "", # output - leave this blank for generation
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(loaded_tokenizer, skip_prompt = True)
_ = loaded_model.generate(input_ids = inputs.input_ids, attention_mask = inputs.attention_mask,
                   streamer = text_streamer, max_new_tokens = 512, pad_token_id = loaded_tokenizer.eos_token_id)
