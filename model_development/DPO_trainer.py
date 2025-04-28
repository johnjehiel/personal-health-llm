from unsloth import FastLanguageModel
import torch

max_seq_length = 2048 # supports RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "johnjehiel/personal-health-LLM-Llama-3.2-1B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # 0 is optimized
    bias = "none",    # "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # supports rank stabilized LoRA
    loftq_config = None,
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

def format_prompt(sample):
    instruction = "You are an AI assistant. You will be given a task. You must generate a correct answer."
    input       = sample["prompt"]
    accepted    = sample["chosen"]
    rejected    = sample["rejected"]

    sample["prompt"]   = alpaca_prompt.format(instruction, input, "")
    sample["chosen"]   = accepted + EOS_TOKEN
    sample["rejected"] = rejected + EOS_TOKEN
    return sample

# Load dataset
from datasets import load_dataset
dataset = load_dataset("johnjehiel/ph-llm-DPO")["train"]
dataset = dataset.rename_column("question", "prompt")
dataset = dataset.map(format_prompt,)

row = dataset[1]
print('INSTRUCTION: ' + '=' * 50)
print(row["prompt"])
print('ACCEPTED: ' + '=' * 50)
print(row["chosen"])
print('REJECTED: ' + '=' * 50)
print(row["rejected"])

# Enable reward modelling stats
from unsloth import PatchDPOTrainer
PatchDPOTrainer()
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig
from unsloth import is_bfloat16_supported

dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = DPOConfig(
        per_device_train_batch_size = 2, #2
        gradient_accumulation_steps = 4, #4
        gradient_checkpointing=True, 
        warmup_ratio = 0.05,
        num_train_epochs = 1,
        learning_rate = 5e-6,
        save_steps=5000,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 100,
        optim = "adamw_8bit",
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "./outputs/DPO/1B",
        report_to = "none", # Use this for WandB etc
    ),
    beta = 0.1,
    train_dataset = dataset,
    tokenizer = tokenizer,
    max_length = 1024,
    max_prompt_length = 512,
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Train the model
dpo_trainer.train()

# Run inference
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "You are a personal health Assistant. Answer the following question correctly", # instruction
        "Provide tips to reduce belly fat.", # input
        "", # output - leave this blank for generation
    )
], return_tensors = "pt").to("cuda")

# Generate with streaming
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

# Save Adapters
model.save_pretrained("personal-health-LLM-Llama-3.2-1B-Instruct-DPO")
tokenizer.save_pretrained("personal-health-LLM-Llama-3.2-1B-Instruct-DPO")

# Save merged model
# model.save_pretrained_merged("personal-health-LLM-Llama-3.2-1B-Instruct-DPO-merged-16bit", tokenizer, save_method = "merged_16bit",)

# model.push_to_hub_merged("personal-health-LLM-Llama-3.2-1B-Instruct-DPO-merged-16bit", tokenizer, save_method = "merged_16bit",)

# model.push_to_hub_merged("personal-health-LLM-Llama-3.2-1B-Instruct-DPO", tokenizer, save_method = "lora",)