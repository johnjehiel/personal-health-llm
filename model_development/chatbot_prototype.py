from unsloth import FastLanguageModel
import torch
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_seq_length = 2048
dtype = None 
load_in_4bit = False

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "johnjehiel/personal-health-LLM-Llama-3.2-1B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = HF_TOKEN
)
FastLanguageModel.for_inference(model)

# Define the system prompt and prompt template for general conversation.
system_prompt = """You are a highly knowledgeable and personal health assistant with expert-level medical knowledge and extensive research experience in the field of healthcare and medicine. You are designed to engage in conversational dialogues and provide precise, evidence-based medical advice and personalized recommendations on a wide range of health topics (e.g., nutrition, fitness, sleep hygiene, mental health, stress management, chronic disease management, and preventative care).

Guidelines:
- Respond only as an assistant based on user inputs.
- Do not fabricate user queries or simulate conversations.
- Provide detailed, expert-level advice and recommendations without including extraneous greetings or sign-offs.
- Focus on clear, actionable, and concise guidance, strictly based on verified, up-to-date, and reputable medical information.
- Ensure that all advice adheres to current clinical guidelines and best practices.
- Maintain a neutral, formal, and professional tone throughout the conversation.
"""

chatbot_prompt_template = """Below is an instruction that describes a task, paired with an optional input that provides further context and a summary of previous conversations. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{health_report}
The summary of previous conversations is as follows:
{summary}

### Response: 
"""

health_metrics_prompt_template = """Below is an instruction that describes a task, paired with an optional input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response: 
{response}"""

# Sleep prompts.
sleep_prompt_1 = '''You are a sleep medicine expert. You are given the following sleep data.
The user is {gender}, {age} years old.
Sleep Summary: 
Bedtime: {Bedtime}
Wakeup time: {Wakeup time}
Sleep duration: {Sleep duration}
Sleep efficiency: {Sleep efficiency}
REM sleep percentage: {REM sleep percentage}
Deep sleep percentage: {Deep sleep percentage}
Light sleep percentage: {Light sleep percentage}
Awakenings: {Awakenings}
Caffeine consumption: {Caffeine consumption}
Alcohol consumption: {Alcohol consumption}
Smoking status: {Smoking status}
Exercise frequency: {Exercise frequency}

List the most important insights. Identify all of the patterns of data that are likely out of the preferred range. Make sure to consider various sleep health dimensions: Routine, Sleep Quality, Alertness, Timing, Efficiency, and Duration. Add a heading for each dimension. Optionally (only do this if extremely important) add a heading called Other for anything else that doesn't fit the above categories. For Routine, consider the average bedtime, wake time, midsleep point and standard deviations of these, focus on the consistency of the routine, not timing. For Sleep Quality, consider light sleep duration, deep sleep duration, REM sleep duration, sleep score, restlessness score, time to quality sleep, and wake time after sleep onset. For Alertness, consider the number of naps and nap length. For Timing, consider midsleep point, bedtime, wake time, make any comments on weekend vs. workday. For Efficiency, consider sleep efficiency, wake time after sleep onset, and time to quality sleep, describe how they compare to similar users. For Duration, consider average sleep duration, weekend vs. workday sleep durations and standard deviations, describe how they compare to similar users. When determining whether a metric is normal or abnormal, always provide the corresponding percentile. Avoid generic statements. Avoid incorrect knowledge, inconsistencies and contradictions. Don't mention "the user". Talk like you're speaking directly to someone. Be concise.
# Sleep insights report
'''

sleep_prompt_2 = '''You are a sleep medicine expert. You are given the following sleep data. 
The user is {gender}, {age} years old. 
Sleep Summary:
Bedtime: {Bedtime}
Wakeup time: {Wakeup time}
Sleep duration: {Sleep duration}
Sleep efficiency: {Sleep efficiency}
REM sleep percentage: {REM sleep percentage}
Deep sleep percentage: {Deep sleep percentage}
Light sleep percentage: {Light sleep percentage}
Awakenings: {Awakenings}
Caffeine consumption: {Caffeine consumption}
Alcohol consumption: {Alcohol consumption}
Smoking status: {Smoking status}
Exercise frequency: {Exercise frequency}

Based on the data, we can get the following insights: 
{insights_response}

What are the underlying causes? Make sure to consider the following causes: Circadian rhythm, Homeostatic drive, Psychophysiologic hyperarousal, and Extrinsic factors. Order the causes from most to least relevant. Identify the likelihood of the causes (e.g. unlikely, possible, very likely). Cite relevant data and insights, for example, "consistently low sleep efficiency despite normal sleep durations suggests low homeostatic drive". Avoid diagnosing health conditions. Avoid providing recommendations. Avoid generic statements. Avoid incorrect knowledge, inconsistencies and contradictions. Don't mention "the user". Talk like you're speaking directly to someone. Be concise. 
# Causes report
'''

sleep_prompt_3 = '''You are a sleep medicine expert. You are given the following sleep data. 
The user is {gender}, {age} years old. 
Sleep Summary: 
Bedtime: {Bedtime}
Wakeup time: {Wakeup time}
Sleep duration: {Sleep duration}
Sleep efficiency: {Sleep efficiency}
REM sleep percentage: {REM sleep percentage}
Deep sleep percentage: {Deep sleep percentage}
Light sleep percentage: {Light sleep percentage}
Awakenings: {Awakenings}
Caffeine consumption: {Caffeine consumption}
Alcohol consumption: {Alcohol consumption}
Smoking status: {Smoking status}
Exercise frequency: {Exercise frequency}

Based on the data, we can get the following insights: 
{insights_response} 
Causes: 
{etiology_response} 

What recommendation(s) can you provide to help this user improve their sleep? Tie recommendations to the very likely and possible causes, for example, "Recommendations to address Circadian rhythm". Tie recommendations to user's sleep data such as average bedtime, average wake time, and number of naps, and recommend a goal bedtime and wake time based on their data. The recommendations should be time-bound, for example for the next week or the next month. Write one short question to ask the user in order to better understand their sleep. Avoid assumptions regarding the trainee's lifestyle or behavioral choices. Avoid generic statements. Avoid incorrect knowledge, inconsistencies and contradictions. Don't mention "the user". Talk like you're speaking directly to someone. Be concise. 
# Recommendations report
'''

# Function to generate model response from a given prompt text.
def generate_response(prompt_text, max_tokens):
    inputs = tokenizer([prompt_text], return_tensors="pt").to(device)
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        # do_sample=True,
        temperature=0.7,
        repetition_penalty=1.05
    )
    full_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return full_response.split("### Response: \n")[-1]

# Function to parse health metrics data from the user input.
def parse_health_metrics(input_text):
    # Expecting each metric on a separate line in "Metric: Value" format.
    metrics = {}
    for line in input_text.strip().split('|'):
        if '=' in line:
            key, value = line.split("=", 1)
            metrics[key.strip()] = value.strip()
    return metrics

# Initialize conversation history for generic conversation.
conversation_history = ""
user_reports = ""
current_conversation = ""
assistant_response = ""

# Initialize the summarization pipeline.
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

print("Welcome, I am your Personal Health Assistant. Type 'quit' or 'exit' to end the session.")
print("To run the health metrics pipeline, please attach your health metrics data (|Metric1=Value1|Metric2=Value2|...)\n")

while True:
    user_input = input("User: ").strip()
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Session ended.")
        break
    
    # Check if the input appears to be health metrics data
    elif "[HEALTH METRICS]" in user_input:
        # Parse the health metrics data.
        health_data = parse_health_metrics(user_input)
        # Check for required keys; if missing, notify the user.
        required_keys = ["gender", "age", "Bedtime", "Wakeup time", "Sleep duration", "Sleep efficiency",
                         "REM sleep percentage", "Deep sleep percentage", "Light sleep percentage",
                         "Awakenings", "Caffeine consumption", "Alcohol consumption", "Smoking status", "Exercise frequency"]
        missing = [key for key in required_keys if key not in health_data]
        if missing:
            print(f"Missing required data for: {', '.join(missing)}. Please provide all required fields.")
            continue

        print(f"\nAssistant:")
        
        # Generate Insights
        prompt1 = sleep_prompt_1.format(**health_data)
        prompt1 = health_metrics_prompt_template.format(
            instruction=prompt1,
            input="",
            response=""
        )
        print("\n--- Generating Sleep Insights Report ---\n")
        insights_response = generate_response(prompt1, max_tokens=1024)
        print("Sleep Insights Report:\n", insights_response, "\n")
        
        # Generate Etiology/Causes Report
        prompt2 = sleep_prompt_2.format(insights_response=insights_response, **health_data)
        prompt2 = health_metrics_prompt_template.format(
            instruction=prompt2,
            input="",
            response=""
        )
        print("\n--- Generating Underlying Causes Report ---\n")
        etiology_response = generate_response(prompt2, max_tokens=1024)
        print("Causes Report:\n", etiology_response, "\n")
        
        # Generate Recommendations Report
        prompt3 = sleep_prompt_3.format(insights_response=insights_response, etiology_response=etiology_response, **health_data)
        prompt3 = health_metrics_prompt_template.format(
            instruction=prompt3,
            input="",
            response=""
        )
        print("\n--- Generating Sleep Recommendations Report ---\n")
        recommendations_response = generate_response(prompt3, max_tokens=1024)
        print("Recommendations Report:\n", recommendations_response, "\n")

        insights_reports = f"# Sleep Insights Report:\n\n{insights_response}"
        insights_reports = summarizer(insights_reports, max_length=200, min_length=150, do_sample=False)[0]['summary_text']
        # print(insights_reports)

        cause_reports = f"# Sleep Causes Report:\n\n{etiology_response}"
        cause_reports = summarizer(cause_reports, max_length=200, min_length=150, do_sample=False)[0]['summary_text']
        # print(cause_reports)

        recommendation_reports = f"# Sleep Recommendations Report:\n\n{recommendations_response}"
        recommendation_reports = summarizer(recommendation_reports, max_length=200, min_length=150, do_sample=False)[0]['summary_text']
        # print(recommendation_reports)
        user_reports = f"{insights_reports}\n\n{cause_reports}\n\n{recommendation_reports}"

    # print conversation history
    elif user_input.lower() == 'c':
            print("-"*20)
            print(f"Conversation history: {conversation_history}")
            print("-"*20)
            continue
    
    # print user reports
    elif user_input.lower() == 'u':
            print("-"*20)
            print(f"User Reports: \n{user_reports}")
            print("-"*20)
            continue
    
    # regenerate response for the same user input
    elif user_input.lower() == 'r':
        prompt_input = chatbot_prompt_template.format(
            instruction=system_prompt,
            health_report= f"Health reports:\n{user_reports}\n" if user_reports else "",
            summary=conversation_history + current_conversation
        )
        assistant_response = generate_response(prompt_input, max_tokens=512)
        print(f"\nAssistant: {assistant_response}\n")
    
    # print current prompt
    elif user_input.lower() == 'p':
        prompt_input = chatbot_prompt_template.format(
            instruction=system_prompt,
            health_report= f"Health reports:\n{user_reports}\n" if user_reports else "",
            summary=conversation_history + current_conversation
        )
        print("-"*20)
        print(f"Input Prompt:\n{prompt_input}")
        print("-"*20)

    # generate new response for user's input query
    else:
        if current_conversation:
            conversation_history += current_conversation[6:] # truncate "\n\nuser: "

            # conversation_history += f"\nAssistant: {assistant_response}\n"
            conversation_history += f"{assistant_response}\n"
        
        # Update current conversation history.
        current_conversation = f"\n\nUser: {user_input}"

        if len(tokenizer([conversation_history], return_tensors="pt").to(device).input_ids[0]) > 200:
            conversation_history = summarizer(conversation_history, max_length=150, min_length=100, do_sample=False)[0]['summary_text']
        
        prompt_input = chatbot_prompt_template.format(
            instruction=system_prompt,
            health_report= f"Health reports:\n{user_reports}\n" if user_reports else "",
            summary=conversation_history + current_conversation
        )
        
        assistant_response = generate_response(prompt_input, max_tokens=512)
        print(f"\nAssistant: {assistant_response}\n")