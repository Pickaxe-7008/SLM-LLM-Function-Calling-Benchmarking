import random
import re
import os
import json
from datasets import load_dataset
from openai import OpenAI
from anthropic import Anthropic


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


random.seed(42)

LLM = "Anthropic"

k = 5


OPENAI_API_KEY = ""
ANTHROPIC_API_KEY = ""

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

ds = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling")
ds_singleturn = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling_singleturn")

nlp_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
similarity_threshold = 0.6



def normalize(text):
    text = text.lower().strip()
    return text

def get_cosine_similarity(str_one, str_two):
    sentences = [str_one, str_two]
    embeddings = nlp_model.encode(sentences)
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]


def eval_json(output_str, ground_truth):
    tool_calls = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", output_str, re.DOTALL)
    parsed_calls = []
    for call in tool_calls:
        try:
            call_json = json.loads(call)
            if "function" in call_json and "parameters" in call_json["function"]:
                parsed_calls.append({
                    "name": call_json["function"].get("name"),
                    "arguments": call_json["function"].get("parameters")
                })
            elif "name" in call_json and "arguments" in call_json:
                parsed_calls.append({
                    "name": call_json.get("name"),
                    "arguments": call_json.get("arguments")
                })
        except json.JSONDecodeError:
            continue

    mismatches = []
    matched_preds = set()
    correct = 0

    for gt_call in ground_truth:
        found_match = False
        for i, pred_call in enumerate(parsed_calls):
            if i in matched_preds:
                continue
            if gt_call["name"] == pred_call["name"] and gt_call["arguments"] == pred_call["arguments"]:
                matched_preds.add(i)
                correct += 1
                found_match = True
                break
        if not found_match:
            mismatches.append({
                "ground_truth": gt_call,
                "prediction": None,
                "reason": "No matching function call found"
            })

    for i, pred_call in enumerate(parsed_calls):
        if i not in matched_preds:
            mismatches.append({
                "ground_truth": None,
                "prediction": pred_call,
                "reason": "Extra function call not in ground truth"
            })

    total = max(len(ground_truth), len(parsed_calls))
    accuracy = correct / total if total > 0 else 1.0

    TP = correct
    FN = len(ground_truth) - TP
    FP = len(parsed_calls) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 1.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "total_calls": total,
        "correct_calls": correct,
        "num_outputs": len(tool_calls),
        "num_ground_truth": len(ground_truth),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mismatches": mismatches
    }

def eval_json_semantic(output_str, ground_truth):
    tool_calls = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", output_str, re.DOTALL)
    parsed_calls = []
    for call in tool_calls:
        try:
            call_json = json.loads(call)
            if "function" in call_json and "parameters" in call_json["function"]:
                parsed_calls.append({
                    "name": call_json["function"].get("name"),
                    "arguments": call_json["function"].get("parameters")
                })
            elif "name" in call_json and "arguments" in call_json:
                parsed_calls.append({
                    "name": call_json.get("name"),
                    "arguments": call_json.get("arguments")
                })
        except json.JSONDecodeError:
            continue

    mismatches = []
    matched_preds = set()
    correct = 0

    for gt_call in ground_truth:
        found_match = False
        for i, pred_call in enumerate(parsed_calls):
            if i in matched_preds:
                continue
            if gt_call["name"] != pred_call["name"]:
                continue

            # Check arguments semantically
            all_args_match = True
            for arg_key, arg_val in gt_call["arguments"].items():
                pred_val = pred_call["arguments"].get(arg_key)
                if isinstance(arg_val, str) and isinstance(pred_val, str):
                    sim = get_cosine_similarity(normalize(arg_val), normalize(pred_val))
                    if sim < similarity_threshold:
                        all_args_match = False
                        break
                else:
                    if arg_val != pred_val:
                        all_args_match = False
                        break

            if all_args_match:
                matched_preds.add(i)
                correct += 1
                found_match = True
                break

        if not found_match:
            mismatches.append({
                "ground_truth": gt_call,
                "prediction": None,
                "reason": "No matching function call found"
            })

    for i, pred_call in enumerate(parsed_calls):
        if i not in matched_preds:
            mismatches.append({
                "ground_truth": None,
                "prediction": pred_call,
                "reason": "Extra function call not in ground truth"
            })

    total = max(len(ground_truth), len(parsed_calls))
    accuracy = correct / total if total > 0 else 1.0

    TP = correct
    FN = len(ground_truth) - TP
    FP = len(parsed_calls) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 1.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "total_calls": total,
        "correct_calls": correct,
        "num_outputs": len(tool_calls),
        "num_ground_truth": len(ground_truth),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mismatches": mismatches
    }

def create_examples(dataset, k, exclude_sample=None):
    examples = []
    attempts = 0

    while len(examples) < k and attempts < k * 5:
        sample = random.choice(dataset["train"])
        attempts += 1

        if exclude_sample is not None and sample == exclude_sample:
            continue

        # Extract messages
        human_msg = " ".join(
            c["value"] for c in sample["conversations"] if c["from"] == "human"
        )

        tool_calls = []
        for c in sample["conversations"]:
            if c["from"] in ["gpt", "assistant"]:
                matches = re.findall(
                    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
                    c["value"],
                    re.DOTALL
                )
                for m in matches:
                    tool_calls.append(m)

        if len(tool_calls) == 0:
            continue

        function_schema = json.dumps(
            json.loads(sample["tools"]),
            indent=2
        )

        example = f"""
        User Request:
        {human_msg}
        
        Available Functions:
        {function_schema}
        
        Assistant Response:
        {''.join(f"<tool_call>{tc}</tool_call>" for tc in tool_calls)}
        """
        examples.append(example.strip())

    return "\n\n".join(examples)




class LLMFunctionCallerFewShot:
    def __init__(self, model_name, k_examples):
        self.model_name = model_name
        self.k_examples = k_examples
        if model_name == "OpenAI":
            self.model = OpenAI()
        elif model_name == "Anthropic":
            self.model = Anthropic()

    def function_caller(self, task_instruction_prompt):
        system_prompt = """
                    You are an AI assistant that generates structured function calls. 
                    Output all function calls required for this user request. Each function call should 
                    be a JSON object wrapped in <tool_call>...</tool_call> tags. 
                    If multiple functions are needed, output multiple <tool_call> objects in the same response. 
                    Do not include explanations or text outside <tool_call> tags. 
                    Follow the schema exactly. Do not include any text outside <tool_call>.
                    """
        if self.model_name == "OpenAI":
            response = self.model.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task_instruction_prompt}
                ],
                temperature=0,
                max_completion_tokens=2048,
            )
            return response.choices[0].message.content
        elif self.model_name == "Anthropic":
            response = self.model.messages.create(
                model="claude-sonnet-4-5",
                system=system_prompt,
                messages=[
                    {"role": "user", "content": task_instruction_prompt}
                ],
                temperature=0,
                max_tokens=2048
            )
            return "".join(b.text for b in response.content if b.type == "text")
        else:
            pass


# Zero-Shot Testing Phase

FunctionAgentZeroShot = LLMFunctionCallerFewShot(LLM, k)
num_runs = 100


#-------------Single Turn Function Calling----------------------#

print("Test runs (Single Turn, Abstention Datapoints Excluded): " + str(num_runs))

accuracy = 0.0
accuracy_semantic = 0.0
average_precision = 0.0
average_precision_semantic = 0.0
average_recall = 0.0
average_recall_semantic = 0.0
average_f1_score = 0.0
average_f1_score_semantic = 0.0

runs_done = 0

while runs_done < num_runs:
    sample = random.choice(ds_singleturn["train"])
    system_msg = " ".join(c["value"] for c in sample["conversations"] if c["from"] == "system")
    human_msg = " ".join(c["value"] for c in sample["conversations"] if c["from"] == "human")

    g_truth = []
    for c in sample["conversations"]:
        if c["from"] in ["gpt", "assistant"]:
            matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", c["value"], re.DOTALL)
            for m in matches:
                g_truth.append(json.loads(m))

    if len(g_truth) == 0:
        continue

    function_schema = sample["tools"]

    input_str = f"""
    You are given a user request and a set of available functions. Your task is to produce one or more JSON objects
    that represent valid calls to the function(s) that best satisfy the user's request. Follow these rules exactly:

    - Output only JSON objects and nothing else.
    - Wrap each JSON object in <tool_call>...</tool_call> tags.
    - If multiple functions are required to fulfill the request, output multiple <tool_call> objects in the same response.
    - Do not include explanations, comments, or natural language outside the <tool_call> tags.
    - Each JSON must strictly conform to the provided function schema.
    - Include all required arguments, with names and types exactly as specified.
    - Infer argument values only from the user request; do not invent unsupported arguments.
    
    Here are some examples:
    
    {create_examples(ds_singleturn, k)}
    
    Available Functions: {json.dumps(json.loads(function_schema), indent=2)}
    System Instructions: {system_msg}
    User Request: {human_msg}
    """

    output = FunctionAgentZeroShot.function_caller(input_str)
    results = eval_json(output, g_truth)
    results_semantic = eval_json_semantic(output, g_truth)

    accuracy += results['accuracy']
    accuracy_semantic += results_semantic['accuracy']
    average_precision += results['precision']
    average_precision_semantic += results_semantic['precision']
    average_recall += results['recall']
    average_recall_semantic += results_semantic['recall']
    average_f1_score += results['f1_score']
    average_f1_score_semantic += results_semantic['f1_score']

    print(f"Run {runs_done + 1} \nResults: {results}\nResults (Semantic Similarity): {results_semantic}")
    runs_done += 1


print("--------------------------------------------------------------------")
print("Average Accuracy over runs:", accuracy / num_runs)
print("Average Precision over runs:", average_precision / num_runs)
print("Average Recall over runs:", average_recall / num_runs)
print("Average F1-score over runs:", average_f1_score / num_runs)
print("--------------------------------------------------------------------")
print("Average Accuracy over runs (Semantic):", accuracy_semantic / num_runs)
print("Average Precision over runs (Semantic):", average_precision_semantic / num_runs)
print("Average Recall over runs (Semantic):", average_recall_semantic / num_runs)
print("Average F1-score over runs (Semantic):", average_f1_score_semantic / num_runs)
print("--------------------------------------------------------------------")

#-------------Multiple Turn Function Calling----------------------#

print("Test runs (Multiple Turns, Abstention Datapoints Excluded): " + str(num_runs))

accuracy = 0.0
accuracy_semantic = 0.0
average_precision = 0.0
average_precision_semantic = 0.0
average_recall = 0.0
average_recall_semantic = 0.0
average_f1_score = 0.0
average_f1_score_semantic = 0.0


runs_done = 0

while runs_done < num_runs:
    sample = random.choice(ds["train"])
    system_msg = " ".join(c["value"] for c in sample["conversations"] if c["from"] == "system")
    human_msg = " ".join(c["value"] for c in sample["conversations"] if c["from"] == "human")

    g_truth = []
    for c in sample["conversations"]:
        if c["from"] in ["gpt", "assistant"]:
            matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", c["value"], re.DOTALL)
            for m in matches:
                g_truth.append(json.loads(m))

    if len(g_truth) == 0:
        continue

    function_schema = sample["tools"]

    input_str = f"""
    You are given a user request and a set of available functions. Your task is to produce one or more JSON objects
    that represent valid calls to the function(s) that best satisfy the user's request. Follow these rules exactly:

    - Output only JSON objects and nothing else.
    - Wrap each JSON object in <tool_call>...</tool_call> tags.
    - If multiple functions are required to fulfill the request, output multiple <tool_call> objects in the same response.
    - Do not include explanations, comments, or natural language outside the <tool_call> tags.
    - Each JSON must strictly conform to the provided function schema.
    - Include all required arguments, with names and types exactly as specified.
    - Infer argument values only from the user request; do not invent unsupported arguments.
    
    Here are some examples:
    
    {create_examples(ds, k)}
    
    Available Functions: {json.dumps(json.loads(function_schema), indent=2)}
    System Instructions: {system_msg}
    User Request: {human_msg}
    """

    output = FunctionAgentZeroShot.function_caller(input_str)
    results = eval_json(output, g_truth)
    results_semantic = eval_json_semantic(output, g_truth)

    accuracy += results['accuracy']
    accuracy_semantic += results_semantic['accuracy']
    average_precision += results['precision']
    average_precision_semantic += results_semantic['precision']
    average_recall += results['recall']
    average_recall_semantic += results_semantic['recall']
    average_f1_score += results['f1_score']
    average_f1_score_semantic += results_semantic['f1_score']

    print(f"Run {runs_done + 1} \nResults: {results}\nResults (Semantic Similarity): {results_semantic}")
    runs_done += 1


print("--------------------------------------------------------------------")
print("Average Accuracy over runs:", accuracy / num_runs)
print("Average Precision over runs:", average_precision / num_runs)
print("Average Recall over runs:", average_recall / num_runs)
print("Average F1-score over runs:", average_f1_score / num_runs)
print("--------------------------------------------------------------------")
print("Average Accuracy over runs (Semantic):", accuracy_semantic / num_runs)
print("Average Precision over runs (Semantic):", average_precision_semantic / num_runs)
print("Average Recall over runs (Semantic):", average_recall_semantic / num_runs)
print("Average F1-score over runs (Semantic):", average_f1_score_semantic / num_runs)
print("--------------------------------------------------------------------")
