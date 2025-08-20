import json
import time
import psutil
import os
import gc
import ollama
import pyRAPL
from pyRAPL import Measurement

MODEL_NAME = "tinyllama:1.1b"
OUTPUT_FILE = os.path.join("results", "tinyllama-1.1b.jsonl")
DATASET_PATH = "mtbench101.jsonl"

# Check if energy_uj is readable
ENERGY_UJ_PATH = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
if not os.access(ENERGY_UJ_PATH, os.R_OK):
    print(
        f"[ERROR] Permission denied: '{ENERGY_UJ_PATH}'\n"
        "You can fix this by running:\n"
        f"  sudo chmod o+r {ENERGY_UJ_PATH}\n"
        "Exiting."
    )
    exit(1)

pyRAPL.setup()

# Task-specific prompts based on MT-Bench-101 research paper
TASK_PROMPTS = {
    "CM": "You are having a multi-turn conversation. Use information from the conversation history to answer the current question accurately. Pay attention to details mentioned earlier in our dialogue.",
    "SI": "This is a multi-turn dialogue where the first turn contains task requirements and subsequent turns provide specific content. Wait for the specific content before providing your response.",
    "AR": "In this conversation, pay careful attention to pronouns and references (like 'it', 'this', 'these', 'that') and what they refer to based on our previous conversation.",
    "TS": "This conversation may involve topic changes. Focus on the current topic being discussed and adapt when the conversation shifts to a new subject.",
    "CC": "Be careful to distinguish between similar-looking questions that may have different meanings. Focus on the specific question being asked now.",
    "CR": "You may be asked to rephrase your previous response with different content while maintaining the same core information.",
    "FR": "You may be asked to rephrase your previous response in a different format (like changing from paragraph to list format) while keeping the same content.",
    "SC": "If I point out an error in your response, please carefully reconsider and correct it if you were indeed wrong.",
    "SA": "If I challenge your response, please evaluate whether my challenge is correct. If your original response was accurate, maintain your position with proper justification.",
    "MR": "We will work together to solve mathematical problems step by step across multiple dialogue turns. Use information from our entire conversation.",
    "GR": "We will work together to solve reasoning problems step by step across multiple dialogue turns. Use information from our entire conversation.",
    "IC": "If a question is unclear or ambiguous, ask clarifying questions to better understand what is being asked before providing your answer.",
    "PI": "Engage proactively in our conversation by asking relevant follow-up questions or making comments that encourage continued dialogue.",
}


def load_dataset(file_path):
    """Load JSONL dataset"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_peak_memory():
    """Get peak memory usage since process start in MB"""
    process = psutil.Process(os.getpid())
    return (
        process.memory_info().peak_wss / 1024 / 1024
        if hasattr(process.memory_info(), "peak_wss")
        else process.memory_info().rss / 1024 / 1024
    )


def force_memory_cleanup():
    """Force garbage collection to get more accurate memory measurements"""
    gc.collect()
    gc.collect()  # Call twice for better cleanup
    gc.collect()


def build_conversation_context(item):
    """Build conversation context from history - exclude the last user turn as that's the current query"""
    context = ""
    if "history" in item and item["history"]:
        # Include all turns except the last user turn (which becomes the current query)
        history_turns = item["history"][:-1]  # Exclude last turn
        for turn in history_turns:
            if "user" in turn and "bot" in turn:
                context += f"Human: {turn['user']}\n"
                context += f"Assistant: {turn['bot']}\n"
    return context


def get_task_specific_prompt(task_type, context, new_query):
    """Generate task-specific prompt based on the task type"""
    base_prompt = TASK_PROMPTS.get(task_type, "")

    if context.strip():
        # For tasks that need conversation history
        if task_type in ["CM", "AR", "MR", "GR", "SC", "SA", "CR", "FR"]:
            prompt = f"{base_prompt}\n\nConversation history:\n{context.strip()}\n\nHuman: {new_query}\nAssistant:"
        else:
            prompt = f"{base_prompt}\n\nHuman: {new_query}\nAssistant:"
    else:
        # No conversation history
        prompt = f"{base_prompt}\n\nHuman: {new_query}\nAssistant:"

    return prompt


def extract_query_from_item(item):
    """Extract the current query from the item"""
    # For this dataset, items have a conversation history and we need to continue naturally
    # The last turn in history is what the model should respond to
    if "history" in item and item["history"]:
        last_turn = item["history"][-1]
        if "user" in last_turn:
            return last_turn["user"]
        else:
            return "Continue the conversation appropriately."
    elif "conversations" in item:
        return item["conversations"][0]["value"]  # First human message
    elif "prompt" in item:
        return item["prompt"]
    elif "question" in item:
        return item["question"]
    else:
        return str(item)  # Fallback


def main():
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(DATASET_PATH)

    # Establish baseline memory usage before starting inference
    print("Establishing baseline memory usage...")
    force_memory_cleanup()
    time.sleep(1)  # Allow system to stabilize
    baseline_memory = get_memory_usage()
    print(f"Baseline memory: {baseline_memory:.2f} MB")

    # Process each dialogue and write results line by line
    total_time = 0.0
    total_energy = 0.0
    total_memory = 0.0
    count = 0

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    print(f"Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, item in enumerate(dataset):
            print(f"\n\033[92mProcessing item {i+1}/{len(dataset)}\033[0m")

            # Get task type
            task_type = item.get("task", "")
            print(f"Task type: {task_type}")

            # Build conversation context
            context = build_conversation_context(item)

            # Extract current query
            current_query = extract_query_from_item(item)

            # Build task-specific prompt
            prompt = get_task_specific_prompt(task_type, context, current_query)

            print(
                "\033[93mFull prompt:\033[0m",
                prompt,
            )

            # Clean up memory before measurement
            force_memory_cleanup()
            pre_inference_memory = get_memory_usage()

            # Measure energy, time, and memory
            m = Measurement("package-0")
            m.begin()
            start_time = time.time()

            # Track peak memory during inference (absolute values)
            peak_memory_mb = pre_inference_memory

            # Stream response using Ollama
            response_chunks = []
            response = ollama.generate(
                model=MODEL_NAME,
                prompt=prompt,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 1028,
                    "stop": ["</s>", "<|endoftext|>"],
                },
                stream=True,
            )

            # Collect streaming response and track peak memory
            for chunk in response:
                if "response" in chunk:
                    print(chunk["response"], end="", flush=True)
                    response_chunks.append(chunk["response"])
                    # Track absolute peak memory during inference
                    current_memory = get_memory_usage()
                    peak_memory_mb = max(peak_memory_mb, current_memory)

            end_time = time.time()
            m.end()

            print()

            # Get energy measurement
            energy_microjoules = m.result.pkg[0] if m.result and m.result.pkg else 0
            energy_joules = energy_microjoules / 1_000_000  # Convert µJ to J

            # Store results
            full_response = "".join(response_chunks) if response_chunks else ""

            # Create result
            result = {
                "id": item.get("id"),
                "task": task_type,
                "response": full_response.strip(),
                "inference_time_seconds": round(end_time - start_time, 4),
                "energy_joules": round(energy_joules, 6),
                "peak_memory_mb": round(peak_memory_mb, 2),  # Absolute peak memory
                "baseline_memory_mb": round(baseline_memory, 2),
                "model_name": MODEL_NAME,
            }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            # Update summary stats
            total_time += result["inference_time_seconds"]
            total_energy += result["energy_joules"]
            total_memory += result["peak_memory_mb"]
            count += 1

    # Print summary
    avg_memory = total_memory / count if count else 0
    print(f"\nSummary:")
    print(f"Total items processed: {count}")
    print(f"Total inference time: {total_time:.2f} seconds")
    print(f"Total energy consumed: {total_energy:.2f} joules")
    print(f"Average peak memory per inference: {avg_memory:.2f} MB")
    print(f"Baseline memory usage: {baseline_memory:.2f} MB")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
