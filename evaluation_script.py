import json
import os
import time
import google.generativeai as genai
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in environment")
    exit(1)
genai.configure(api_key=GEMINI_API_KEY)

# Task-specific evaluation prompts based on MT-Bench-101
EVALUATION_PROMPTS = {
    "CM": """Evaluate the AI assistant's ability to recall and utilize previously mentioned information from earlier in the conversation. 
    
Evaluation Criteria:
1. Appropriately recalls relevant details from earlier parts of the conversation
2. Integrates remembered information into current responses coherently
3. Maintains context established by previous dialogue exchanges
4. Facilitates smooth and logical progression without contradictions

Scoring Guidelines:
1-3 points: Poor recall, inconsistent/contradictory responses, fails to maintain context
4-6 points: Moderate memory ability, sporadic integration, occasional context disregard  
7-9 points: Reliable recall and utilization, coherent dialogue with minor lapses
10 points: Exceptional memory recall, seamless integration, perfect context preservation""",
    "SI": """Evaluate the AI assistant's ability to handle separate input scenarios where task requirements and specific content are provided in different turns.
    
Evaluation Criteria:
1. In the first round, should ask for specific details rather than directly answering
2. From second round onwards, should provide correct specific answers when given the task content

Scoring Guidelines:
1-3 points: Failed to understand task request, neither asked questions nor provided relevant information
4-6 points: Understood some aspects but response could be more specific or relevant
7-9 points: Provided useful response that was mostly correct and targeted
10 points: Perfect understanding and comprehensive, accurate answer meeting all expectations""",
    "AR": """Evaluate the AI assistant's understanding of referential information (pronouns, demonstratives like 'it', 'this', 'these', 'that').
    
Evaluation Criteria:
1. Correctly understands referential information from questions relating to previous dialogue
2. Provides responses consistent with current questions and accurate information

Scoring Guidelines:
1-3 points: Fails to recognize or correctly interpret referential information
4-6 points: Shows partial understanding but may include inaccuracies
7-9 points: Good understanding with only slight inaccuracies or omissions
10 points: Excellent understanding and use of referential information""",
    "TS": """Evaluate the AI assistant's ability to handle topic shifts in conversation.
    
Evaluation Criteria:
1. Detects and acknowledges topic changes without reverting to previous subjects
2. Provides relevant responses to new topics without improper influence from preceding dialogue
3. Offers coherent and contextually appropriate responses to new subjects
4. Demonstrates clear break from past conversation threads

Scoring Guidelines:
1-3 points: Struggles with topic transitions, frequently influenced by previous topics
4-6 points: Moderate adaptation ability with occasional lingering effects from earlier discussions
7-9 points: Adapts well with minimal reference to prior topics
10 points: Excels at topic shifts, seamless transition without irrelevant carryover""",
    "CC": """Evaluate the AI assistant's ability to resist interference from similar-looking queries with distinct meanings.
    
Evaluation Criteria:
1. Response directly corresponds to current question content with accurate information
2. Not influenced by previous dialogue patterns, remaining focused on current question only

Scoring Guidelines:
1-3 points: Largely influenced by previous interactions, fails to address current question accurately
4-6 points: Shows some resistance but includes irrelevant details from previous dialogues
7-9 points: Mostly resistant to interference, accurately addresses current question
10 points: Completely free from interference, wholly relevant and accurate response""",
    "CR": """Evaluate the AI assistant's content rephrasing ability.
    
Evaluation Criteria:
1. Maintains the same main idea as the original response
2. Complies with specific rewriting requirements set by the user

Scoring Guidelines:
1-3 points: Significantly deviates from original main idea or fails to meet rewriting requirements
4-6 points: Captures original idea but only partially meets requirements or lacks fluency
7-9 points: Maintains original idea and satisfies most requirements with minor issues
10 points: Perfectly preserves original idea and fulfills all rewriting requirements""",
    "FR": """Evaluate the AI assistant's format rephrasing ability.
    
Evaluation Criteria:
1. Maintains the same main idea as the original response
2. Complies with specific format rewriting requirements (structure, presentation)

Scoring Guidelines:
1-3 points: Significantly deviates from original main idea or fails to meet format requirements
4-6 points: Captures original idea but only partially meets format requirements
7-9 points: Maintains original idea and satisfies most format requirements with minor issues
10 points: Perfectly preserves original idea and fulfills all format rewriting requirements""",
    "SC": """Evaluate the AI assistant's self-correction ability when errors are pointed out.
    
Evaluation Criteria:
1. Thoroughly assesses previous response when error is identified
2. Acknowledges mistake and provides updated, accurate response when error is valid

Scoring Guidelines:
1-3 points: Fails to recognize or adequately address identified error
4-6 points: Recognizes error but may only partially correct or provide incomplete update
7-9 points: Correctly identifies and acknowledges error, makes substantial correction
10 points: Exceptional self-correction, prompt acknowledgment and comprehensive update""",
    "SA": """Evaluate the AI assistant's ability to maintain confidence in correct responses when incorrectly challenged.
    
Evaluation Criteria:
1. Engages with challenge and assesses its validity
2. Maintains original correct answer with clear, reasoned explanation when challenge is incorrect

Scoring Guidelines:
1-3 points: Does not engage appropriately or changes response without justification
4-6 points: Engages but lacks confidence, provides weak justification
7-9 points: Appropriately assesses incorrect challenge, maintains position with clear justification
10 points: Excellent confidence maintenance with strong, convincing explanation""",
    "MR": """Evaluate the AI assistant's mathematical reasoning capabilities across multiple turns.
    
Evaluation Criteria:
1. Accuracy of answer against reference solution
2. Completeness and clarity of step-by-step reasoning process
3. Ability to incorporate relevant historical dialogue information
4. Clear communication that aids understanding

Scoring Guidelines:
1-3 points: Incorrect answers and/or unclear reasoning process, missing key steps
4-6 points: Partially correct with minor errors, reasoning may lack detail in some steps
7-9 points: Correct answers with mostly complete reasoning process
10 points: Completely correct with detailed, clear step-by-step reasoning""",
    "GR": """Evaluate the AI assistant's general reasoning capabilities across multiple turns.
    
Evaluation Criteria:
1. Accuracy of answer against reference solution
2. Completeness and clarity of step-by-step reasoning process
3. Ability to integrate relevant historical dialogue information
4. Clear communication that aids understanding

Scoring Guidelines:
1-3 points: Incorrect answers and/or unclear reasoning process, missing key steps
4-6 points: Partially correct with minor errors, reasoning may lack detail
7-9 points: Correct answers with well-articulated reasoning process
10 points: Completely correct with detailed, clear reasoning aligned with sound principles""",
    "IC": """Evaluate the AI assistant's use of clarifying questions when dealing with ambiguous queries.
    
Evaluation Criteria:
1. Recognizes when questions contain ambiguities requiring clarification
2. Uses counter-questions effectively to address missing information
3. Provides detailed, accurate responses once query is clarified

Scoring Guidelines:
1-3 points: Fails to identify need for clarification or uses it ineffectively
4-6 points: Recognizes need but uses clarification suboptimally
7-9 points: Effectively uses questions to address unclear elements, provides accurate responses
10 points: Expertly discerns when to clarify, employs precisely, responds with detailed accuracy""",
    "PI": """Evaluate the AI assistant's proactive interaction abilities.
    
Evaluation Criteria:
1. Takes initiative in contributing beyond direct answers with relevant follow-up questions
2. Maintains conversation flow and encourages further discourse
3. Interactive elements are appropriate and foster natural, engaging conversation
4. Responsive to input while being proactive

Scoring Guidelines:
1-3 points: Poor interactivity, minimal responses, or misplaced attempts that hamper flow
4-6 points: Moderate interactivity, occasionally engaging but not consistently maintaining momentum
7-9 points: Highly interactive, regularly using questions while preserving relevancy
10 points: Excellent interactivity, skillfully enriching conversation without dominating""",
}


def load_results_files(results_dir: str) -> List[str]:
    """Load all JSON result files from the results directory"""
    json_files = []
    for filename in os.listdir(results_dir):
        if filename.endswith(".jsonl") or filename.endswith(".json"):
            json_files.append(os.path.join(results_dir, filename))
    return json_files


def load_jsonl_file(filepath: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error parsing line in {filepath}: {e}")
                continue
    return data


def build_context_from_history(raw_history: List[Dict[str, str]]) -> str:
    """Build conversation context string excluding the final reference answer.

    raw_history is a list of {"user":..., "bot":...} turns from the dataset.
    We include all complete (user, bot) pairs except we REMOVE the bot part of the last turn so
    that the evaluator only sees the final user question the model answered, without leaking
    the reference answer.
    """
    if not raw_history:
        return ""
    if len(raw_history) == 1:
        # Only final question, no earlier context
        return f"User: {raw_history[-1].get('user','').strip()}"
    lines = []
    # All but last turn fully
    for pair in raw_history[:-1]:
        user_msg = pair.get("user", "").strip()
        bot_msg = pair.get("bot", "").strip()
        if user_msg:
            lines.append(f"User: {user_msg}")
        if bot_msg:
            lines.append(f"Assistant: {bot_msg}")
    # Final user message only
    final_user = raw_history[-1].get("user", "").strip()
    if final_user:
        lines.append(f"User: {final_user}")
    return "\n".join(lines)


def evaluate_response_with_gemini(
    task_type: str,
    response: str,
    id: str,
    context: Optional[str] = None,
) -> Dict:
    """Evaluate a single response using Gemini 2.5 Flash Lite with optional prior context."""

    evaluation_prompt = EVALUATION_PROMPTS.get(task_type, EVALUATION_PROMPTS["GR"])

    context_block = (
        f"Conversation Context (previous turns):\n{context}\n\n" if context else ""
    )

    full_prompt = f"""Please act as an impartial judge and evaluate the following AI assistant's response.

{evaluation_prompt}

{context_block}AI Assistant's Response to Evaluate (answer to the LAST user message above):
"{response}"

Please provide:
1. A detailed explanation of your evaluation
2. A numerical score from 1-10 based on the criteria above
3. Specific strengths and weaknesses observed

Format your response as:
EXPLANATION: [Your detailed explanation]
SCORE: [Your numerical score 1-10]
STRENGTHS: [Observed strengths]
WEAKNESSES: [Observed weaknesses]
"""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        g_response = model.generate_content(full_prompt)
        evaluation_text = g_response.text

        # Parse the response to extract score
        score = None
        explanation = ""
        strengths = ""
        weaknesses = ""

        lines = evaluation_text.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith("EXPLANATION:"):
                current_section = "explanation"
                explanation = line.replace("EXPLANATION:", "").strip()
            elif line.startswith("SCORE:"):
                current_section = "score"
                score_text = line.replace("SCORE:", "").strip()
                try:
                    score = float(score_text.split()[0])
                except Exception:
                    score = None
            elif line.startswith("STRENGTHS:"):
                current_section = "strengths"
                strengths = line.replace("STRENGTHS:", "").strip()
            elif line.startswith("WEAKNESSES:"):
                current_section = "weaknesses"
                weaknesses = line.replace("WEAKNESSES:", "").strip()
            elif line and current_section:
                if current_section == "explanation":
                    explanation += " " + line
                elif current_section == "strengths":
                    strengths += " " + line
                elif current_section == "weaknesses":
                    weaknesses += " " + line

        if score is None:
            import re

            score_matches = re.findall(
                r"(\d+(?:\.\d+)?)/10|\b(\d+(?:\.\d+)?)\b", evaluation_text
            )
            if score_matches:
                for match in score_matches:
                    potential_score = float(match[0] if match[0] else match[1])
                    if 1 <= potential_score <= 10:
                        score = potential_score
                        break

        return {
            "id": id,
            "task_type": task_type,
            "score": score if score is not None else 0,
            "explanation": explanation.strip(),
            "strengths": strengths.strip(),
            "weaknesses": weaknesses.strip(),
            "evaluation_status": "success",
        }
    except Exception as e:
        print(f"Error evaluating item {id}: {str(e)}")
        return {
            "id": id,
            "task_type": task_type,
            "score": 0,
            "explanation": f"Evaluation failed: {str(e)}",
            "strengths": "",
            "weaknesses": "",
            "evaluation_status": "error",
        }


def main():
    results_dir = "results"
    dataset_path = "mtbench101.jsonl"
    delay = 0

    # Load dataset histories into memory (id -> history list)
    id_to_history: Dict[Any, List[Dict[str, str]]] = {}
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, "r", encoding="utf-8") as ds:
                for line in ds:
                    try:
                        rec = json.loads(line)
                        hid = rec.get("id")
                        history = rec.get("history")
                        if hid is not None and isinstance(history, list):
                            id_to_history[hid] = history
                    except Exception:
                        continue
        except Exception as e:
            print(f"Failed to load dataset histories: {e}")
    else:
        print("MTBench 101 dataset file not found.")
        exit(1)

    # Load all result files
    result_files = load_results_files(results_dir)
    print(f"Found {len(result_files)} result files to evaluate")

    for result_file in result_files:
        print(f"\nProcessing file: {result_file}")

        # Load the results
        results = load_jsonl_file(result_file)
        print(f"Loaded {len(results)} results from {result_file}")

        # Check which items are already evaluated (look for 'score' key and 'evaluation_status' == 'success')
        already_evaluated_ids = set()
        for i, item in enumerate(results):
            # If the item has a 'score' and 'evaluation_status' == 'success', consider it evaluated
            if (
                isinstance(item, dict)
                and item.get("evaluation_status") == "success"
                and "score" in item
            ):
                # Use 'id' if present, else index
                already_evaluated_ids.add(item.get("id", i))
        if already_evaluated_ids:
            print(f"Already evaluated: {len(already_evaluated_ids)} items")

        # Prepare new results list (for rewriting the file)
        new_results = []

        for i, result in enumerate(results):
            # Determine unique item id
            id = result.get("id", i)

            # If already evaluated, just append as is
            if id in already_evaluated_ids:
                new_results.append(result)
                continue

            print(f"Evaluating item {i+1}/{len(results)} (ID: {id})")

            task_type = result.get("task", "GR")
            model_response = result.get("response", "")

            if not model_response.strip():
                print(f"Skipping item {id} - no response")
                new_results.append(result)
                continue

            # Build context if we have history
            raw_history = id_to_history.get(id)
            context_str = (
                build_context_from_history(raw_history) if raw_history else None
            )

            evaluation = evaluate_response_with_gemini(
                task_type, model_response, id, context=context_str
            )

            evaluation.update(
                {
                    "original_response": model_response,
                    "model_name": result.get("model_name", ""),
                    "inference_time_seconds": result.get("inference_time_seconds", 0),
                    "energy_joules": result.get("energy_joules", 0),
                    "peak_memory_mb": result.get("peak_memory_mb", 0),
                }
            )

            # Ensure 'response' field is last in the output
            # Copy all fields except 'response', then add 'response' at the end
            output_item = dict()
            for k, v in evaluation.items():
                if k != "original_response":
                    output_item[k] = v
            # Add all other fields from result except 'response' (if not already present)
            for k, v in result.items():
                if k != "response" and k not in output_item:
                    output_item[k] = v
            # Add 'response' as last field
            output_item["response"] = model_response

            new_results.append(output_item)

            # Write progress after each evaluation
            with open(result_file, "w", encoding="utf-8") as f:
                # Write all processed items (evaluated and skipped)
                for item in new_results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                # Write remaining unprocessed items
                for remaining_item in results[i + 1 :]:
                    f.write(json.dumps(remaining_item, ensure_ascii=False) + "\n")

            print(f"  Task: {task_type}, Score: {evaluation['score']}")

            time.sleep(delay)

        print(f"Evaluations saved to: {result_file}")

    print(f"\nAll evaluations completed!")


if __name__ == "__main__":
    main()
