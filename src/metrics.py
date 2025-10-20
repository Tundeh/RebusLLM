
import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
try:
    from openai import OpenAI  # type: ignore
except Exception:  # missing optional dependency
    OpenAI = None  # type: ignore

#this seciton contain metrics to evaluate performance of model on the rebus data

#this metrics will involve comparing the model's answer with the ground truth answer

#we will also use a llm as a judge to evaluate the performance

@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    exact_match: bool
    semantic_similarity: float
    llm_judge_score: float
    llm_judge_reasoning: str
    model_answer: str
    ground_truth: str
    sample_id: str

def normalize_text(text: str) -> str:
    """Normalize text for comparison by removing punctuation and converting to lowercase"""
    # Remove common punctuation and extra whitespace
    text = re.sub(r'[^\w\s]', '', text.lower().strip())
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text

def exact_match_score(predicted: str, ground_truth: str) -> bool:
    """
    Check if the predicted answer exactly matches the ground truth.
    Compares normalized versions of both strings.
    """
    predicted_norm = normalize_text(predicted)
    ground_truth_norm = normalize_text(ground_truth)
    return predicted_norm == ground_truth_norm

def extract_idiom_from_response(response: str) -> str:
    """
    Extract the idiom from model response, handling different response formats.
    """
    response = response.strip()
    
    # Try to extract from JSON format first
    try:
        # Look for JSON-like structure
        json_match = re.search(r'\{[^}]*"idiom"[^}]*\}', response)
        if json_match:
            json_str = json_match.group()
            data = json.loads(json_str)
            if 'idiom' in data:
                return data['idiom'].strip()
    except:
        pass
    
    # Try to extract from simple JSON format
    try:
        data = json.loads(response)
        if isinstance(data, dict) and 'idiom' in data:
            return data['idiom'].strip()
    except:
        pass
    
    # If no JSON format, assume the entire response is the idiom
    # Remove common prefixes/suffixes
    response = re.sub(r'^(the idiom is|idiom:|answer:)\s*', '', response, flags=re.IGNORECASE)
    response = re.sub(r'[.!?]+$', '', response)
    
    return response.strip()

def llm_judge_evaluation(
    model_answer: str, 
    ground_truth: str, 
    client: Any = None,
    model_name: str = "gpt-4o-mini"
) -> tuple[float, str]:
    """
    Use an LLM as a judge to evaluate the quality of the model's answer.
    Returns a score (0-10) and reasoning.
    """
    if client is None:
        # If OpenAI SDK not available, return neutral score
        if OpenAI is None:
            return 5.0, "LLM judge disabled or OpenAI SDK unavailable"
        try:
            client = OpenAI()
        except Exception:
            return 5.0, "Failed to initialize OpenAI client for LLM judge"
    
    prompt = f"""
You are evaluating a rebus puzzle answer. Please rate the model's answer on a scale of 0-10.

Ground Truth Answer: "{ground_truth}"
Model's Answer: "{model_answer}"

Consider:
1. Semantic correctness (does the answer mean the same thing?)
2. Exact match vs. close approximation
3. Format appropriateness (idiom vs. explanation vs. other)

Provide your evaluation as a JSON object with:
- "score": number between 0-10
- "reasoning": brief explanation of your evaluation

Respond with only the JSON object.
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert evaluator of rebus puzzle answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        score = float(result.get("score", 5.0))
        reasoning = result.get("reasoning", "No reasoning provided")
        
        return score, reasoning
        
    except Exception as e:
        return 5.0, f"Error in LLM judge evaluation: {str(e)}"

def evaluate_single_sample(
    model_answer: str,
    ground_truth: str, 
    sample_id: str,
    llm_judge_client: Optional[OpenAI] = None
) -> EvaluationResult:
    """
    Evaluate a single sample with all metrics.
    """
    # Extract idiom from model response
    extracted_idiom = extract_idiom_from_response(model_answer)
    
    # Calculate exact match
    exact_match = exact_match_score(extracted_idiom, ground_truth)
    
    # Calculate semantic similarity (simple word overlap for now)
    predicted_words = set(normalize_text(extracted_idiom).split())
    ground_truth_words = set(normalize_text(ground_truth).split())
    
    if len(ground_truth_words) == 0:
        semantic_similarity = 1.0 if len(predicted_words) == 0 else 0.0
    else:
        intersection = predicted_words.intersection(ground_truth_words)
        union = predicted_words.union(ground_truth_words)
        semantic_similarity = len(intersection) / len(union) if len(union) > 0 else 0.0
    
    # LLM judge evaluation
    llm_score, llm_reasoning = llm_judge_evaluation(
        extracted_idiom, ground_truth, llm_judge_client
    )
    
    return EvaluationResult(
        exact_match=exact_match,
        semantic_similarity=semantic_similarity,
        llm_judge_score=llm_score,
        llm_judge_reasoning=llm_reasoning,
        model_answer=extracted_idiom,
        ground_truth=ground_truth,
        sample_id=sample_id
    )

def calculate_overall_metrics(results: List[EvaluationResult]) -> Dict[str, float]:
    """
    Calculate overall metrics from a list of evaluation results.
    """
    if not results:
        return {}
    
    exact_matches = sum(1 for r in results if r.exact_match)
    total_samples = len(results)
    
    return {
        "exact_match_rate": exact_matches / total_samples,
        "average_semantic_similarity": sum(r.semantic_similarity for r in results) / total_samples,
        "average_llm_judge_score": sum(r.llm_judge_score for r in results) / total_samples,
        "total_samples": total_samples
    }

def save_evaluation_results(
    results: List[EvaluationResult], 
    overall_metrics: Dict[str, float],
    output_path: str
) -> None:
    """
    Save evaluation results to a JSON file.
    """
    output_data = {
        "overall_metrics": overall_metrics,
        "individual_results": [
            {
                "sample_id": r.sample_id,
                "exact_match": r.exact_match,
                "semantic_similarity": r.semantic_similarity,
                "llm_judge_score": r.llm_judge_score,
                "llm_judge_reasoning": r.llm_judge_reasoning,
                "model_answer": r.model_answer,
                "ground_truth": r.ground_truth
            }
            for r in results
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation results saved to {output_path}")
    print(f"Overall metrics: {overall_metrics}")

