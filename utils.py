import re
from math_verify import parse, verify, LatexExtractionConfig

def extract_boxed_answer(solution: str, context: str = "solution") -> str:
    """Extract the content inside \\boxed{} from the solution."""
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    match = re.search(pattern, solution)
    
    if match:
        return match.group(1)
    else:
        return ""

def verify_answer(predicted: str, ground_truth: str) -> bool:
    """
    Verify if predicted answer matches ground truth using math-verify.
    Falls back to string comparison if parsing fails.
    """
    if not predicted or not ground_truth:
        return False
    
    try:
        # Wrap in $ $ for LaTeX math mode
        predicted_latex = f"${predicted}$"
        ground_truth_latex = f"${ground_truth}$"
        parsed_ground_truth = parse(ground_truth_latex, extraction_config=[LatexExtractionConfig()])
        parsed_predicted = parse(predicted_latex, extraction_config=[LatexExtractionConfig()])
        return verify(parsed_ground_truth, parsed_predicted)
    except Exception as e:
        print(f"Warning: Failed to verify answer: {e}")
        return False