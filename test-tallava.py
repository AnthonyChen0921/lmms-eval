import torch
from lmms_eval.evaluator import simple_evaluate
from lmms_eval.models import get_model
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def handle_non_serializable(obj):
    if callable(obj):
        return str(obj) 
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    return str(obj)


tasks = ["pope"]

results = simple_evaluate(
    model="tallava",
    model_args="pretrained=ToviTu/ta-llava-pretrain-phase2",  
    tasks=["pope"],
    num_fewshot=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbosity="INFO",
)

# save the results
output_file = "evaluation_results.json"
try:
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4, default=handle_non_serializable)
    print(f"Results saved to {output_file}")
except Exception as e:
    print(f"Failed to save results: {e}")