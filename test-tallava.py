import torch
from lmms_eval.evaluator import simple_evaluate
from lmms_eval.models import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

tasks = ["pope"]

results = simple_evaluate(
    model="tallava",
    model_args="pretrained=ToviTu/ta-llava-pretrain-phase2",  
    tasks=["pope"],
    num_fewshot=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbosity="INFO",
)


print(results)
