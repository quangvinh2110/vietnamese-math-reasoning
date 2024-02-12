from src.infer.rule_based_infer import adding_rule_back_up
from src.infer.tora_infer import Pipeline


question = "Số lớn nhất trong các số: 5,216; 5,126; 5,621; 5,612 là:"
choices = [
    "A. 5,612",
    "B. 5,621",
    "C. 5,216",
    "D. 5,126"
]

tora_pipeline = Pipeline(
    model_path="models_hub/toracode-13b_v2.1",
    prompt_type="zalo",
    device=0
)


output, rule_matched = adding_rule_back_up(question, choices)
if not rule_matched:
    output = tora_pipeline(question, choices)

print(question)
print(choices)
print(output)

