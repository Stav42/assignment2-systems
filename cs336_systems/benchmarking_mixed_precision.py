import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print("  fc1 (+relu) output dtype:", x.dtype)
        x = self.ln(x)
        print("  layernorm output dtype:  ", x.dtype)
        x = self.fc2(x)
        print("  fc2 output dtype (logits):", x.dtype)
        return x


def run(device_type: str, dtype: torch.dtype):
    print(f"\n=== autocast(device_type={device_type!r}, dtype={dtype}) ===")
    device = torch.device(device_type)
    model = ToyModel(20, 5).to(device)
    print("  param dtype (outside autocast):", next(model.parameters()).dtype)

    x = torch.randn(4, 20, device=device)
    target = torch.randint(0, 5, (4,), device=device)

    with torch.autocast(device_type=device_type, dtype=dtype):
        print("  param dtype (inside autocast): ", next(model.parameters()).dtype)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, target)
        print("  loss dtype:               ", loss.dtype)

    loss.backward()
    print("  fc1.weight.grad dtype:     ", model.fc1.weight.grad.dtype)


if torch.backends.mps.is_available():
    run("mps", torch.float16)

run("cpu", torch.bfloat16)
