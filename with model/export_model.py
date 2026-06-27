import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

model.eval()

ids = torch.randint(100, (1, 16))
mask = torch.ones((1, 16), dtype=torch.long)
traced = torch.jit.trace(
    model,
    (ids, mask),
    strict=False
)

traced.save("model.pt")

print("saved model.pt")
