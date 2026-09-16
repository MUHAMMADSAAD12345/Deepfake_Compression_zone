import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.models.fer_model import VGG19FER
from src.models.ser_model import SERMLP

m = VGG19FER().to("cuda")
x = torch.randn(4, 1, 48, 48).to("cuda")
print(f"FER params: {sum(p.numel() for p in m.parameters()):,}")
print(f"FER output: {m(x).shape}")

m2 = SERMLP().to("cuda")
x2 = torch.randn(4, 180).to("cuda")
print(f"SER params: {sum(p.numel() for p in m2.parameters()):,}")
print(f"SER output: {m2(x2).shape}")

print("All GPU models working!")
