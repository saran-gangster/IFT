# IKT

100x parameter efficiency + LONG CTX

```python
from IKT import InfiniKANTransformer
 # Model parameters
num_tokens = 10000
dim = 512
depth = 12
heads = 8
dim_head = 64
attn_dropout = 0.1
ff_dropout = 0.1
use_mem_delta_rule = True
kan_implementation = "EFFICIENT_KAN"

# Initialize the model
model = InfiniKANTransformer(
    num_tokens=num_tokens,
    dim=dim,
    depth=depth,
    heads=heads,
    dim_head=dim_head,
    attn_dropout=attn_dropout,
    ff_dropout=ff_dropout,
    use_mem_delta_rule=use_mem_delta_rule,
    kan_implementation=kan_implementation,
)

x = torch.randint(0, num_tokens, (1, 512))

# Forward pass with memory retrieval
logits, memories = model(x, return_new_memories=True) # Set False if you don't want memory

# Output shapes
print(f"Logits shape: {logits.shape}")  # (1, 512, 10000)
print(f"Number of layers with memories: {len(memories)}")  # 12 (number of layers)
print(f"Memory shape (KV matrix): {memories[0][0].shape}")  # (1, 8, 64, 64) (batch, heads, dim_head, dim_head)
print(f"Memory shape (Normalization vector): {memories[0][1].shape}")  # (1, 8, 64) (batch, heads, dim_head)

```
