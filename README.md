![CAME_code](came.png)

# CAME 

This repository provides a script and recipe to train the BERT model with our proposed CAME optimizer in:

CAME: Confidence-guided Adaptive Memory Efficient Optimization

This work has been accepted by ACL2023 main conference (***Outstanding Paper Award***).

In this work, we studied a confidence-guided strategy to reduce the instability of existing memory efficient optimizers. Based on this strategy, we proposed CAME to simultaneously achieve two goals: fast convergence as in traditional adaptive methods, and low memory usage as in memory-efficient methods.

# Install
``` 
pip install came_pytorch
```

# Usage
```
from came_pytorch import CAME
optimizer = CAME(model.parameters(), lr=2e-4, weight_decay=1e-2, betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16))
```


# Experimental Results for Some Language Models

## LLaMA-7B Fine-tuning --- Alpaca-7B

|                | MMLU      | BBH       | WikiText | Hellaswag | TruthfulQA (MC) | BoolQ     | COPA  | WSC       | WIC       |
|----------------|-----------|-----------|----------|-----------|-----------------|-----------|-------|-----------|-----------|
| Alpaca-7B      | 40.21     | **33.97** | 6.74     | 59.76     | **38.89**       | **79.57** | 88.00 | 46.15     | 49.84     | 
| Alpaca-7B-CAME | **40.59** | 33.42     | **6.38** | **59.80** | 38.61           | 79.08     | 88.00 | **49.04** | **50.78** |

## GPT-2 Pre-train

![CAME_gpt2](gpt-2_came.png)
