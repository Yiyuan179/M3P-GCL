# M<sup>3</sup>P-GCL

PyTorch Implementation for "Beyond Homophily: Graph Contrastive Learning with Macro-Micro Message Passing ".

## Description

we propose M<sup>3</sup>P-GCL framework to address the limitations of homophily assumption in current GCL frameworks by introducing an Aligned Priority-Supporting View Encoding (APS-VE) strategy for structual and attribute views at the macro-level, and an Adaptive Self-Propagation (ASP) strategy for self-loop at the micro-level. These innovations diversify the message passing mechanism, enabling M<sup>3</sup>P-GCL to enhance performance in homophilous and non-homophilous graphs.

<div style="text-align: center;">
    <img src="./framework.png" />
    <p><strong>Figure:</strong> The overview of M<sup>3</sup>P-GCL framework.</p>
</div>

## Overview

- `.\model.py`: M<sup>3</sup>P-GCL model implementation.
- `.\train.py`: Model training.
- `.\run.sh`: Reproduction script for experimental results across 7 datasets.

## Requirements

```
numpy==1.26.2
scikit_learn==1.4.0
torch==2.0.1
torch-geometric==2.5.3
```

## Code Reference
This model is developed based on [JialuChenChina/ASP: The code of "Attribute and Structure preserving Graph Contrastive Learning" (AAAI 2023 oral)](https://github.com/JialuChenChina/ASP).