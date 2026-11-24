# Data

The dataset is not stored in this repository. The notebook downloads it from Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset("NLBSE/nlbse27-code-comment-classification")
```

| Split | Functions | Use in this project |
|---|---|---|
| train | 214,586 | 90% training (193,127), 10% validation (21,459), stratified, seed 42 |
| test | 30,655 | final evaluation only |

Columns: `Projectname`, `Filepath`, `Function`, `Label` (`[vulnerability, SATD]`).
Source: MADE-WIC (Mock et al., ASE 2024), licensed CC BY 4.0.
