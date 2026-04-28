# Tree of Thoughts — Re-implementation

CS 4782 Final Project

## 1. Introduction

This repository re-implements the method introduced in *Tree of Thoughts: Deliberate Problem Solving with Large Language Models* (Yao et al., 2023). ToT generalizes Chain-of-Thought prompting by exploring a tree of intermediate "thoughts" with explicit search (BFS/DFS) and self-evaluation, enabling LLMs to plan, look ahead, and backtrack on tasks that resist single-pass generation.

## 2. Chosen Result

*To be filled in:* the specific table/figure from the paper we target (e.g., Game of 24 success rate in Table 2, or Creative Writing coherency in Figure 6) and why it is central to the paper's claim.

## 3. GitHub Contents

- `code/` — re-implementation source, configs, and scripts.
- `data/` — datasets used for evaluation, or instructions for obtaining them.
- `results/` — generated figures, tables, and run logs.
- `poster/` — final poster PDF.
- `report/` — 2-page final report PDF.
- `LICENSE`, `.gitignore`, `README.md`.

## 4. Re-implementation Details

*To be filled in:* models used (e.g., GPT-4 / GPT-3.5), task implementations (Game of 24, Creative Writing, Mini Crosswords), search strategy (BFS/DFS), value/vote prompts, and evaluation metrics.

## 5. Reproduction Steps

*To be filled in:* environment setup, API keys, dependency install, commands to run each experiment, and required compute.

```bash
# example placeholder
pip install -r code/requirements.txt
python code/run.py --task game24 --method tot
```

## 6. Results / Insights

*To be filled in:* re-implementation results compared against the paper's reported numbers.

## 7. Conclusion

*To be filled in:* key takeaways and lessons from the re-implementation.

## 8. References

- Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models.* NeurIPS 2023.
- Original code: https://github.com/princeton-nlp/tree-of-thought-llm

## 9. Acknowledgements

Completed as the final project for CS 4782 (Cornell University, Spring 2026).
