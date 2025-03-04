# Numeral-aware Headline Generation

This project focuses on generating news headlines that accurately incorporate numerical information from source articles. Based on the [SemEval-2024 Task 7](https://aclanthology.org/2024.semeval-1.182/), this implementation fine-tunes Meta's Llama 3.1-8B-Instruct model to generate headlines with correct numerical reasoning.

## Project Overview

Numerical information in headlines significantly impacts reader perception and engagement. For example, "Stock price to increase by 30%" conveys a different level of significance compared to "Stock price to increase by 3%". This project aims to:

1. Extract key numerical information from news articles
2. Perform necessary numerical reasoning to ensure factual accuracy
3. Generate concise, informative headlines incorporating this numerical data
4. Evaluate headline quality using both automated metrics and peer assessment

## Dataset

The project uses the [NumHG dataset](https://arxiv.org/abs/2309.01455), which contains:
- News articles paired with numerical-focused headlines
- Numerical content annotations
- Ground truth numbers for evaluation

## Methodology

Three approaches were implemented and compared:

1. **Base Model**: Meta's Llama 3.1-8B-Instruct without fine-tuning
2. **QLoRA Fine-tuned Model**: The base model fine-tuned on the NumHG dataset using QLoRA (Quantized Low-Rank Adaptation)
3. **Chain-of-Thought Fine-tuned Model**: The fine-tuned model with additional prompting guidance to enhance numerical reasoning

### Data Preprocessing

- Formatted input texts and target headlines as conversation pairs
- Applied Llama 3.1's chat template for tokenization and labeling
- Structured the data to optimize for headline generation tasks

### Fine-Tuning Details

QLoRA was used to efficiently update model parameters with the following configurations:
- LoRA Rank = 16
- LoRA Alpha = 16
- Target Modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
- Epochs = 2
- Batch Size = 32 with Gradient Accumulation = 8 (effective batch size of 256)
- Learning Rate = 2e-5
- Weight Decay = 0.01

### Chain-of-Thought Prompting

For the third approach, a structured prompt was used to guide the model's reasoning:

```
Generate a single headline for this news article that includes at least one key numeric feature:

{article body}

When generating the headline consider the following:
1. What is the subject of the article?
2. What is the sentiment of the article?
3. Does the headline accurately portray the subject and sentiment of the article?
4. Is the headline an appropriate length?
5. Is the key numeric feature formatted as a number?

Be sure to return only the generated headline without any enclosing quotation marks.
```

## Results

The evaluation used several metrics to assess headline quality and numerical accuracy:

| Metric | Base Model | Fine-Tuned Model | FT + Chain-of-Thought |
|--------|------------|------------------|------------------------|
| Rouge-1 | 0.320807 | 0.44197 | 0.435587 |
| Rouge-2 | 0.110107 | 0.191655 | 0.186737 |
| Rouge-L | 0.273208 | 0.433951 | 0.385779 |
| All Accuracy | 0.314291 | 0.530726 | 0.538115 |
| Copy Accuracy | 0.433953 | 0.572497 | 0.571465 |
| Cal Accuracy | 0.037059 | 0.433951 | 0.460849 |
| MoverScore | 0.144228 | 0.257495 | 0.251104 |
| BERTScore F1 | 0.364521 | 0.465213 | 0.458261 |

Key findings:
- Fine-tuning significantly improved all metrics compared to the base model
- Chain-of-thought prompting further improved calculation accuracy
- The fine-tuned models showed particular improvement in numerical reasoning tasks

## Repository Structure

- `Final Evaluation Code.ipynb`: Notebook for evaluating model performance
- `Final Modeling Notebook.ipynb`: Notebook for model preparation and fine-tuning
- Additional documentation and presentation materials

## Future Work

Potential improvements identified for future iterations:
- Fine-tuning with alternative input formats (masked headlines, calculation targets)
- Multi-stage fine-tuning approach for better numerical reasoning
- Optimizing chain-of-thought prompts programmatically
- Expanding training to all five folds of the NumHG dataset

## References

1. Chen, C. C., Huang, J. T., Huang, H. H., Takamura, H., & Chen, H. H. (2024). SemEval-2024 Task 7: Numeral-aware language understanding and generation. In *Proceedings of the 18th International Workshop on Semantic Evaluation*, 1482-1491.

2. Huang, J. T., Chen, C. C., Huang, H. H., & Chen, H. H. (2023). NumHG: A dataset for number-focused headline generation. *arXiv preprint arXiv:2309.01455*.
