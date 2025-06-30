# Adding two small numbers with a multi-million parameter Transformer

> When you have a hammer - everything looks like a nail
>
> -- <cite>Abraham Maslow</cite>

In this spirit, we will explore learning to add two numbers using transformer based large language models (LLMs) such as General Purpose Transformers (GPT). This is relevant, since even the today most advanced LLMs still struggle with simple math. Specifically, I plan to explore the following topics:

- [ ] proof that GPT is capable of learning an algorithm to add numbers
- [ ] show in a toy example that the algorithm is not learned using traditional means.
- [x] demonstrate how to use PyTorch's built in transformer components

## Usage

Run the `pytorch_transformer_addition.py` script to train a tiny transformer on
random addition problems using `torch.nn.Transformer`. The example prints the
loss during training and shows a sample prediction:

```bash
python src/pytorch_transformer_addition.py
```
