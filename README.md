# DentalGPT: A Multi-Expert Transformer Model for Dental Inquiry Resolution


## Overview

**DentalGPT** is a specialized conversational AI system designed to deliver accurate, contextually relevant dental advice in Vietnamese. It targets the significant gap in existing language models for domain-specific and non-English healthcare consultation, particularly in **dentistry**.

Developed by **Dat Tran** and **Vu Cao** from *Thuyloi University, Vietnam*, DentalGPT builds on top of the **DeepSeek-R1** model with a **Mixture-of-Experts (MoE)** architecture, fine-tuned on a **custom Vietnamese dental dataset** of ~3 million dialogue samples.

---

## Core Features

- Fine-tuned on **Vietnamese** dental conversations using DeepSeek-R1
- Utilizes:
  - **Supervised Fine-Tuning (SFT)**
  - **QLoRA** for efficient optimization
  - **Reinforcement Learning from Human Feedback (RLHF)**
- Covers medical literature, clinical guidelines, and expert responses
- Tailored for **professional accuracy** and **practical relevance**

---

## Performance Metrics

| Metric           | Score  |
|------------------|--------|
| Perplexity       | 1.88   |
| BLEU Score       | 0.53   |
| BERTScore        | 0.93   |
| MMLU Accuracy    | 91.0   |

**Outperforms GPT-4o** on domain-specific MMLU tasks.

---

## Why DentalGPT?

- Addresses the lack of **Vietnamese-language** medical consultation tools
- User-friendly chatbot interface
- Applicable in digital health platforms
- Reliable dental inquiry resolution engine

---

## Citation (APA Style)

Tran, D., & Cao, V. (2025). *DentalGPT: A Multi-Expert Transformer Model for Dental Inquiry Resolution*. Thuyloi University, Vietnam.

---

## Demo & Usage

**Try the live demo user interface here:**  
[https://nvcoing.github.io/DentalGPT/](https://nvcoing.github.io/DentalGPT/)
---

## Tech Stack

- [DeepSeek-R1](https://huggingface.co/DeepSeekAI/deepseek-llm)
- PyTorch / Transformers
- QLoRA
- RLHF (Reinforcement Learning from Human Feedback)

---

## Acknowledgements

Special thanks to the medical experts, dentists, and data annotators who contributed to the dataset and evaluation.

---

