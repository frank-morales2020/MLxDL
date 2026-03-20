[![Status](https://img.shields.io/badge/Status-Active%20Development-orange?style=flat-square)](https://github.com/frank-morales2020/MLxDL)
[![Stars](https://img.shields.io/github/stars/frank-morales2020/MLxDL?style=flat-square&color=yellow)](https://github.com/frank-morales2020/MLxDL/stargazers)
[![Forks](https://img.shields.io/github/forks/frank-morales2020/MLxDL?style=flat-square&color=lightgrey)](https://github.com/frank-morales2020/MLxDL/network/members)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Hugging Face Models](https://img.shields.io/badge/HF%20Models-47-blue?style=flat-square&logo=huggingface)](https://huggingface.co/frankmorales2020)
[![Last Commit](https://img.shields.io/github/last-commit/frank-morales2020/MLxDL?style=flat-square)](https://github.com/frank-morales2020/MLxDL/commits/main)

# 🚀 MLxDL: Advanced Machine Learning & Deep Learning Ecosystem

A comprehensive, actively maintained laboratory for cutting-edge AI research and implementation. It bridges theoretical exploration (Agentic AI, World Models/JEPA) with practical, production-oriented systems — including multi-agent orchestration, domain-specific LLM fine-tuning (aviation, medical, finance), and governance frameworks.

This repo is the source code hub for prototypes and the origin of **47 published models** on Hugging Face.

## 🌟 Key Highlights

### 🤖 1. Agentic AI & Multi-Agent Systems
Implementation of autonomous agents with complex reasoning, tool use, and accountability.
- **H2E Framework**: Engineering "Provable Agency" for alignment, human-to-expert transitions, and governance.
- Multi-Agent Orchestration: Demos using LangGraph, CrewAI, Claude, Gemini, and DeepSeek for parallel execution and technical reviews.
- **Agentic RAG**: Advanced retrieval with reasoning loops, self-correction, and OODA-style planning.

### 🧠 2. World Models & JEPA
Exploring predictive architectures beyond generative text.
- **V-JEPA & LEJEPA**: Implementations of Yann LeCun’s Joint-Embedding Predictive Architecture.
- Focus on latent-space reasoning, "Sketched Isotropic Gaussian Regularization" (SIGReg), and proactive planning.

### 🏢 3. Industrial LLM Applications
- **Medical AGI**: DeepSeek and reasoning models applied to radiology, oncology, MEDAL/MedmcQA fine-tunes.
- **Aviation AI**: Flight planning optimization, waypoints, airspace restriction management.
- **Financial Reasoning**: BTC/crypto market analysis, algorithmic trading bots.

## 📦 Published Models on Hugging Face (47 total)

All models are fine-tuned using notebooks in this repo (LoRA/PEFT, flash-attention, GGUF conversions, distillation). Primarily based on Mistral-7B, Llama-3.x, DeepSeek, etc.

Full profile: [https://huggingface.co/frankmorales2020](https://huggingface.co/frankmorales2020) (sorted by recently updated)

### Classification by Domain – Expand for Full List

<details>
<summary>1. Text-to-SQL / Database Query Generation (~13 models)</summary>

- [Mistral-7B-text-to-sql](https://huggingface.co/frankmorales2020/Mistral-7B-text-to-sql)
- [Mistral-7B-text-to-sql-flash-attention-2](https://huggingface.co/frankmorales2020/Mistral-7B-text-to-sql-flash-attention-2)
- [Mistral-7B-text-to-sql-without-flash-attention-2](https://huggingface.co/frankmorales2020/Mistral-7B-text-to-sql-without-flash-attention-2)
- [Mistral-7B-text-to-sql-flash-attention-2-dataeval](https://huggingface.co/frankmorales2020/Mistral-7B-text-to-sql-flash-attention-2-dataeval)
- [Mistral-7B-text-to-sql-flash-attention-2-FAISS](https://huggingface.co/frankmorales2020/Mistral-7B-text-to-sql-flash-attention-2-FAISS)
- [Mistral-7B-text-to-sql-flash-attention-2-FAISS-10epoch](https://huggingface.co/frankmorales2020/Mistral-7B-text-to-sql-flash-attention-2-FAISS-10epoch)
- [Mistral-7B-text-to-sql-flash-attention-2-FAISS-NEWPOC](https://huggingface.co/frankmorales2020/Mistral-7B-text-to-sql-flash-attention-2-FAISS-NEWPOC)
- [Meta-Llama-3-8B-text-to-sql-flash-attention-2](https://huggingface.co/frankmorales2020/Meta-Llama-3-8B-text-to-sql-flash-attention-2)
- [deepseek-llm-7b-base-spider](https://huggingface.co/frankmorales2020/deepseek-llm-7b-base-spider)
- [deepseek_r1_text2sql_finetuned](https://huggingface.co/frankmorales2020/deepseek_r1_text2sql_finetuned)
- [Mistral-7B-text-to-RLHF](https://huggingface.co/frankmorales2020/Mistral-7B-text-to-RLHF)
- Related variants/experiments

Related notebooks: `AGENTIC_T2SQL_DEMO.ipynb`, `FineTuning_LLM-Mistral-7B...text-to-SQL.ipynb`

</details>

<details>
<summary>2. Aviation / Flight Planning (~6 models)</summary>

- [Mistral-7B-v0.1_AviationQA](https://huggingface.co/frankmorales2020/Mistral-7B-v0.1_AviationQA)
- [Meta-Llama-3-8B_AviationQA-cosine](https://huggingface.co/frankmorales2020/Meta-Llama-3-8B_AviationQA-cosine)
- [FlightPlan_Transformer_LLM](https://huggingface.co/frankmorales2020/FlightPlan_Transformer_LLM)
- [FlightPlan_Transformer_LLM_1GPU_Colab](https://huggingface.co/frankmorales2020/FlightPlan_Transformer_LLM_1GPU_Colab)
- [flight_plan_waypoints_finetuned](https://huggingface.co/frankmorales2020/flight_plan_waypoints_finetuned)
- Llama-3.1 orchestrator variants tied to aviation reasoning

Related notebooks: Aviation fine-tune scripts, waypoint processing

Dataset: [flight_plan_waypoints](https://huggingface.co/datasets/frankmorales2020/flight_plan_waypoints)

</details>

<details>
<summary>3. Medical / Biomedical QA (~8 models)</summary>

- [Mistral-7B-v0.1_MedmcQA](https://huggingface.co/frankmorales2020/Mistral-7B-v0.1_MedmcQA)
- [Mistral-7B-v0.1_McGill-MEDAL](https://huggingface.co/frankmorales2020/Mistral-7B-v0.1_McGill-MEDAL)
- [Meta-Llama-3-8B-MEDAL-flash-attention-2](https://huggingface.co/frankmorales2020/Meta-Llama-3-8B-MEDAL-flash-attention-2)
- [Meta-Llama-3-8B-MEDAL-flash-attention-2-cosine](https://huggingface.co/frankmorales2020/Meta-Llama-3-8B-MEDAL-flash-attention-2-cosine)
- [NEW-Meta-Llama-3-8B-MEDAL-flash-attention-2-cosine-evaldata](https://huggingface.co/frankmorales2020/NEW-Meta-Llama-3-8B-MEDAL-flash-attention-2-cosine-evaldata)
- [POC-Meta-Llama-3-8B-MEDAL-flash-attention-2-cosine-evaldata](https://huggingface.co/frankmorales2020/POC-Meta-Llama-3-8B-MEDAL-flash-attention-2-cosine-evaldata)
- [2025-Meta-Llama-3-8B-MEDAL-flash-attention-2-cosine](https://huggingface.co/frankmorales2020/2025-Meta-Llama-3-8B-MEDAL-flash-attention-2-cosine)
- Dated variants (e.g., 11 APRIL 2025, 07 MAY 2025)

Related notebooks: `FineTuning_Llama3_MEDAL.ipynb`, `AAI_MEDICAL_DEEPSEEK.ipynb`

</details>

<details>
<summary>4. Cryptocurrency / BTC Expert (~3 models)</summary>

- [Mistral-7B-BTC-Expert](https://huggingface.co/frankmorales2020/Mistral-7B-BTC-Expert)
- [Mistral-7B-BTC-JEPA-LLM-Expert](https://huggingface.co/frankmorales2020/Mistral-7B-BTC-JEPA-LLM-Expert)
- Related expert/distilled variants

Related notebooks: `BTC_AAI_LLM_BOT.ipynb`, `FINAL_LLM_JEPA_MISTRAL_FT_BTC.ipynb`

</details>

<details>
<summary>5. Other Specialized Domains (~17 models)</summary>

- [Voxtral-Mini-4B-H2E-FineTune](https://huggingface.co/frankmorales2020/Voxtral-Mini-4B-H2E-FineTune) – Speech
- [Mistral-7B-Philosophy-H2E](https://huggingface.co/frankmorales2020/Mistral-7B-Philosophy-H2E) – Philosophy
- [akkadian-to-english-translator](https://huggingface.co/frankmorales2020/akkadian-to-english-translator)
- [kkadian-to-spanish-translator](https://huggingface.co/frankmorales2020/kkadian-to-spanish-translator)
- [Llama-3.1-8B-Orchestrator-GGUF](https://huggingface.co/frankmorales2020/Llama-3.1-8B-Orchestrator-GGUF)
- [unsloth-DeepSeek-R1-Distill-Llama-8B-mental_health_counseling](https://huggingface.co/frankmorales2020/unsloth-DeepSeek-R1-Distill-Llama-8B-mental_health_counseling) – Mental Health
- [Mistral-7B-v0.1_Emotion](https://huggingface.co/frankmorales2020/Mistral-7B-v0.1_Emotion)
- [lora_fine_tuned_phi-4_quantized_vision](https://huggingface.co/frankmorales2020/lora_fine_tuned_phi-4_quantized_vision)
- [torchtune-Llama-2-7b](https://huggingface.co/frankmorales2020/torchtune-Llama-2-7b)
- [my-awesome-setfit-model](https://huggingface.co/frankmorales2020/my-awesome-setfit-model)
- [bert-base-cased_fine_tuned_glue_cola](https://huggingface.co/frankmorales2020/bert-base-cased_fine_tuned_glue_cola)
- [mistral-7b-alpha-finetuned-llm-science-exam-tpu-colab-v6e-1](https://huggingface.co/frankmorales2020/mistral-7b-alpha-finetuned-llm-science-exam-tpu-colab-v6e-1)
- [gpt-oss-20b-multilingual-reasoner](https://huggingface.co/frankmorales2020/gpt-oss-20b-multilingual-reasoner)
- [mistral-7b-qwen-Next-80B-A3B-Instruct-distilled](https://huggingface.co/frankmorales2020/mistral-7b-qwen-Next-80B-A3B-Instruct-distilled)
- [mistral-7b-gpt-oss-20b-distilled](https://huggingface.co/frankmorales2020/mistral-7b-gpt-oss-20b-distilled)
- General/distillation/experimental variants (e.g., squad_alignment, dialogsum tests)

</details>

## 📂 Core Repository Structure & Top Impactful Notebooks

Many include **Open in Colab** buttons.

| Rank | Notebook                              | Description / Why Impactful                                                                 |
|------|---------------------------------------|---------------------------------------------------------------------------------------------|
| 1    | `JEPA_AGI_DEMO.ipynb`                 | Flagship World Models demo — latent predictive reasoning inspired by Yann LeCun.           |
| 2    | `AAI_10LEVEL_DEMO.ipynb`              | Roadmap for 10 levels of Agentic AI capability and complexity.                              |
| 3    | `DEEPSEEK_R1_DISTILL_QWEN_7B_COLAB.ipynb` | Distillation of reasoning capabilities into efficient models — trending for local use.     |
| 4    | `AGENTIC_T2SQL_DEMO.ipynb`            | Self-correcting Text-to-SQL agents with practical business utility.                         |
| 5    | `CAG_DeepSeek_Mistral_Gemini.ipynb`   | Cache-Augmented Generation (CAG) vs RAG comparisons across top models.                      |

Additional: `FineTuning_Llama3_MEDAL.ipynb`, `BTC_AAI_LLM_BOT.ipynb`, `H2E_Holonomic_Integration.ipynb`

## 🛠️ Tech Stack
- **Frameworks**: LangChain, LangGraph, CrewAI, Haystack, Hugging Face Transformers, PEFT/LoRA.
- **Models**: DeepSeek-V3/R1, Llama 3.x, Mistral, Claude, Gemini, Qwen.
- **Infrastructure**: Google Colab (GPU/TPU), AWS, PostgreSQL, ChromaDB/FAISS.

## 👨‍🔬 About the Author
**Frank Morales** — Former Boeing Associate Technical Fellow, SMIEEE, Global Top 10 Thought Leader in Agentic AI & Open Source. Founder & Lead AI Researcher at Sovereign Machine Lab Association (SOMALA).

- **Medium**: [AI Simplified in Plain English](https://medium.com/@frankmorales_91352)
- **Thinkers360**: [Profile](https://www.thinkers360.com/tl/profiles/view/25153)
- **LinkedIn**: [Frank Morales](https://www.linkedin.com/in/frank-morales1964/)

## 🤝 Contributing
Contributions welcome!
1. Fork the repo.
2. Create branch: `git checkout -b feature/AmazingFeature`
3. Commit: `git commit -m 'Add AmazingFeature'`
4. Push: `git push origin feature/AmazingFeature`
5. Open Pull Request.

## 📜 License
Distributed under the MIT License. See [LICENSE](LICENSE).

**"Moving from reactive capability to provable integrity."** — *The H2E Vision*
