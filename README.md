[![Status](https://img.shields.io/badge/Status-Active%20Development-orange?style=flat-square)](https://github.com/frank-morales2020/MLxDL)
[![Stars](https://img.shields.io/github/stars/frank-morales2020/MLxDL?style=flat-square&color=yellow)](https://github.com/frank-morales2020/MLxDL/stargazers)
[![Forks](https://img.shields.io/github/forks/frank-morales2020/MLxDL?style=flat-square&color=lightgrey)](https://github.com/frank-morales2020/MLxDL/network/members)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Hugging Face Models](https://img.shields.io/badge/HF%20Models-47-blue?style=flat-square&logo=huggingface)](https://huggingface.co/frankmorales2020)
[![Last Commit](https://img.shields.io/github/last-commit/frank-morales2020/MLxDL?style=flat-square)](https://github.com/frank-morales2020/MLxDL/commits/main)

# 🚀 MLxDL: Advanced Machine Learning & Deep Learning Ecosystem

This is a comprehensive, actively maintained laboratory for cutting-edge AI research and implementation. It bridges theoretical exploration (Agentic AI, World Models/JEPA) with practical, production-oriented systems — including multi-agent orchestration, domain-specific LLM fine-tuning (aviation, medical, finance), and governance frameworks.

Repository serves as the source code hub for prototypes and the origin of 47 published models on Hugging Face.

## 🌟 Key Highlights

### 🤖 1. Agentic AI & Multi-Agent Systems
Implementation of autonomous agents with complex reasoning, tool use, and accountability.
- **H2E Framework**: Engineering "Provable Agency" for alignment, human-to-expert transitions, and governance.
- Multi-Agent Orchestration: Demos using LangGraph, CrewAI, Claude, Gemini, DeepSeek for parallel execution and technical reviews.
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

Fine-tuned models (mostly Mistral-7B, Llama-3.x, DeepSeek bases) created via notebooks in this repo — using LoRA/PEFT, flash-attention, GGUF, distillation.

Full list: [https://huggingface.co/frankmorales2020](https://huggingface.co/frankmorales2020) (sorted by recently updated)

### Highlighted by Domain
| Domain                        | Approx. Count | Example Models (with links)                                                                 | Related Notebooks                                      |
|-------------------------------|---------------|---------------------------------------------------------------------------------------------|--------------------------------------------------------|
| Text-to-SQL / Database Query  | ~13          | [Mistral-7B-text-to-sql-flash-attention-2](https://huggingface.co/frankmorales2020/Mistral-7B-text-to-sql-flash-attention-2)<br>[deepseek_r1_text2sql_finetuned](https://huggingface.co/frankmorales2020/deepseek_r1_text2sql_finetuned) | `AGENTIC_T2SQL_DEMO.ipynb`                             |
| Aviation / Flight Planning    | ~6           | [Mistral-7B-v0.1_AviationQA](https://huggingface.co/frankmorales2020/Mistral-7B-v0.1_AviationQA)<br>[FlightPlan_Transformer_LLM](https://huggingface.co/frankmorales2020/FlightPlan_Transformer_LLM) | Aviation & waypoint fine-tune scripts                  |
| Medical / Biomedical QA       | ~8           | [Meta-Llama-3-8B-MEDAL-flash-attention-2-cosine](https://huggingface.co/frankmorales2020/Meta-Llama-3-8B-MEDAL-flash-attention-2-cosine)<br>[Mistral-7B-v0.1_MedmcQA](https://huggingface.co/frankmorales2020/Mistral-7B-v0.1_MedmcQA) | `FineTuning_Llama3_MEDAL.ipynb`                        |
| Cryptocurrency / BTC          | ~3           | [Mistral-7B-BTC-Expert](https://huggingface.co/frankmorales2020/Mistral-7B-BTC-Expert)<br>[Mistral-7B-BTC-JEPA-LLM-Expert](https://huggingface.co/frankmorales2020/Mistral-7B-BTC-JEPA-LLM-Expert) | `BTC_AAI_LLM_BOT.ipynb`                                |
| Other Specialized             | ~17          | [Voxtral-Mini-4B-H2E-FineTune](https://huggingface.co/frankmorales2020/Voxtral-Mini-4B-H2E-FineTune) (Speech)<br>[Mistral-7B-Philosophy-H2E](https://huggingface.co/frankmorales2020/Mistral-7B-Philosophy-H2E) | Distillation, translator, vision notebooks             |

Dataset: [flight_plan_waypoints](https://huggingface.co/datasets/frankmorales2020/flight_plan_waypoints)

## 📂 Core Repository Structure & Top Impactful Notebooks

Many notebooks include "Open in Colab" buttons for instant execution.

| Rank | Notebook                              | Description / Why Impactful                                                                 |
|------|---------------------------------------|---------------------------------------------------------------------------------------------|
| 1    | `JEPA_AGI_DEMO.ipynb`                 | Flagship World Models demo — latent predictive reasoning inspired by Yann LeCun.           |
| 2    | `AAI_10LEVEL_DEMO.ipynb`              | Roadmap for 10 levels of Agentic AI capability and complexity.                              |
| 3    | `DEEPSEEK_R1_DISTILL_QWEN_7B_COLAB.ipynb` | Distillation of reasoning capabilities into efficient models — trending for local use.     |
| 4    | `AGENTIC_T2SQL_DEMO.ipynb`            | Self-correcting Text-to-SQL agents with practical business utility.                         |
| 5    | `CAG_DeepSeek_Mistral_Gemini.ipynb`   | Cache-Augmented Generation (CAG) vs RAG comparisons across top models.                      |

Additional standouts: `FineTuning_Llama3_MEDAL.ipynb` (medical domain), `BTC_AAI_LLM_BOT.ipynb` (finance), `H2E_Holonomic_Integration.ipynb` (governance core).

## What's New / Recent Activity
- Latest commit: Mar 5, 2026 (Colab-based updates).
- Ongoing: New fine-tunes, agent demos, and JEPA extensions.
- HF activity: Most recent models include Voxtral speech fine-tune and Philosophy expert (updated mid-Feb 2026).

## 🛠️ Tech Stack
- **Frameworks**: LangChain, LangGraph, CrewAI, Haystack, Hugging Face Transformers, PEFT/LoRA.
- **Models**: DeepSeek-V3/R1, Llama 3.x, Mistral, Claude, Gemini, Qwen.
- **Infrastructure**: Google Colab (GPU/TPU), AWS, PostgreSQL, ChromaDB/FAISS.

## 👨‍🔬 About the Author
**Frank Morales** — Former Boeing Associate Technical Fellow, SMIEEE, Global Top 10 Thought Leader in Agentic AI & Open Source. Founder & Lead AI Researcher at Sovereign Machine Lab Association (SMLA).

- **Medium**: [AI Simplified in Plain English](https://medium.com/@frankmorales_91352)
- **Thinkers360**: [Profile](https://www.thinkers360.com/tl/profiles/view/25153)
- **LinkedIn**: [Frank Morales](https://www.linkedin.com/in/frank-morales1964/)

## 🤝 Contributing
Contributions welcome — new agents, fine-tunes, bug fixes, or patterns!
1. Fork the repo.
2. Create branch: `git checkout -b feature/AmazingFeature`
3. Commit: `git commit -m 'Add AmazingFeature'`
4. Push: `git push origin feature/AmazingFeature`
5. Open Pull Request.

## 📜 License
Distributed under the MIT License. See [LICENSE](LICENSE).

**"Moving from reactive capability to provable integrity."** — *The H2E Vision*
