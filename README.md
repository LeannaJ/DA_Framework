# AI Agent for Automated Data Analysis (Ongoing) 🤖

This project explores the use of large language models (LLMs) and Retrieval-Augmented Generation (RAG) to augment the end-to-end workflow of a data analyst. By integrating Cursor's AI agent, LangChain, and modular prompt chaining, we aim to automate rule-based tasks while enabling human judgment where necessary—leading to a more efficient and human-centered analysis process.

---

## 📌 Project Overview

The goal is to design a **modular AI-powered analysis framework** that distinguishes between tasks suitable for automation and those requiring human decision-making. Through a step-by-step prompt workflow, the system guides both the AI and the human analyst through stages such as data loading, cleaning, exploration, modeling, and reporting.

---

## 📊 Approach and Results

### Workflow Design

A visual map was created to outline the entire data analysis pipeline. Each step is tagged as either **rule-based (AI-automatable)** or **judgment-based (human-in-the-loop)**, enabling a flexible collaboration between human and machine.

### Key Features

- **Prompt-driven structure** for loading, cleaning, analyzing, and reporting  
- **Chained prompts** with decision checkpoints (e.g., choosing modeling strategy or selecting features)  
- **Human-in-the-loop design**: allows real-time decisions via chat interaction  
- **Scalability**: Prompt templates built for extensibility across various datasets and domains

### Challenges Faced

- Distinguishing clearly between tasks suitable for full automation vs. human decision-making  
- Designing **adaptable and context-aware** prompts  
- Ensuring logical flow and continuity in chained prompts without losing user intent

### Solutions Implemented

- Built a **workflow chart** visualizing steps in data analysis and their classification  
- Implemented prompt chaining for modular step execution  
- Enabled **interactive decision checkpoints** for user involvement in complex tasks  
- Incorporated **Cursor, LangChain, and LangSmith** to manage agent interactions and debugging

### Results

This project is currently in progress.

- ✅ Initial tests show improved productivity and reduced repetitive work for analysts  
- 🧠 Promising results in collaborative AI + human decision pipelines  
- 🔄 Next steps include:
  - Integrating LangChain agents for advanced reasoning
  - Expanding support for diverse data types
  - Developing **evaluation metrics** to measure effectiveness of AI-augmented workflows

---

## 📁 Deliverables

- 🗂 Code and data (in this repository)  
- ✍️ Prompt templates for each analysis phase  
- 📊 Presentation deck summarizing framework, features, and early findings

---

## 🛠 Tech Stack

- Python  
- Cursor AI  
- LangChain & LangSmith  
- Hugging Face Transformers  
- Retrieval-Augmented Generation (RAG)  
- Pandas, NumPy
