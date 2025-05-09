# Prosense: Defending Text Generation with Adversarial Feedback

Prosense is a multi-stage training framework designed to improve the robustness of large language models (LLMs) against adversarial inputs. It fine-tunes quantized LLMs using clean and adversarially augmented datasets, evaluates model performance via reasoning-based judgments, and refines the model through structured adversarial feedback using **Graph-of-Thought (GOT)** parsing.

> 🚀 **Model Base**: `unsloth/mistral-7b-bnb-4bit`  
> 🧠 **Goal**: Increase factuality and robustness through adversarial feedback  
> 🧪 **Datasets**: Open-Instruct, TruthfulQA, GOT-parsed failures

---

## 📁 Project Structure

```
Prosense-Adversarial-Robustness/
│
├── Phase1_Clean_FineTuning/
│   └── phase1.ipynb
│
├── Phase2_Adversarial_Hybrid/
│   ├── HybridDataCreation.ipynb
│   ├── MergingWithHybridDataset.ipynb
│   └── Phase2_Final.ipynb
│
├── Phase3_Evaluation_Parsing/
│   ├── Phase3_1_Collect_TruthfulQA_Responses.ipynb
│   ├── Phase3_2_Judge_and_Filter_Failures.ipynb
│   ├── Phase3_3_Parse_GOT_Graph_By_LLaMA.ipynb
│   └── Phase3_Final.ipynb
│
└── Phase3_Level2_Refinement/
    ├── Phase4_1_Collect_Level1_Responses.ipynb
    ├── Phase4_2_Judge_Parse_Level1_By_LLaMA.ipynb
    ├── Phase4_3_Level2_Finetune.ipynb
    ├── Phase4_4_Collect_Level2_Responses.ipynb
    ├── Phase4_5_Judge_Level2_Responses.ipynb
    └── Phase4_6_Reasoning_Graph_Visualization.ipynb
```

---

## 📌 Methodology Overview

### 🔹 Phase 1 – Baseline Fine-Tuning
Fine-tunes Mistral-7B (4-bit quantized) on the Open-Instruct dataset.

> ✅ Output: `mistral-prosense-clean` – Clean model baseline

---

### 🔹 Phase 2 – Adversarial Hybrid Fine-Tuning
- Generates adversarial examples from clean data
- Merges clean + adversarial sets
- Re-trains model to enhance robustness

> ✅ Output: `mistral-prosense-hybrid` – Robustness-enhanced model

---

### 🔹 Phase 3 – TruthfulQA Evaluation + GOT Parsing
- Evaluates the hybrid model on TruthfulQA
- Judges failed samples with another LLM (Gemma or LLaMA)
- Parses failures into **Graph-of-Thought (GOT)** reasoning format
- Fine-tunes again using these structured reasoning examples

> ✅ Output: `mistral-prosense-parsed` – GOT-aware model

---

### 🔹 Phase 3 Again – Level 2 Feedback Cycle
- Re-evaluates the parsed model (Level 1)
- Parses second round of failed cases
- Final fine-tuning with deeper feedback
- Visualizes reasoning improvement via GOT graphs

> ✅ Output: `mistral-finetuning-again` – Final robust model

---

## ⚙️ Dependencies

Install required packages:

```bash
pip install transformers accelerate datasets bitsandbytes unsloth textattack torch
```

> Python 3.10+ and a 24GB+ VRAM GPU are recommended for fine-tuning.

---

## 🚀 How to Run

Run notebooks in order:

### Phase 1: Clean Fine-Tuning
1. `phase1.ipynb`

### Phase 2: Adversarial Training
2. `HybridDataCreation.ipynb`
3. `MergingWithHybridDataset.ipynb`
4. `Phase2_Final.ipynb`

### Phase 3: Evaluation + GOT Parsing
5. `Phase3_1_Collect_TruthfulQA_Responses.ipynb`
6. `Phase3_2_Judge_and_Filter_Failures.ipynb`
7. `Phase3_3_Parse_GOT_Graph_By_LLaMA.ipynb`
8. `Phase3_Final.ipynb`

### Phase 4: Level 2 Feedback Cycle
9. `Phase4_1_Collect_Level1_Responses.ipynb`
10. `Phase4_2_Judge_Parse_Level1_By_LLaMA.ipynb`
11. `Phase4_3_Level2_Finetune.ipynb`
12. `Phase4_4_Collect_Level2_Responses.ipynb`
13. `Phase4_5_Judge_Level2_Responses.ipynb`
14. `Phase4_6_Reasoning_Graph_Visualization.ipynb`

---

## 📊 Evaluation Results 
You can add metrics here if you tracked model performance across stages, such as:

- TruthfulQA Pass Rate: 35% → 55% (Post Phase 3)
- Adversarial Robustness Improvement: +30%

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).  
Feel free to use, adapt, or distribute with proper attribution.

---

