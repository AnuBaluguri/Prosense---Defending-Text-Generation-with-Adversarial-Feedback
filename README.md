# Prosense: Defending Text Generation with Adversarial Feedback

Prosense is a multi-stage training framework designed to improve the robustness of large language models (LLMs) against adversarial inputs. It fine-tunes quantized LLMs using clean and adversarially augmented datasets, evaluates model performance via structured reasoning, and iteratively refines the model through curriculum-based adversarial feedback.

> 🚀 **Model Base**: `unsloth/mistral-7b-bnb-4bit`  
> 🧠 **Goal**: Improve response truthfulness and adversarial resilience via multi-stage fine-tuning  
> 🧪 **Datasets**: Open-Instruct, TruthfulQA

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
│   ├── Phase2_Model_Responses_Collection_Code.ipynb
│   ├── Parsing_GOT_Style_By_Llama_Code.ipynb
│   └── Phase3_Final.ipynb
│
└── Phase3_Level2_Refinement/
    ├── Level1_Model_Responses_Collection_Code.ipynb
    ├── Level1_Model_Responses_Judging&Parsing_Llama_Code.ipynb
    ├── Phase3_Level2_Finetuning_Code.ipynb
    ├── Phase3_Level2_Model_Responses_Collection_Code.ipynb
    └── Reasoning_graphs.ipynb
```

---

## 📌 Methodology & Pipeline

### 🔹 Phase 1 – Baseline Fine-Tuning
Fine-tunes a 4-bit quantized Mistral-7B model using the Open-Instruct dataset.  
Includes preprocessing, tokenization, and model checkpoint saving.

> Output: `mistral-prosense-clean` (baseline model)

---

### 🔹 Phase 2 – Adversarial Hybrid Fine-Tuning
Generates adversarial examples using perturbation-based attacks, merges them with clean samples, and retrains the model on this hybrid dataset.

> Output: `mistral-prosense-hybrid` (model trained on mixed data)

---

### 🔹 Phase 3 – Evaluation, Parsing, and CoT Fine-Tuning
Evaluates the hybrid model on TruthfulQA, collects failed responses, and uses LLaMA to parse them into Chain-of-Thought (CoT) style structured reasoning. This data is used to fine-tune the model further.

> Output: `mistral-prosense-parsed` (CoT-refined model)

---

### 🔹 Phase 3 Again – Level 2 Feedback and Final Evaluation
Repeats evaluation and parsing on the Level 1 model, followed by one more round of fine-tuning and final performance judging. Also generates reasoning graphs for qualitative analysis.

> Output: Final `mistral-finetuning-again` + reasoning graph insights

---

## 🛠️ Dependencies

Create a new environment and install these key packages:

```bash
pip install transformers accelerate datasets bitsandbytes unsloth
pip install textattack  # or OpenAttack if used
```

Python ≥ 3.10 is recommended. GPU with 24GB+ VRAM for fine-tuning.

---

## 🚀 Getting Started

1. **Clone the repository**  
   `git clone https://github.com/your-username/Prosense-Adversarial-Robustness.git`

2. **Set up your environment**  
   Install the dependencies listed above or use a `requirements.txt`.

3. **Run each phase**  
   Open and run the Jupyter notebooks in sequence, starting from Phase 1.

---

## 📊 Evaluation Results (Optional)
You can add metrics here if you tracked model performance across stages, such as:

- TruthfulQA Pass Rate: 35% → 55% (Post Phase 3)
- Adversarial Robustness Improvement: +30%

---


## 📄 License

This project is intended for academic and research use. Please cite appropriately if you reuse or adapt this pipeline.

