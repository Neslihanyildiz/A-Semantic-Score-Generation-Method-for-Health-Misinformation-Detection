# Dual-KB GraphRAG + SVM-SMOTE + TomekLinks
### Imbalance Ablation Study on PUBHEALTH

---

## 📋 Project Overview

This project implements **TrumorGPT**, a health misinformation detection pipeline that combines:
- **Dual Knowledge Base (KB) GraphRAG** — separate True/False knowledge bases for discriminative retrieval
- **Semantic Health Knowledge Graph (SHKG)** — entity co-occurrence graph with PageRank scoring
- **SVM-SMOTE + TomekLinks** — hybrid resampling for class imbalance
- **Bagging-AdaBoost classifier** — ensemble learning on 12-dimensional hybrid features
- **Topic-Specific TextRank (TST)** — LDA + BERT fusion for sentence centrality (paper method)

The study evaluates two experimental settings on the **PUBHEALTH** dataset:
1. **Imbalanced (2:1)** — 600 True, 300 False claims
2. **Balanced (1:1)** — 300 True, 300 False claims

---

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| RAM | 8 GB | 16 GB |
| GPU | Optional | CUDA-compatible (T4 or better) |
| Disk Space | ~5 GB | ~10 GB |
| OS | Linux / macOS / Windows | Ubuntu 20.04+ |

---

## ⚙️ Installation Steps

### Step 1 — Clone or Download the Notebook

Place the file `Artificial_Intelligence_Final.ipynb` in your working directory.

```bash
# If using git:
git clone <your-repo-url>
cd <your-repo-folder>

# Or simply copy the .ipynb file into a local folder.
```

---

### Step 2 — Create a Virtual Environment (Recommended)

```bash
python -m venv trumorgpt_env

# Activate on Linux / macOS:
source trumorgpt_env/bin/activate

# Activate on Windows:
trumorgpt_env\Scripts\activate
```

---

### Step 3 — Install All Required Libraries

Run **Cell 1** inside the notebook, or install manually via terminal:

```bash
pip install sentence-transformers datasets matplotlib seaborn scikit-learn \
            pandas numpy xgboost lightgbm imbalanced-learn transformers \
            torch tqdm spacy networkx
```

> **Note:** If you have a CUDA GPU, install PyTorch with CUDA support first:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

---

### Step 4 — Download the spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

> This downloads the small English NLP model used for named entity recognition (NER) and dependency parsing.

---

### Step 5 — Launch Jupyter Notebook

```bash
pip install jupyter  # if not already installed
jupyter notebook
```

Then open `Artificial_Intelligence_Final.ipynb` from the Jupyter interface.

Alternatively, use **JupyterLab**:
```bash
pip install jupyterlab
jupyter lab
```

Or open directly in **VS Code** with the Jupyter extension installed.

---

## 🚀 Execution Order

Run the cells **strictly in order** (Cell 1 → Cell 25). Below is a summary of what each cell does:

| Cell | Step | Description |
|------|------|-------------|
| **1** | Install | Installs all pip packages and downloads spaCy model |
| **2** | Imports | Loads all libraries; initializes spaCy, seaborn theme, display options |
| **3** | Data (Imbalanced) | Downloads PUBHEALTH from HuggingFace; samples 600 True + 300 False claims; splits train/test |
| **4** | Data (Balanced) | Creates 1:1 balanced benchmark from same PUBHEALTH source (300+300) |
| **5** | SHKG | Builds Semantic Health Knowledge Graph (entity co-occurrence + PageRank) from **training data only** |
| **6** | Features (Imbalanced) | Loads SBERT + DeBERTa NLI models; extracts 12-D hybrid features for imbalanced split; caches to `.npz` |
| **7** | Features (Balanced) | Extracts 12-D features for balanced split; caches to `.npz` |
| **8** | Visualization | Plots top entities and KB proximity difference (KB_Diff violin plot) |
| **9** | Visualization | Word count analysis, logic margin scatter, KB_Diff boxplot |
| **10** | Pipeline (4 combos) | Runs all 4 method × balance combinations (Paper Method + Our Method, each on Balanced & Imbalanced) |
| **11** | SMOTE Visualization | Applies SVM-SMOTE + TomekLinks and visualizes feature space transformation |
| **12** | SVM Boundary | Plots SVM decision boundary before vs. after resampling |
| **13** | PageRank (SHKG) | Measures real PageRank convergence on SHKG graph for multiple damping factors |
| **14** | PageRank (TST) | Measures real PageRank convergence on TST semantic graph |
| **15** | Semantic Logic Net | Visualizes top-40 high-contradiction claims as a semantic logic network |
| **16** | Metrics (Imbalanced) | Confusion matrix, metric bar chart, ROC curve for imbalanced results |
| **17** | Metrics (Balanced) | Confusion matrix, metric bar chart, ROC curve for balanced results |
| **18** | Ablation Table | Side-by-side metric comparison and ROC overlay (Imbalanced vs Balanced) |
| **19** | Confidence Plot | Confidence score distribution + qualitative probe on test claims |
| **20** | Explainable AI | Feature importance (Bagging-AdaBoost), LDA separability, PCA variance |
| **21** | 3D PCA | 3D PCA visualization of test feature space |
| **22** | Efficiency Report | Inference latency benchmarking (classifier only) |
| **23** | Academic Note | Contributions, honest results, and known limitations summary |
| **24** | Full Paper Method | TST + Topic-Enhanced Centrality + Jaccard similarity (full paper reproduction) |
| **25** | Final Comparison | Complete comparison table (Paper vs Novelty, Balanced vs Imbalanced) + bar charts |

---

## ⚠️ Important Notes

### Kernel Restart After Cell 1
After running **Cell 1** (installation), you **must restart the Jupyter kernel** before continuing:
- In Jupyter: `Kernel → Restart Kernel`
- Then run Cell 2 onward

### Runtime Expectations
| Cell | Estimated Time |
|------|---------------|
| Cell 1 (install) | 3–10 min |
| Cell 5 (SHKG build) | 1–3 min |
| Cell 6 (feature extraction, imbalanced) | **20–60 min** (NLI inference dominates) |
| Cell 7 (feature extraction, balanced) | **15–40 min** |
| Cell 10 (4 pipeline combos) | 5–15 min |
| Cell 24 (full paper method) | **30–90 min** (TST triple extraction) |

> GPU (T4 or better) reduces Cells 6, 7, and 24 significantly.

### Internet Connection Required
- Cell 3 downloads the PUBHEALTH dataset from HuggingFace (~50 MB)
- Cell 6 downloads `all-MiniLM-L6-v2` SBERT model (~90 MB)
- Cell 6 downloads `cross-encoder/nli-deberta-v3-base` model (~700 MB)

### Feature Caching
Cells 6 and 7 save extracted features to disk:
- `trumorgpt_features_imbalanced.npz`
- `trumorgpt_features_balanced.npz`

If the kernel crashes after extraction, you can reload these with `np.load(...)` instead of re-running the slow extraction step.

---

## 📁 Output Files Generated

| File | Generated In | Description |
|------|-------------|-------------|
| `trumorgpt_features_imbalanced.npz` | Cell 6 | Cached 12-D features (imbalanced split) |
| `trumorgpt_features_balanced.npz` | Cell 7 | Cached 12-D features (balanced split) |
| `fig1_hybrid_space.png` | Cell 11 | Feature space before/after SMOTE |
| `fig_svmsmote_boundary.png` | Cell 12 | SVM decision boundary comparison |
| `fig_pubhealth_metrics.png` | Cell 16 | Metrics visualization (imbalanced) |
| `fig_balanced_metrics.png` | Cell 17 | Metrics visualization (balanced) |
| `fig_ablation_comparison.png` | Cell 18 | Ablation bar chart + ROC overlay |
| `fig_feature_importance.png` | Cell 20 | Feature importance bar chart |
| `fig_pca3d.png` | Cell 21 | 3D PCA feature space |
| `fig_final_comparison.png` | Cell 25 | Final comparison bar charts |

---

## 🏗️ Pipeline Architecture

```
PUBHEALTH Dataset
       │
       ├─── Imbalanced Split (600T / 300F)
       │         │
       │    SHKG (train-only) ──► PageRank scores
       │         │
       │    Dual-KB (True KB + False KB)
       │         │
       │    SBERT + DeBERTa NLI ──► 12-D Feature Vector
       │         │
       │    SVM-SMOTE + TomekLinks
       │         │
       │    Bagging(AdaBoost(DecisionTree))
       │         │
       │    Threshold Tuning ──► Final Predictions
       │
       └─── Balanced Split (300T / 300F)
                 └─── Same pipeline above
```

---

## 📊 Feature Vector (12-D)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `cos_true` | Cosine similarity to nearest True KB entry |
| 2 | `sup_true` | NLI "Supports" score vs. True KB context |
| 3 | `con_true` | NLI "Contradicts" score vs. True KB context |
| 4 | `cos_false` | Cosine similarity to nearest False KB entry |
| 5 | `sup_false` | NLI "Supports" score vs. False KB context |
| 6 | `con_false` | NLI "Contradicts" score vs. False KB context |
| 7 | `wc_norm` | Normalized word count (clipped to [0,1]) |
| 8 | `graph_score` | Mean PageRank of claim entities in SHKG |
| 9 | `hedge_score` | Hedging word density |
| 10 | `sens_score` | Sensational word density |
| 11 | `num_score` | Numeric expression density |
| 12 | `question_flag` | Binary: does the claim contain "?"? |

---

## 🔬 Experimental Design

| Combination | Method | Dataset | Resampling |
|-------------|--------|---------|------------|
| 1 | Paper Method (Jaccard GraphRAG) | Balanced (1:1) | None |
| 2 | Paper Method (Jaccard GraphRAG) | Imbalanced (2:1) | SVM-SMOTE + TomekLinks |
| 3 | Our Method (Dual-KB + Bagging) | Balanced (1:1) | None |
| 4 | Our Method (Dual-KB + Bagging) | Imbalanced (2:1) | SVM-SMOTE + TomekLinks |

---

## 🐞 Troubleshooting

**`ModuleNotFoundError`** → Run Cell 1 again and restart kernel.

**`HuggingFace dataset download fails`** → Check internet connection; Cell 3 has a fallback `pd.read_parquet` URL.

**`SVMSMOTE` error with small minority class** → The code automatically adjusts `k_neighbors` to `min(5, minority_count - 1)`.

**`PowerIterationFailedConvergence`** → Already handled in TST code with a fallback uniform distribution.

**Out of memory on NLI inference** → Set `device=-1` to force CPU, or reduce dataset size in Cell 3.

---

## 📚 Dependencies

```
sentence-transformers
datasets
matplotlib
seaborn
scikit-learn
pandas
numpy
xgboost
lightgbm
imbalanced-learn
transformers
torch
tqdm
spacy (+ en_core_web_sm)
networkx
```

---

## 👤 Author Notes

- SHKG is built from **training data only** to prevent test-set leakage into PageRank scores.
- Self-match exclusion during training feature extraction uses index-based lookup (not similarity threshold) for reliability.
- No direct comparison to GPT-3.5/LLaMA/Claude/Gemini baselines is made, as those numbers come from different datasets and splits.
- The `en_core_web_sm` model does **not** include a `DISEASE` entity type; biomedical NER (e.g., scispaCy `en_ner_bc5cdr_md`) would improve `graph_score` quality.

---


## Team

| Name | GitHub | Role |
|------|--------|------|
| Sakarie Hussein Ali | [@sack-ali](https://github.com/sack-ali) | Pipeline architecture, graph construction, GraphRAG |
| Neslihan Yıldız | [@Neslihanyildiz](https://github.com/Neslihanyildiz) | Data preprocessing, BERT embeddings, evaluation |
| Sema Nur Aktaş | [@semanurakts](https://github.com/semanurakts)| Topic modeling, TST, NLI reasoning |

**Institution:** Üsküdar University — Faculty of Engineering and Natural Sciences
**Course:** Artificial Intelligence
**Year:** 2024–2025
