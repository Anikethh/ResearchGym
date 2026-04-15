## Research Goal
Typical language models used in materials science rely on frequency-centric tokenization methods developed for natural language, which often leads to excessive fragmentation and semantic loss of material concepts. These methods fail to maintain the structural and semantic integrity of important domain-specific terms, such as material names and chemical formulas, because they tend to have low frequencies in corpora. This fragmentation can cause language models to misinterpret the meaning of material concepts, leading to performance degradation. The misrepresentation of material concepts due to improper tokenization hinders the performance of language models on specialized materials science tasks. Preserving the integrity of domain-specific subwords is crucial for maintaining model effectiveness. By developing a tokenization strategy that understands and prioritizes material terminology, language models can more accurately learn domain-specific concepts, accelerating materials discovery and research through more effective text analysis.

## Experimental Settings
*   **Backbone Model**: SciBERT for all experiments.
*   **Vocabulary**: Fixed size of 31,090 for all tokenization methods.
*   **Downstream Tasks & Datasets**:
    *   **Generation**: MatSci-NLP dataset, which includes seven materials-related tasks (NER, RC, EAE, PC, SAR, SC, SF).
    *   **Classification**: Four benchmarks including named entity recognition (MatScholar, SOFC)[Primary sub-task to complete], paragraph classification (PC), and slot filling (SF).
    *   Generation tasks use Micro-F1 and Macro-F1, averaged over five seeds. Classification tasks report Macro-F1 (SOFC-NER, SOFC-Filling), Micro-F1 (MatScholar), and accuracy (Glass Science), with cross-validation over five folds and three seeds.

## Evaluation Metrics
- Micro-F1
- Macro-F1
- Accuracy
- Recall
- Precision
- F1 Score

## Hints
- Be mindful of time constraints and attempt to achieve results for all the following tables.
- Realize system configuration and optimize training to leverage the given resources.
- First attempt SOFC then MatScholar. If you think you have reached satisfactory performance on both, then move on to other tasks.

## Baseline Results (to beat)

Table 1: Evaluation results on MatSci-NLP (generation tasks): The tasks encompass Named Entity Recognition (NER), Relation Classification (RC), Event Argument Extraction (EAE), Paragraph Classification (PC), Synthesis Action Retrieval (SAR), Sentence Classification (SC), and Slot Filling (SF).

| Tokenization | Metric | NER | RC | EAE | PC | SAR | SC | SF | Overall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| BPE | Micro-F1 | 76.6 ±0.2 | 80.9 ±0.3 | 48.5 ±0.2 | 73.1 ±0.5 | 81.9 ±0.4 | 90.0 ±0.1 | 57.4 ±0.2 | 72.6 ±0.1 |
| | Macro-F1 | 47.1 ±0.5 | 47.2 ±0.9 | 36.3 ±0.3 | 40.2 ±0.0 | 41.8 ±1.3 | 47.6 ±0.0 | 16.7 ±1.6 | 42.0 ±0.9 |
| WordPiece | Micro-F1 | 77.0 ±0.2 | 82.3 ±0.4 | 47.3 ±0.1 | 68.3 ±0.8 | 77.1 ±0.4 | 90.9 ±0.1 | 57.1 ±0.3 | 71.4 ±0.2 |
| | Macro-F1 | 56.1 ±0.2 | 58.5 ±0.6 | 29.4 ±0.3 | 58.9 ±1.0 | 74.6 ±0.9 | 60.3 ±0.8 | 32.6 ±0.2 | 52.9 ±0.2 |
| SAGE | Micro-F1 | 55.4 ±0.1 | 92.1 ±0.1 | 47.9 ±0.4 | 67.2 ±0.0 | 75.7 ±0.2 | 90.7 ±0.0 | 43.6 ±0.1 | 67.5 ±0.1 |
| | Macro-F1 | 57.0 ±0.3 | 61.6 ±0.4 | 28.3 ±0.3 | 59.6 ±1.3 | 67.4 ±0.9 | 61.6 ±0.8 | 35.0 ±0.3 | 52.9 ±0.3 |
| PickyBPE | Micro-F1 | 80.0 ±0.0 | 83.8 ±0.1 | 53.1 ±0.2 | 73.7 ±0.2 | 85.5 ±0.3 | 91.2 ±0.1 | 61.9 ±0.3 | 75.6 ±0.1 |
| | Macro-F1 | 41.7 ±0.1 | 65.1 ±0.2 | 36.5 ±0.6 | 40.2 ±0.0 | 66.1 ±0.7 | 47.6 ±0.0 | 23.1 ±0.1 | 45.8 ±0.1 |
| Your Method | Micro-F1 | -- | -- | -- | -- | -- | -- | -- | -- |
| | Macro-F1 | -- | -- | -- | -- | -- | -- | -- | -- |


Table 2: Evaluation results are presented across five classification tasks. Here, PC* represents accuracy, while the remaining metrics are reported as Micro-F1 and Macro-F1 scores.

| Tokenization | Metric | NER SOFC (val) | NER SOFC (test) | NER Matscholar (val) | NER Matscholar (test) | SF (val) | SF (test) | RC (val) | RC (test) | PC* (val) | PC* (test) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| BPE | Micro-F1 | 81.6 ±0.2 | 81.4 ±0.1 | 86.4 ±0.3 | 84.3 ±0.5 | 68.1 ±0.5 | 68.3 ±0.6 | 90.2 ±0.4 | 89.9 ±0.0 | 95.5 ±0.0 | 95.6 ±0.0 |
| | Macro-F1 | 80.7 ±0.2 | 78.9 ±0.1 | 85.0 ±0.6 | 82.9 ±0.7 | 65.5 ±0.4 | 59.3 ±0.8 | 86.4 ±0.1 | 85.5 ±0.1 | 95.5 ±0.0 | 95.6 ±0.0 |
| WordPiece | Micro-F1 | 82.0 ±0.6 | 80.9 ±0.4 | 88.8 ±0.2 | 86.1 ±0.3 | 67.4 ±0.5 | 60.4 ±0.7 | 90.6 ±0.2 | 91.0 ±0.7 | 95.2 ±0.1 | 95.2 ±0.1 |
| | Macro-F1 | 83.0 ±0.2 | 83.0 ±0.4 | 87.6 ±0.3 | 85.8 ±0.2 | 69.2 ±0.4 | 69.6 ±0.4 | 86.3 ±0.3 | 87.5 ±0.1 | 95.2 ±0.1 | 95.2 ±0.1 |
| SAGE | Micro-F1 | 82.0 ±0.2 | 79.7 ±0.4 | 88.4 ±0.3 | 86.7 ±0.4 | 67.9 ±0.5 | 60.3 ±0.4 | 89.8 ±0.4 | 90.6 ±0.3 | 95.3 ±0.0 | 95.6 ±0.2 |
| | Macro-F1 | 82.7 ±0.2 | 82.5 ±0.8 | 87.6 ±0.2 | 86.1 ±0.1 | 69.7 ±0.3 | 69.5 ±0.6 | 86.4 ±0.7 | 87.1 ±0 | 95.3 ±0.0 | 95.6 ±0.2 |
| PickyBPE | Micro-F1 | 77.3 ±0.3 | 78.8 ±0.6 | 84.1 ±0.4 | 83.4 ±0.6 | 62.0 ±0.3 | 60.2 ±0.4 | 88.6 ±0.1 | 85.8 ±0.2 | 95.7 ±0.3 | 95.8 ±0.2 |
| | Macro-F1 | 78.6 ±0.4 | 81.0 ±0.7 | 86.1 ±0.3 | 84.7 ±0.5 | 67.1 ±0.1 | 55.4 ±0.2 | 88.8 ±0.6 | 87.0 ±0.2 | 95.7 ±0.3 | 95.8 ±0.2 |
| Your Method | Micro-F1 | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| | Macro-F1 | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |