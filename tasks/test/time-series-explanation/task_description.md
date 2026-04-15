## Research Goal
Recent explainable AI (XAI) methods for time series primarily focus on the magnitude of feature importance, overlooking the directional impact (positive or negative) on predictions. This leads to a suboptimal identification of significant points. Furthermore, existing evaluation metrics are flawed because they inadvertently cancel out the effects of features with opposing contributions, misrepresenting the effectiveness of attribution methods. In safety-critical domains like healthcare, energy, and transportation, high transparency in predictive models is necessary for safe and reliable operations. The black-box nature of deep neural networks makes it challenging to understand their decision-making processes, undermining trust and accountability. This work aims to provide more faithful and directionally-aware explanations for time series models, which is crucial for improving interpretability in applications where it directly impacts safety and effectiveness.

## Experimental Settings
*   **Datasets**:
    *   Synthetic: Switch-Feature, State.
    *   Real-world: Personal Activity Monitoring (PAM)[Primary subtask], Boiler, Epilepsy, Wafer and Freezer.
*   **Data Splits**: For synthetic datasets, 800 training and 200 test samples were used. For real-world datasets, evaluations were performed over five random cross-validation repetitions.
*   **Metrics**:
    *   Proposed: Cumulative Prediction Difference (CPD).
    *   Existing: Area Under Precision (AUP), Area Under Recall (AUR).
*   **Evaluation Setup**:
    *   Compared against 13 baselines including FO, AFO, IG, GradSHAP, DeepLIFT, LIME, FIT, WinIT, Dynamask, Extrmask, ContraLSP, TimeX, and TimeX++.

**Data prep**: loaders read from `data/<dataset>` inside this task (see `datasets/*.py`). Ensure the
preprocessed artifacts from the original paper are placed in those folders (e.g. the `PAM/processed_data`
and `splits` arrays, `boiler/split=<fold>.pt`, `epilepsy/split_<fold>.npy`, `Wafer/Wafer_{TRAIN,TEST}.txt`,
`FreezerRegularTrain/FreezerRegularTrain_{TRAIN,TEST}.txt`). Synthetic runners also expect `data/hmm/`
plus the `simulated_data_l2x/` pickles generated via `python synthetic/switchstate/switchgenerator.py`.
**Output dirs**: runners like \real/main.py write CSVs under `logs/pam`. Create that folder (or whatever `--output-file` parent you pass) before launching, otherwise the script will exit when it tries to append results.`

You can use the curl command to download the required datasets (links provided in readme file)
curl --location --remote-header-name --remote-name "<Link>"

And unpack into data/<dataset_name>/…
tar -xzf <downloaded-archive> -C data/<dataset_name>/

**Workflow checklist**:
1. Prepare datasets under `data/` as described above (run synthetic generators once if needed).
2. Train/evaluate using the provided scripts in `scripts/` or equivalent commands.
3. Run `./grading/grade.sh` to update `task_description.md` and capture the JSON summary before finishing.

## Evaluation Metrics
- Cumulative Prediction Difference (CPD)
- Area Under Precision (AUP)
- Area Under Recall (AUR)

## Baseline Results (to beat)

Table 1: Performance comparison of various XAI methods on real-world datasets with 10% feature masking. Results are aggregated as mean ± standard error over five random cross-validation repetitions and presented across multiple datasets, including PAM, Boiler (Multivariate), Epilepsy, Wafer, and Freezer (Univariate). Evaluation metrics include cumulative prediction difference (CPD) attribution performance under two feature substitution strategies: average substitution (Avg.) and zero substitution (Zero).

| Method | PAM Avg. | PAM Zero | Boiler Avg. | Boiler Zero | Epilepsy Avg. | Epilepsy Zero | Wafer Avg. | Wafer Zero | Freezer Avg. | Freezer Zero |
|---|---|---|---|---|---|---|---|---|---|---|
| AFO | 0.140±0.009 | 0.200±0.013 | 0.262±0.020 | 0.349±0.035 | 0.028±0.003 | 0.030±0.004 | 0.018±0.003 | 0.018±0.003 | 0.143±0.054 | 0.143±0.054 |
| GradSHAP | 0.421±0.014 | 0.518±0.012 | 0.752±0.055 | 0.747±0.092 | 0.052±0.004 | 0.054±0.004 | 0.485±0.014 | 0.485±0.014 | 0.397±0.110 | 0.397±0.110 |
| Extrmask | 0.291±0.007 | 0.380±0.009 | 0.338±0.028 | 0.400±0.031 | 0.028±0.003 | 0.029±0.003 | 0.202±0.026 | 0.202±0.026 | 0.176±0.057 | 0.176±0.057 |
| ContraLSP | 0.046±0.007 | 0.059±0.011 | 0.408±0.035 | 0.496±0.043 | 0.016±0.001 | 0.016±0.001 | 0.121±0.032 | 0.121±0.032 | 0.176±0.055 | 0.176±0.055 |
| TimeX++ | 0.057±0.004 | 0.070±0.004 | 0.124±0.028 | 0.208±0.043 | 0.030±0.004 | 0.032±0.004 | 0.000±0.000 | 0.000±0.000 | 0.216±0.056 | 0.216±0.056 |
| IG | 0.448±0.013 | 0.573±0.022 | 0.759±0.053 | 0.752±0.013 | 0.052±0.004 | 0.054±0.004 | 0.500±0.017 | 0.500±0.017 | 0.405±0.111 | 0.405±0.111 |
| Your Method | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |



Table 2: Performance comparison of various XAI methods on Switch Feature dataset. Results are reported as mean ± standard error over five cross-validation repetitions, evaluated using AUP, AUR, and CPD (10% masking) for true saliency map and cumulative masking strategies.

**Switch-Feature**

| Method | CPD ↑ | AUP ↑ | AUR ↑ |
|---|---|---|---|
| FO | 0.191±0.006 | 0.902±0.009 | 0.374±0.006 |
| AFO | 0.182±0.007 | 0.836±0.012 | 0.416±0.008 |
| GradSHAP | 0.196±0.006 | 0.892±0.010 | 0.387±0.006 |
| DeepLIFT | 0.196±0.007 | 0.918±0.019 | 0.432±0.011 |
| LIME | 0.195±0.006 | 0.949±0.015 | 0.391±0.016 |
| FIT | 0.106±0.001 | 0.522±0.005 | 0.437±0.002 |
| Dynamask | 0.069±0.001 | 0.362±0.003 | 0.754±0.008 |
| Extrmask | 0.174±0.002 | 0.978±0.004 | 0.745±0.007 |
| ContraLSP | 0.158±0.002 | 0.970±0.005 | 0.851±0.005 |
| IG | 0.196±0.007 | 0.918±0.019 | 0.433±0.011 |
| Your Method | -- | -- | -- |

Table 2: Performance comparison of various XAI methods on State dataset. Results are reported as mean ± standard error over five cross-validation repetitions, evaluated using AUP, AUR, and CPD (10% masking) for true saliency map and cumulative masking strategies.

**State**

| Method | CPD ↑ | AUP ↑ | AUR ↑ |
|---|---|---|---|
| FO | 0.158±0.004 | 0.882±0.021 | 0.303±0.005 |
| AFO | 0.143±0.007 | 0.809±0.037 | 0.374±0.007 |
| GradSHAP | 0.156±0.004 | 0.857±0.019 | 0.315±0.009 |
| DeepLIFT | 0.162±0.002 | 0.926±0.008 | 0.359±0.008 |
| LIME | 0.163±0.002 | 0.944±0.008 | 0.333±0.010 |
| FIT | 0.057±0.000 | 0.483±0.001 | 0.607±0.002 |
| Dynamask | 0.052±0.001 | 0.335±0.003 | 0.506±0.002 |
| Extrmask | 0.055±0.001 | 0.557±0.024 | 0.012±0.001 |
| ContraLSP | 0.025±0.000 | 0.495±0.011 | 0.015±0.001 |
| IG | 0.162±0.002 | 0.922±0.009 | 0.357±0.008 |
| Your Method | -- | -- | -- |
