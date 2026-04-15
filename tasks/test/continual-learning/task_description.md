## Research Goal
Continual Learning (CL) with foundation models has emerged as a promising paradigm, but existing prompt-based and Low-Rank Adaptation-based (LoRA-based) methods have limitations. These methods often require expanding a prompt or LoRA pool, or retaining samples of previous tasks for rehearsal. This poses significant scalability challenges as the number of sequential tasks grows. Current CL methods with foundation models often fail to satisfy three desirable properties simultaneously: being rehearsal-free, maintaining inference efficiency, and allowing for end-to-end optimization of all parameters. The reliance on growing prompt/LoRA pools compromises inference scalability, while storing samples from previous tasks is not scalable in resource-constrained or large-scale settings. A new method is needed to address these limitations and achieve a more scalable and practical solution for continual learning.

## Experimental Settings
*   **Evaluation Benchmarks**: ImageNet-R, ImageNet-A, CIFAR-100, CUB-200.
*   **Task Splits**:
    *   ImageNet-R: 5 tasks (40 classes/task), 10 tasks (20 classes/task), or 20 tasks (10 classes/task).
    *   ImageNet-A: 10 tasks (20 classes/task).
    *   CIFAR-100: 10 tasks (10 classes/task). [Primary Sub-task]
    *   CUB-200: 10 tasks (20 species/task).
*   **Foundation Models**: ViT-B/16 (pre-trained on ImageNet-21K and fine-tuned on ImageNet-1K).
*   **Metrics**: Average accuracy (Acc), Average anytime accuracy (AAA).
*   **Evaluation Setup**: Results should be reported as mean across three runs with standard errors (use seeds 1992 - 1994).
   
## Datasets
- ImageNet-R: https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing
- ImageNet-A: https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p/view?usp=sharing
- CIFAR-100: torchvision/tensorflow datasets
- CUB-200: https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb/view?usp=sharing


## Evaluation Metrics
- Acc: The Acc metric measures the overall performance by computing the average accuracy across all N tasks upon the completion of CL.
- AAA: AAA further accumulates the average accuracy of all encountered tasks after training on each new task.

## Hints
- Leverage LoRa based or inspired techniques to only fine-tune a small subset of parameters.
- Be mindful of time constraints and attempt to achieve results for all the following tables.
- Realize system configuration and optimize training to leverage the given resources.

## Baseline Results (to beat)

Table 1: Performance comparison on ImageNet-R across different task lengths.

| Method | ImageNet-R (N = 5) Acc ↑ AAA↑ | ImageNet-R (N = 10) Acc ↑ AAA ↑ | ImageNet-R (N = 20) Acc ↑ AAA ↑ |
| :--- | :--- | :--- | :--- |
| Full Fine-Tuning | 64.92 (0.87) 75.57 (0.50) | 60.57 (1.06) 72.31 (1.09) | 49.95 (1.31) 65.32 (0.84) |
| L2P | 73.04 (0.71) 76.94 (0.41) | 71.26 (0.44) 76.13 (0.46) | 68.97 (0.51) 74.16 (0.32) |
| DualPrompt | 69.99 (0.57) 72.24 (0.41) | 68.22 (0.20) 73.81 (0.39) | 65.23 (0.45) 71.30 (0.16) |
| CODA-Prompt | 76.63 (0.27) 80.30 (0.28) | 74.05 (0.41) 78.14 (0.39) | 69.38 (0.33) 73.95 (0.63) |
| HiDe-Prompt | 74.77 (0.25) 78.15 (0.24) | 74.65 (0.14) 78.46 (0.18) | 73.59 (0.19) 77.93 (0.19) |
| InfLoRA | 76.95 (0.23) 81.81 (0.14) | 74.75 (0.64) 80.67 (0.55) | 69.89 (0.56) 76.68 (0.57) |
| Your Method | -- -- | -- -- | -- -- |

Table 2: Performance comparison on ImageNet-A.

| Method | ImageNet-A (N = 10) Acc ↑ AAA ↑ |
| :--- | :--- |
| Full Fine-Tuning | 16.31 (7.89) 30.04 (13.18) |
| L2P | 42.94 (1.27) 51.40 (1.95) |
| DualPrompt | 45.49 (0.96) 54.68 (1.24) |
| CODA-Prompt | 45.36 (0.78) 57.03 (0.94) |
| HiDe-Prompt | 42.70 (0.60) 56.32 (0.40) |
| InfLoRA | 49.20 (1.12) 60.92 (0.61) |
| Your Method | -- -- |

Table 3a: Performance comparison on CIFAR100.

| Method | CIFAR100 Acc ↑ AAA ↑ |
| :--- | :--- |
| Full Fine-Tuning | 69.49 (0.50) 80.35 (0.87) |
| L2P | 83.18 (1.20) 87.69 (1.05) |
| DualPrompt | 81.48 (0.86) 86.41 (0.66) |
| CODA-Prompt | 86.31 (0.12) 90.67 (0.22) |
| InfLoRA | 86.75 (0.35) 91.72 (0.15) |
| Your Method | -- -- |

Table 3b: Performance comparison on CUB-200.

| Method | CUB200 Acc ↑ AAA ↑ |
| :--- | :--- |
| Full Fine-Tuning | 51.43 (1.41) 69.74 (0.93) |
| L2P | 65.18 (2.49) 76.12 (1.27) |
| DualPrompt | 68.00 (1.06) 79.40 (0.88) |
| CODA-Prompt | 71.92 (0.33) 78.76 (0.65) |
| InfLoRA | 70.82 (0.23) 81.39 (0.14) |
| Your Method | -- -- |