[README.md](https://github.com/user-attachments/files/30946637/README.md)
# Reproduction of HCPD on TriviaQA

This repository records a reproduction of **HCPD (Human-like Criteria Probing for Hallucination Detection)** using the TriviaQA benchmark and Llama 3.1 8B as the evaluated target model. The original implementation is available in the [original HCPD repository](https://github.com/TRISKEL10N/HCPD).

## 1. Project Overview

Large language models can produce fluent but factually incorrect responses. HCPD addresses this problem under the zero-source setting, where hallucination detection must rely only on the observed question–answer pair, without access to external knowledge sources or the target model's internal states.

HCPD uses a scoring agent to construct context-dependent evaluation criteria, assign importance weights, score the response under each criterion, and aggregate the scores into an overall truthfulness score. The original method further applies reward-based alignment and multi-sampling aggregation to improve detection reliability and interpretability.

## 2. Reproduction Purpose

The purpose of this reproduction was to evaluate the reproducibility and practical applicability of the HCPD framework for zero-source hallucination detection. By implementing the official evaluation pipeline on the TriviaQA dataset, this study examines whether HCPD can effectively distinguish truthful responses from hallucinated responses without relying on external knowledge sources or access to the target model’s internal states.
Rather than replicating every experiment reported in the original paper, this reproduction focuses on verifying the main methodology and critically analysing its performance under a selected experimental configuration.

## 3. Reproduction Process

The official HCPD repository was cloned and configured using Python 3.11, PyTorch 2.6.0, and CUDA 12.4. The required dependencies and datasets were installed using the setup scripts provided by the authors:
```bash
git clone https://github.com/TRISKEL10N/HCPD.git
cd HCPD
bash setup.sh
conda activate HCPD
bash generate_datasets.sh
```
After configuring the model paths, the TriviaQA experiment was executed using:
```bash
bash run_TriviaQA.sh
```
The main experimental configuration is summarised below:

| Component                   | Configuration       |
| --------------------------- | ------------------- |
| Dataset                     | TriviaQA            |
| Target model                | Llama 3.1 8B        |
| HCPD scoring agent          | Qwen2.5-7B-Instruct |
| Semantic-consistency metric | BLEURT              |
| Random seed                 | 42                  |
| Truthful samples            | 385                 |
| Hallucinated samples        | 445                 |
| Total evaluated samples     | 830                 |


The reproduction covered environment installation, dataset loading, model configuration, TriviaQA evaluation, and the calculation of AUROC and TPR at three decision thresholds. The evaluated dataset contained 385 truthful and 445 hallucinated responses, representing approximately 46.4% and 53.6% of the samples, respectively.

This reproduction focused on validating the core HCPD pipeline under one selected configuration. Experiments on SciQ, NQ Open, and CoQA, evaluations of additional target models, comparisons with all reported baselines, ablation studies, and multi-seed experiments were outside the scope of this reproduction.

## 4. Results
The datasets required for evaluation were first downloaded and prepared using:

```bash
bash generate_datasets.sh
```
The following results were obtained through checkpoint-based validation rather than training the HCPD model from scratch. The pretrained checkpoint provided by the original authors was evaluated using:
```bash
bash quick_validation.sh
```
The results are shown below:
| Metric | Result |
| --- | ---: |
| AUROC | **0.8741** |
| TPR at threshold 9.2 | 0.1091 |
| TPR at threshold 8.2 | 0.5091 |
| TPR at threshold 7.2 | 0.7065 |

The Area Under the Receiver Operating Characteristic curve (**AUROC**) evaluates how well the detector ranks truthful and hallucinated responses across all possible decision thresholds. An AUROC of **0.8741** indicates strong ranking ability in this run: the detector assigned meaningfully separable scores to the two classes, although the separation was not perfect.

## 5. Discussion

The most important result is the AUROC of 0.8741. Because AUROC summarizes performance over the full threshold range, it provides stronger evidence of discriminative ability than any single threshold result. The value suggests that the reproduced pipeline learned or retained a useful relationship between HCPD's truthfulness scores and the BLEURT-derived labels on TriviaQA.

The threshold results show that changing the threshold affects how many positive samples are detected. At the strictest reported threshold of 9.2, the TPR was only 0.1091, meaning that relatively few positive instances were accepted under this cutoff. Lowering the threshold to 8.2 increased TPR to 0.5091, and lowering it further to 7.2 increased TPR to 0.7065. This pattern is expected: lowering the threshold classifies more samples as positive, so more positive samples are correctly identified.

These results also show why selecting a deployment threshold from TPR alone would be insufficient. A lower threshold improves sensitivity, but it may also increase the false-positive rate. Since the result file does not report FPR, precision, specificity, F1, or a confusion matrix, it is not possible to determine which of the three thresholds provides the best operational balance. The class distribution was only moderately imbalanced, with 445 hallucinated and 385 truthful samples. This makes AUROC a reasonable summary metric.

Overall, the experiment successfully validates the TriviaQA evaluation path and produces reasonable and internally consistent detection statistics. However, the result should be treated as evidence that the selected configuration can run and separate the two classes—not as a complete reproduction of every result reported in the HCPD paper.

## 6. Limitations

- Only TriviaQA and one target-model configuration were evaluated.
- Only one random seed was recorded, so the results may vary between runs.
- The log does not include FPR, AUPRC, precision, recall by class, F1, or a confusion matrix.
- The three TPR values cannot identify an optimal threshold without corresponding false-positive statistics.


## 7. Conclusion

This reproduction successfully executed the HCPD TriviaQA evaluation configuration using Llama 3.1 8B and Qwen2.5-7B-Instruct. The resulting AUROC of 0.8741 demonstrates strong hallucination-ranking performance in the recorded run. The threshold analysis further confirms the expected trade-off between strictness and sensitivity. Future verification should add multiple seeds, complete ROC and precision–recall statistics, and direct comparisons with the corresponding paper configuration.

## 8. Citation

If you use the original HCPD implementation, cite the authors' work:

```bibtex
@inproceedings{yang2026zerosource,
  title={Zero-source {LLM} Hallucination Detection with Human-like Criteria Probing},
  author={Yang, Jiahao and Zhang, Shuhai and Kang, Hailong and Liu, Feng and Chen, Qi and Tan, Mingkui},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=s4Jn6bKYGI}
}
```
## 9. Reference

Yang, J., Zhang, S., Kang, H., Liu, F., Chen, Q., and Tan, M.
Zero-source LLM Hallucination Detection with Human-like Criteria Probing.
Forty-third International Conference on Machine Learning, 2026.

Original HCPD implementation:
https://github.com/TRISKEL10N/HCPD
