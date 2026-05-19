MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export UV_LINK_MODE=copy
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

## LLaMA3.1-8B
python src/open_r1/haldet_eval_vllm.py \
    --base_model $MODEL_PATH --dataset_name TriviaQA_llama3.1-8B \
    --lora_path "data_bleurt_ckpt/seed_42/TriviaQA_llama3.1-8B/Qwen2.5-7B-HalDet-GRPO_lr0.0002_beta0.05/checkpoint-7470"  \
    --seed 42 --metric_type bleurt

python src/open_r1/haldet_eval_vllm.py \
    --base_model $MODEL_PATH --dataset_name SciQtrain_llama3.1-8B \
    --lora_path "data_bleurt_ckpt/seed_42/SciQtrain_llama3.1-8B/Qwen2.5-7B-HalDet-GRPO_lr0.0002_beta0.04/checkpoint-10125"  \
    --seed 42 --metric_type bleurt

python src/open_r1/haldet_eval_vllm.py \
    --base_model $MODEL_PATH --dataset_name NQOpen_llama3.1-8B \
    --lora_path "data_bleurt_ckpt/seed_42/NQOpen_llama3.1-8B/Qwen2.5-7B-HalDet-GRPO_lr0.0002_beta0.04/checkpoint-6765"  \
    --seed 42 --metric_type bleurt

python src/open_r1/haldet_eval_vllm.py \
    --base_model $MODEL_PATH --dataset_name CoQA_llama3.1-8B \
    --lora_path "data_bleurt_ckpt/seed_42/CoQA_llama3.1-8B/Qwen2.5-7B-HalDet-GRPO_lr0.0002_beta0.04/checkpoint-11250"  \
    --seed 42 --metric_type bleurt

## Qwen3-8B
python src/open_r1/haldet_eval_vllm.py \
    --base_model $MODEL_PATH --dataset_name TriviaQA_qwen3-8B \
    --lora_path "data_bleurt_ckpt/seed_41/TriviaQA_qwen3-8B/Qwen2.5-7B-HalDet-GRPO_lr0.0001_beta0.05/checkpoint-3735"  \
    --seed 41 --metric_type bleurt

python src/open_r1/haldet_eval_vllm.py \
    --base_model $MODEL_PATH --dataset_name SciQtrain_qwen3-8B \
    --lora_path "data_bleurt_ckpt/seed_41/SciQtrain_qwen3-8B/Qwen2.5-7B-HalDet-GRPO_lr0.0001_beta0.05/checkpoint-10125"  \
    --seed 41 --metric_type bleurt

python src/open_r1/haldet_eval_vllm.py \
    --base_model $MODEL_PATH --dataset_name NQOpen_qwen3-8B \
    --lora_path "data_bleurt_ckpt/seed_41/NQOpen_qwen3-8B/Qwen2.5-7B-HalDet-GRPO_lr0.0001_beta0.04/checkpoint-10824"  \
    --seed 41 --metric_type bleurt

python src/open_r1/haldet_eval_vllm.py \
    --base_model $MODEL_PATH --dataset_name CoQA_qwen3-8B \
    --lora_path "data_bleurt_ckpt/seed_41/CoQA_qwen3-8B/Qwen2.5-7B-HalDet-GRPO_lr0.0001_beta0.04/checkpoint-11250"  \
    --seed 41 --metric_type bleurt