MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export UV_LINK_MODE=copy
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

INTERVAL=1353
SEED=42

## LLaMA3.1-8B
lr=0.0002
### Train 
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/haldet_grpo.py \
    --config recipes/Qwen2.5-7B-Instruct/grpo/config_demo.yaml --seed $SEED  --vllm_mode colocate --vllm_gpu_memory_utilization 0.5 --metric_type bleurt \
    --dataset_name NQOpen_llama3.1-8B --learning_rate $lr --num_train_epochs 1 --save_steps 451 --model_name_or_path $MODEL_PATH

### Test
for idx in {1..10};do
    python src/open_r1/haldet_eval_vllm.py \
        --base_model $MODEL_PATH --dataset_name NQOpen_llama3.1-8B \
        --lora_path "data_bleurt/seed_$((SEED))/NQOpen_llama3.1-8B/Qwen2.5-7B-HalDet-GRPO_lr${lr}_beta0.04/checkpoint-$((INTERVAL*idx))"  \
        --seed $((SEED))  --metric_type bleurt
done

## Qwen3-8B
lr=0.0001
### Train 
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/haldet_grpo.py \
    --config recipes/Qwen2.5-7B-Instruct/grpo/config_demo.yaml --seed $SEED  --vllm_mode colocate --vllm_gpu_memory_utilization 0.5 --metric_type bleurt \
    --dataset_name NQOpen_qwen3-8B --learning_rate $lr --num_train_epochs 1 --save_steps 451 --model_name_or_path $MODEL_PATH

### Test
for idx in {1..10};do
    python src/open_r1/haldet_eval_vllm.py \
        --base_model $MODEL_PATH --dataset_name NQOpen_qwen3-8B \
        --lora_path "data_bleurt/seed_$((SEED))/NQOpen_qwen3-8B/Qwen2.5-7B-HalDet-GRPO_lr${lr}_beta0.04/checkpoint-$((INTERVAL*idx))"  \
        --seed $((SEED))  --metric_type bleurt
done