import os
import json
import logging
import argparse
import numpy as np
from typing import List, Dict, Any

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from open_r1.utils.prompt import Haldet_multi_answer
from open_r1.utils.haldet_utils import load_hal_datasets, get_prompt_data, extract_scores_from_generation, plot_histogram

logger = logging.getLogger(__name__)

def build_conversation(
    example: Dict[str, Any],
    prompt_column: str,
    system_prompt: str = None
) -> List[Dict[str, str]]:
    """Build a conversation in messages format (for use with vLLM chat template)"""
    prompt_column_list = prompt_column.split('&')
    question = example[prompt_column_list[0]]
    answer = example[prompt_column_list[1]]
    prompt_text = get_prompt_data(Haldet_multi_answer, question, [answer])

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})
    return messages


def evaluate_with_vllm(
    base_model_path: str,
    lora_path: str,
    dataset,
    threshold: float,
    system_prompt: str = None,
    prompt_column: str = "question&answer&ground_truth",
    num_generations: int = 1,
    per_device_eval_batch_size: int = 16,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
):
    # 1. Build prompts
    messages_list = [
        build_conversation(ex, prompt_column, system_prompt)
        for ex in dataset
    ]
    ground_truths = [ex.get("ground_truth", None) for ex in dataset]

    # 2. Initialize vLLM
    llm = LLM(
        model=base_model_path,
        tokenizer=base_model_path,
        enable_lora=True,
        max_lora_rank=32,  # Adjust according to your LoRA configuration (usually 8, 16, 64)
        max_num_seqs=per_device_eval_batch_size,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        trust_remote_code=True,
        max_model_len=32768,  # Adjust according to the model
        gpu_memory_utilization=0.24
    )

    # 3. Sampling configuration
    sampling_params = SamplingParams(
        n=num_generations,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_new_tokens,
        stop_token_ids=[llm.get_tokenizer().eos_token_id],
    )

    # 4. Prepare LoRA requests (with each prompt bound to the same LoRA)
    lora_request = [LoRARequest("eval_lora", 1, lora_path)] * len(messages_list)

    # 5. Batch generate
    tokenizer = llm.get_tokenizer()
    prompts = tokenizer.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True,)
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=lora_request,
        use_tqdm=True
    )

    # 6. Collect results
    results = []
    # tokenizer = llm.get_tokenizer()
    for i, output in enumerate(outputs):
        # Rebuilding prompt text (for logging and saving)
        prompt_text = tokenizer.apply_chat_template(
            messages_list[i], tokenize=False, add_generation_prompt=True
        )
        responses = [o.text for o in output.outputs]

        results.append({
            "prompt": prompt_text,
            "responses": responses,
            "scores": [extract_scores_from_generation(resp) for resp in responses],
            "ground_truth": ground_truths[i],
        })

    score_tru = []
    score_hal = []
    for result in results:
        score = [s for s in result['scores'] if len(s) == len([result['ground_truth']])]
        if result['ground_truth'] > threshold:
            score_tru.append(np.mean(score))
        elif result['ground_truth'] <= threshold:
            score_hal.append(np.mean(score))

        # score_vote_tru = [s for s in score if float(s[0]) >= 5.0]
        # score_vote_hal = [s for s in score if float(s[0]) < 5.0]
        # if len(score_vote_tru) > len(score_vote_hal):
        #     score_vote = score_vote_tru
        # elif len(score_vote_tru) < len(score_vote_hal):
        #     score_vote = score_vote_hal
        # elif len(score_vote_tru) == len(score_vote_hal):
        #     score_vote = score
        
        # if result['ground_truth'] >= 0.5:
        #     score_tru.append(np.mean(score_vote))
        # elif result['ground_truth'] < 0.5:
        #     score_hal.append(np.mean(score_vote))
        
    print(f'true num: {len(score_tru)}, hallunicated num: {len(score_hal)}')
    return score_tru, score_hal

    # min_length = min(len(score_tru), len(score_hal))
    # print(f'true num: {len(score_tru[:min_length])}, hallunicated num: {len(score_hal[:min_length])}')
    # return score_tru[:min_length], score_hal[:min_length]


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA-finetuned model with vLLM")
    parser.add_argument("--base_model", type=str, required=True, help="Base model path (e.g., meta-llama/Llama-3-8b)")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name or path")
    parser.add_argument("--dataset_prompt_column", type=str, default="question&answer&ground_truth", help="Prompt column format")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--system_prompt", type=str, default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", help="Optional system prompt")
    parser.add_argument("--num_generations", type=int, default=5, help="Number of generations per prompt")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=20, help="Batch size for vLLM")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition Penalty for sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset loading")
    parser.add_argument("--metric_type", type=str, default="bleurt", choices=['bleurt', 'rouge', 'deepseek'])

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}, split: {args.dataset_split}")
    # dataset_dict = load_hal_datasets(args.dataset_name, args.seed)
    dataset_dict = load_hal_datasets(args.dataset_name, args.metric_type, args.seed)
    dataset = dataset_dict[args.dataset_split]

    threshold = 0.1 if 'Wikipedia' in args.dataset_name and args.metric_type == 'deepseek' else 0.5
    print(f'threshold: {threshold}')

    # Run evaluation
    score_tru, score_hal = evaluate_with_vllm(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        dataset=dataset,
        threshold=threshold,
        system_prompt=args.system_prompt,
        prompt_column=args.dataset_prompt_column,
        num_generations=args.num_generations,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    plot_histogram(score_tru, score_hal, args)


if __name__ == "__main__":
    main()