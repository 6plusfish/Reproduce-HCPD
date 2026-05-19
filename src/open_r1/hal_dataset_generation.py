import os
import torch
import random
import argparse
import numpy as np
import pickle as pkl

from tqdm import tqdm
from vllm import LLM, SamplingParams
# from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, AwqConfig
from open_r1.utils.haldet_utils import MODEL_INFO, load_ori_datasets, reconstruct_train_data_pair, reconstruct_test_data_pair


def clean_data(decoded):
    if '\nAnswer the question concisely.' in decoded:
        decoded = decoded.split('\nAnswer the question concisely.')[0]
    
    if 'Answer the question concisely' in decoded:
        decoded = decoded.split('Answer the question concisely')[0]
        
    if 'The answer to the question' in decoded:
        decoded = decoded.split('The answer to the question')[0]
    
    if 'How to Write a Concise Statement' in decoded:
        decoded = decoded.split('How to Write a Concise Statement')[0]         
        
    if 'Q:' in decoded:
        decoded = decoded.split('Q:')[0]     

    if '\nYou are an AI assistant' in decoded:    
        decoded = decoded.split('\nYou are an AI assistant')[0]  
            
    if 'You are an AI assistant' in decoded:    
        decoded = decoded.split('You are an AI assistant')[0]  
        
    if 'A:' in decoded:
        decoded = decoded.split('A:')[0]  
    
    if 'B:' in decoded:
        decoded = decoded.split('B:')[0] 
        
    if 'C:' in decoded:
        decoded = decoded.split('C:')[0] 
        
    if 'D:' in decoded:
        decoded = decoded.split('D:')[0] 
    
    if "\nOkay, let's see" in decoded:
        decoded = decoded.split("\nOkay, let's see")[0]
    
    return decoded

def generate_dataset(begin_index, end_index, args, model_or_llm, tokenizer_or_none, ori_dataset, period_token_id, used_indices=None):
    question_list = []
    answers_list = []
    prompts = []      # shared prompt strings
    hf_prompts_tensors = []  # only for HF path
    indices = []      # to map back to question

    # === Step 1: Build prompts uniformly ===
    for i in range(begin_index, end_index):
        if args.dataset == 'TydiQA':
            idx = int(used_indices[i])
            question = ori_dataset[idx]['question']
            context = ori_dataset[idx]['context']
            prompt_str = f"Concisely answer the following question based on the information in the given passage: \n Passage: {context} \n Q: {question} \n A:"
            max_new_tokens = 64
        elif args.dataset == 'CoQA':
            # prompt_str = ori_dataset[i]['prompt']
            # question = "" # or extract if CoQA has explicit question
            question = ori_dataset[i]['question']
            content = ori_dataset[i]['prompt'].split('Q:')[0]
            prompt_str = f"Based on the following context, answer the question concisely. Context: {content} Q: {question} A:"
            max_new_tokens = 64
        elif args.dataset == 'Wikipedia':
            concept = ori_dataset[i]['title']
            prompt_str = f"This is a Wikipedia passage about {concept}:"
            question = concept
            max_new_tokens = 500
        else:
            question = ori_dataset[i]['question']
            prompt_str = f"Answer the question concisely. Q: {question} A:"
            max_new_tokens = 64

        prompts.append(prompt_str)
        indices.append(i)
        if args.most_likely:
            # Tokenize for HF (must be done per sample due to generate() API)
            input_ids = tokenizer_or_none(prompt_str, return_tensors='pt').input_ids.to(args.device)
            hf_prompts_tensors.append(input_ids)

    # === Step 2: Generate based on mode ===
    if args.most_likely:
        # ---- HF Path (beam search) ----
        model = model_or_llm
        tokenizer = tokenizer_or_none
        for i, prompt_tensor in enumerate(tqdm(hf_prompts_tensors, desc=f'Generating (HF beam) on {args.dataset}')):
            generated = model.generate(
                prompt_tensor,
                num_beams=5,
                num_return_sequences=1,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=period_token_id  # list supported in HF
            )
            decoded = tokenizer.decode(generated[0, prompt_tensor.shape[-1]:], skip_special_tokens=True)
            decoded = clean_data(decoded)
            if args.dataset == 'TydiQA':
                question_list.append(ori_dataset[int(used_indices[indices[i]])]['question'])
            elif args.dataset == 'CoQA':
                # question_list.append(ori_dataset[indices[i]]['prompt'])
                content = ori_dataset[indices[i]]['prompt'].split('Q:')[0]
                ques = ori_dataset[indices[i]]['question']
                question_list.append(f"Context: {content} Q: {ques}")
            elif args.dataset == 'Wikipedia':
                concept = ori_dataset[indices[i]]['title']
                question_list.append(f"This is a Wikipedia passage about {concept}:")
            else:
                question_list.append(ori_dataset[indices[i]]['question'])
            # question_list.append(ori_dataset[indices[i]]['question'] if args.dataset != 'TydiQA' else ori_dataset[int(used_indices[indices[i]])]['question'])
            answers_list.append([decoded])
    else:
        # ---- vLLM Path (sampling) ----
        llm = model_or_llm
        sampling_params = SamplingParams(
            n=9,
            temperature=0.5,
            top_p=1.0,
            max_tokens=max_new_tokens,
            stop=[
                "\n",
                "Q:",
                "A:",
                "B:",
                "C:",
                "D:",
                "You are an AI assistant",
                "Answer the question concisely",
                "The answer to the question",
                "How to Write a Concise Statement",
                "Okay, let's see"
            ]
        )
        outputs = llm.generate(prompts, sampling_params)
        for i, output in enumerate(tqdm(outputs, desc=f'Generating (vLLM sampling) on {args.dataset}')):
            idx = indices[i]
            if args.dataset == 'TydiQA':
                question = ori_dataset[int(used_indices[idx])]['question']
            elif args.dataset == 'CoQA':
                # question = ori_dataset[idx]['prompt']
                content = ori_dataset[idx]['prompt'].split('Q:')[0]
                ques = ori_dataset[idx]['question']
                question = f"Context: {content} Q: {ques}"
            elif args.dataset == 'Wikipedia':
                concept = ori_dataset[idx]['title']
                question = f"This is a Wikipedia passage about {concept}:"
            else:
                question = ori_dataset[idx]['question']
            
            question_list.append(question)
            answers = [clean_data(o.text) for o in output.outputs]
            answers_list.append(answers)

    return question_list, answers_list

def get_gt_label(questions, answers, ref_answers, dataset_name, metric_type):
    gts = np.zeros(0)
    assert len(answers) == len(ref_answers), "Answers and reference answers must have the same length."
    
    if metric_type == 'bleurt':
        from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
        label_tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')
        label_model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20').to(args.device).eval()

        for i in tqdm(range(len(answers)), desc=f'Labeling using {metric_type}'):
            all_results = np.zeros((len(ref_answers[i]), len(answers[i])))
            with torch.no_grad():
                for anw in range(len(ref_answers[i])):
                    inputs = label_tokenizer(answers[i], [ref_answers[i][anw]] * len(answers[i]),
                                        padding='longest', return_tensors='pt', truncation=True, max_length=512)
                    for key in list(inputs.keys()):
                        inputs[key] = inputs[key].cuda()
                    res = np.asarray(label_model(**inputs).logits.flatten().tolist())
                    all_results[anw] = res
            gts = np.concatenate([gts, np.max(all_results, axis=0)], 0)
            # gt_label = np.asarray(gts > 0.5, dtype=np.int32)
    elif metric_type == 'rouge':
        import evaluate
        rouge = evaluate.load('rouge')

        for i in tqdm(range(len(answers)), desc=f'Labeling using {metric_type}'):
            all_results = np.zeros((len(ref_answers[i]), len(answers[i])))
            all_results1 = np.zeros((len(ref_answers[i]), len(answers[i])))
            all_results2 = np.zeros((len(ref_answers[i]), len(answers[i])))
            for anw in range(len(ref_answers[i])):
                results = rouge.compute(predictions=answers[i],
                                        references=[ref_answers[i][anw]] * len(answers[i]),
                                        use_aggregator=False)
                all_results[anw] = results['rougeL']
                all_results1[anw] = results['rouge1']
                all_results2[anw] = results['rouge2']

            # breakpoint()
            gts = np.concatenate([gts, np.max(all_results, axis=0)], 0)
            # gt_label = np.asarray([g['rouge1'].fmeasure > 0.5 for g in gts], dtype=np.int32)
    elif metric_type == 'deepseek':
        import re
        from open_r1.utils.haldet_utils import deepseek_eval
        
        def parse_score(result_str, normalize=True):
            if not result_str or "Error" in result_str:
                return 0.0
            match = re.search(r'Score:\s*([1-9]|10)(?:\.\d+)?', result_str, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                score = max(1.0, min(10.0, score))
                return score / 10.0 if normalize else score
            match = re.search(r'\b([1-9]|10)\b', result_str)
            if match:
                score = float(match.group(1))
                return score / 10.0 if normalize else max(1.0, min(10.0, score))
            return 0.0
        
        tasks = []
        for i in range(len(answers)):
            question = questions[i] if questions and i < len(questions) else ""
            for ref_idx in range(len(ref_answers[i])):
                for ans_idx in range(len(answers[i])):
                    tasks.append({
                        'sample_idx': i,
                        'ref_idx': ref_idx,
                        'ans_idx': ans_idx,
                        'prompt': [question, ref_answers[i][ref_idx], answers[i][ans_idx]]
                    })
        
        prompts = [t['prompt'] for t in tasks]
        raw_results = deepseek_eval(
            prompts=prompts,
            dataset_name=dataset_name,
            api_key='sk-xxx',
            base_url='https://api.deepseek.com',
            max_workers=1000
        )
        
        result_map = {}
        for idx, task in enumerate(tasks):
            key = (task['sample_idx'], task['ref_idx'], task['ans_idx'])
            result_map[key] = parse_score(raw_results[idx], normalize=True)
        
        for i in tqdm(range(len(answers)), desc=f'Labeling using {metric_type}'):
            all_results = np.zeros((len(ref_answers[i]), len(answers[i])))
            for ref_anw in range(len(ref_answers[i])):
                for anw in range(len(answers[i])):
                    key = (i, ref_anw, anw)
                    all_results[ref_anw, anw] = result_map.get(key, 0.0)
            
            gts = np.concatenate([gts, np.max(all_results, axis=0)], 0)
    else:
        raise ValueError(f"Unknown metric: {metric_type}")
    
    return gts    

def main(args):
    print(f"Loading generative LLM {args.gen_model_name}...")
    model_path = MODEL_INFO[args.gen_model_name]['model']
    
    # gen_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # gen_model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto").to(args.device)
    if args.most_likely:
        gen_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if '70B' in args.gen_model_name:
            gen_model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype="auto", device_map="auto").to(args.device)
        else:
            gen_model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map="auto").to(args.device)
        llm = None
        period_token_id = [gen_tokenizer(_)['input_ids'][-1] for _ in ['\n']]
        period_token_id += [gen_tokenizer.eos_token_id]
    else:
        gen_model = None
        gen_tokenizer = None
        if '70B' in args.gen_model_name:
            llm = LLM(model=model_path, trust_remote_code=True, quantization="awq", dtype="auto", max_model_len=16384, tensor_parallel_size=torch.cuda.device_count())
        else:
            llm = LLM(model=model_path, trust_remote_code=True, dtype="bfloat16", tensor_parallel_size=torch.cuda.device_count())
        period_token_id = None
        
    print(f"Loading original dataset {args.dataset}...")
    begin_index = 0
    if args.dataset == 'TydiQA':
        ori_dataset, used_indices = load_ori_datasets(args.dataset, model_path)
        end_index = len(used_indices)
    else:
        ori_dataset = load_ori_datasets(args.dataset, model_path)
        used_indices = None
        end_index = len(ori_dataset)

    print(f"Generating dataset {args.dataset}...")
    # question_list, answers_list = generate_dataset(begin_index, end_index, args, gen_model, gen_tokenizer, ori_dataset, period_token_id, used_indices) if args.dataset == 'TydiQA' else generate_dataset(begin_index, end_index, args, gen_model, gen_tokenizer, ori_dataset, period_token_id)
    question_list, answers_list = generate_dataset(
        begin_index, end_index, args,
        model_or_llm=gen_model if args.most_likely else llm,
        tokenizer_or_none=gen_tokenizer if args.most_likely else None,
        ori_dataset=ori_dataset,
        period_token_id=period_token_id,
        used_indices=used_indices
    )

    del gen_model, gen_tokenizer
    torch.cuda.empty_cache()

    ref_answers_list = []
    for i in range(end_index):
        if args.dataset == 'TruthfulQA':
            best_answer = ori_dataset[i]['best_answer']
            correct_answer = ori_dataset[i]['correct_answers']
            ref_answers = [best_answer] + correct_answer
        elif args.dataset == 'TriviaQA':
            ref_answers = ori_dataset[i]['answer']['aliases']
        elif args.dataset == 'CoQA':
            ref_answers = ori_dataset[i]['answer']
        elif args.dataset == 'TydiQA':
            ref_answers = ori_dataset[int(used_indices[i])]['answers']['text']
        elif args.dataset == "SciQtrain":
            ref_answers = [ori_dataset[i]['correct_answer']]
        elif args.dataset == 'NQOpen':
            ref_answers = ori_dataset[i]['answer']
        elif args.dataset == 'Wikipedia':
            ref_answers = [ori_dataset[i]['text']]
        
        ref_answers_list.append(ref_answers)
        
    gts = get_gt_label(question_list, answers_list, ref_answers_list, args.dataset, args.metric_type)
    print(f'qusetions num: {len(question_list)}; answer num: {len(answers_list)}; gt_label num: {gts.shape}')

    most_likely = '_most_likely' if args.most_likely else ''

    if args.most_likely:
        if not os.path.exists(f'./generated_datasets/{args.dataset}/{args.metric_type}/test/'):
            os.makedirs(f'./generated_datasets/{args.dataset}/{args.metric_type}/test/')
        
        test_data_pair = reconstruct_test_data_pair(question_list, answers_list, gts)
        with open(f'./generated_datasets/{args.dataset}/{args.metric_type}/test/{args.dataset}_{args.gen_model_name}{most_likely}_data_pair.pkl', 'wb') as f:
            pkl.dump(test_data_pair, f)
    else:
        if not os.path.exists(f'./generated_datasets/{args.dataset}/{args.metric_type}/train/'):
            os.makedirs(f'./generated_datasets/{args.dataset}/{args.metric_type}/train/')

        assert len(set(map(len, answers_list))) <= 1, "Each question must have the same number of answers."
        reward_list = np.round(gts*10).astype(int).reshape(len(answers_list), len(answers_list[0])).tolist()

        train_data_pair = reconstruct_train_data_pair(question_list, ref_answers_list, answers_list, reward_list)
        with open(f'./generated_datasets/{args.dataset}/{args.metric_type}/train/{args.dataset}_{args.gen_model_name}{most_likely}_data_pair.pkl', 'wb') as f:
            pkl.dump(train_data_pair, f)

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument("--gen_model_name", type=str, default="llama3.1-8B", choices=['llama2_chat_7B', 'llama2_chat_13B', 'llama3.1-8B', 'llama3.1-70B','qwen2.5-3B', 'qwen2.5-7B', 'qwen2.5-14B', 'qwen3-8B'])
    parser.add_argument("--dataset", type=str, default="TruthfulQA", choices=['TruthfulQA', 'TriviaQA', 'TydiQA', 'CoQA', 'SciQ', 'SciQtrain', 'NQOpen', 'Wikipedia'])
    parser.add_argument("--metric_type", type=str, default="bleurt", choices=['bleurt', 'rouge', 'deepseek'])
    parser.add_argument('--num_gen', type=int, default=5)
    parser.add_argument('--most_likely', action="store_true")
    parser.add_argument('--seed', type=int, default=5)

    args = parser.parse_args()

    # add device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True
    
    return args


if __name__ == "__main__":
    args = parse_args_and_config()
    main(args)