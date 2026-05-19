import os
import re
import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from datasets import Dataset, DatasetDict
from datasets import load_dataset, load_from_disk
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_INFO = {
    "qwen2.5-3B": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
    },
    "qwen2.5-7B": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
    },
    "qwen2.5-14B": {
        "model": "Qwen/Qwen2.5-14B-Instruct",
    },
    "qwen3-8B": {
        "model": "Qwen/Qwen3-8B",
    },
    "llama2_chat_7B": {
        "model": "meta-llama/Llama-2-7b-chat-hf",
    },
    "llama2_chat_13B": {
        "model": "meta-llama/Llama-2-13b-chat-hf",
    },
    "llama2_chat_70B": {
        "model": "meta-llama/Llama-2-70b-chat-hf",
    },
    "llama3.1-8B": {
        "model": "meta-llama/Llama-3.1-8B",
    },
    "llama3.1-70B": {
        "model": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    },
    # "llama3.1-70B": {
    #     "model": "meta-llama/Llama-3.1-70B",
    # },
}

def list2dict(data_list):
    return Dataset.from_list(data_list)

def load_ori_datasets(dataset_name, model_path):
    # Load datasets based on the dataset name
    if dataset_name == 'TruthfulQA':
        dataset = load_dataset("truthful_qa", 'generation')['validation']
        return dataset

    elif dataset_name == 'TriviaQA':
        dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        id_mem = set()

        def remove_dups(batch):
            if batch['question_id'][0] in id_mem:
                return {_: [] for _ in batch.keys()}
            id_mem.add(batch['question_id'][0])
            return batch

        dataset = dataset.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
        return dataset

    elif dataset_name == 'TydiQA':
        dataset = load_dataset("tydiqa", "secondary_task", split="train")
        used_indices = []
        for i in range(len(dataset)):
            if 'english' in dataset[i]['id']:
                used_indices.append(i)
        
        return dataset, used_indices
    
    elif dataset_name == 'SciQ':
        dataset = load_dataset("allenai/sciq", split="validation")
        return dataset
    
    elif dataset_name == "SciQtrain":
        dataset_train = load_dataset("allenai/sciq", split="train")
        train_idx = np.load('./generated_datasets/sciq_train_idx.npy')
        dataset = []
        for i in train_idx:
            i = int(i)
            dataset.append(dataset_train[i])
        return dataset
        
    elif dataset_name == 'NQOpen':
        dataset = load_dataset("google-research-datasets/nq_open", split="validation") 
        return dataset
    
    elif dataset_name == 'CoQA':
        import json
        import pandas as pd
        from datasets import Dataset

        def _save_dataset():
            # https://github.com/lorenzkuhn/semantic_uncertainty/blob/main/code/parse_coqa.py
            save_path = f'./coqa_dataset'
            if not os.path.exists(save_path):
                # https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json
                with open(f'./generated_datasets/coqa-dev-v1.0.json', 'r') as infile:
                    data = json.load(infile)['data']

                dataset = {}

                dataset['story'] = []
                dataset['question'] = []
                dataset['answer'] = []
                dataset['additional_answers'] = []
                dataset['id'] = []

                for sample_id, sample in enumerate(data):
                    story = sample['story']
                    questions = sample['questions']
                    answers = sample['answers']
                    additional_answers = sample['additional_answers']
                    for question_index, question in enumerate(questions):
                        dataset['story'].append(story)
                        dataset['question'].append(question['input_text'])
                        dataset['answer'].append({
                            'text': answers[question_index]['input_text'],
                            'answer_start': answers[question_index]['span_start']
                        })
                        dataset['id'].append(sample['id'] + '_' + str(question_index))
                        additional_answers_list = []

                        for i in range(3):
                            additional_answers_list.append(additional_answers[str(i)][question_index]['input_text'])

                        dataset['additional_answers'].append(additional_answers_list)
                        story = story + ' Q: ' + question['input_text'] + ' A: ' + answers[question_index]['input_text']
                        if not story[-1] == '.':
                            story = story + '.'

                dataset_df = pd.DataFrame.from_dict(dataset)

                dataset = Dataset.from_pandas(dataset_df)

                dataset.save_to_disk(save_path)
            return save_path

        # dataset = datasets.load_from_disk(_save_dataset())
        def get_dataset(tokenizer, split='validation'):
            # from https://github.com/lorenzkuhn/semantic_uncertainty/blob/main/code/parse_coqa.py
            dataset = load_from_disk(_save_dataset())
            id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))

            def encode_coqa(example):
                example['answer'] = [example['answer']['text']] + example['additional_answers']
                example['prompt'] = prompt = example['story'] + ' Q: ' + example['question'] + ' A:'
                return tokenizer(prompt, truncation=False, padding=False)

            dataset = dataset.map(encode_coqa, batched=False, load_from_cache_file=False)
            dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
            return dataset

        dataset = get_dataset(AutoTokenizer.from_pretrained(model_path, trust_remote_code=True))
        return dataset
    
    elif dataset_name == 'Wikipedia':
        dataset_wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
        wikipedia_idx = np.load('./generated_datasets/wikipedia_idx.npy')
        dataset = []
        for i in wikipedia_idx:
            i = int(i)
            dataset.append(dataset_wiki[i])
        return dataset
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_hal_datasets(dataset_name, metric_type, seed):
    dataset = dataset_name.split('_')[0]
    with open(f'./generated_datasets/{dataset}/{metric_type}/train/{dataset_name}_data_pair.pkl', 'rb') as f:
        train_data = pkl.load(f)
    with open(f'./generated_datasets/{dataset}/{metric_type}/test/{dataset_name}_most_likely_data_pair.pkl', 'rb') as f:
        test_data = pkl.load(f)
    
    train_q_list = [item['question'] for item in train_data]
    test_q_list = [item['question'] for item in test_data]
    assert train_q_list == test_q_list, "Questions in train and test sets must be aligned"

    # indices = list(range(len(train_q_list)))
    if dataset in ['TriviaQA']:
        indices = list(range(int(len(train_q_list)/3)))
    elif dataset in ['CoQA']:
        indices = list(range(len(train_q_list)))[:3000]
    else:
        indices = list(range(len(train_q_list)))
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=0.25, 
        random_state=seed
    )

    train_subdata = [train_data[idx] for idx in train_idx]
    # train_subdata = [dict(question=data['question'], answer=answer, ground_truth=gt) for data in train_subdata for answer, gt in zip(data['answer'][:5], data['ground_truth'][:5])]
    train_subdata = [dict(question=data['question'], answer=answer, ground_truth=gt) for data in train_subdata for answer, gt in zip(data['answer'], data['ground_truth'])]
    test_subdata = [test_data[idx] for idx in test_idx]
    for item in train_subdata:
        print(item)

    train_dataset, test_dataset = Dataset.from_list(train_subdata), Dataset.from_list(test_subdata)
    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset,})

    return dataset_dict

def reconstruct_train_data_pair(question, ref_answer, answer, gts):
    data_pair = []
    for ques, ref_ans, ans, gt in zip(question, ref_answer, answer, gts):
        data_pair.append(
            dict(question=ques, answer=[ref_ans[0]]+ans, ground_truth=[10]+gt)
        )
    return data_pair

def reconstruct_test_data_pair(question, answer, gts):
    data_pair = []
    for ques, ans, gt in zip(question, answer, gts):
        data_pair.append(
            dict(question=ques, answer=ans, ground_truth=gt)
        )
    return data_pair

def assemble_responses(resp_list):
    return "".join([f"[The Begin of Response {i+1}]\n{resp}\n[The End of Response {i+1}]\n\n" for i, resp in enumerate(resp_list)])

def get_prompt_data(prompt_template, question, answers):
    score_placeholder = ", ".join(["x"] * len(answers))
    prompt = prompt_template.format(question, assemble_responses(answers), score_placeholder)
    
    return prompt

def extract_scores_from_generation(text: str) -> list[float]:
    pattern = re.compile(
        r'(?:\\{1,2}boxed\{|\[)'
        r'\s*([^\]\}]+?)\s*'
        r'(?:\}|\])'
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return []
    last_content = matches[-1].group(1)
    parts = re.split(r'\s*,\s*', last_content.strip())
    floats = []
    for p in parts:
        try:
            floats.append(float(p))
        except ValueError:
            pass
    return floats

def get_score_align_reward(completions, ground_truth, prompts=None, completion_ids=None, question=None, answer=None):
    scores = [extract_scores_from_generation(pred[0]['content']) for pred in completions]
    assert len(scores) == len(ground_truth), "Generations number dismatch."
    
    reward_list = []
    for score, gt in zip(scores, ground_truth):
        if len(score) != len([gt]):
            reward_list.append(0.0)
        elif len(score) == 1:
            error = 1 - abs(score[0] - gt) / 9.0
            reward_list.append(error)
        elif len(score) >= 2:
            # from scipy.stats import kendalltau
            # tau, _ = kendalltau(score, gt)
            # reward = tau  # ∈ [-1, 1]
            from sklearn.metrics import ndcg_score
            reward = ndcg_score([score], [gt])
            reward_list.append(float(reward))
        else:
            reward_list.append(0.0)
    
    return reward_list

def get_vote_score_align_reward(completions, ground_truth, threshold=5.0, prompts=None, completion_ids=None, question=None, answer=None):
    scores = [extract_scores_from_generation(pred[0]['content']) for pred in completions]
    assert len(scores) == len(ground_truth), "Generations number dismatch."
    
    reward_list = []
    for score, gt in zip(scores, ground_truth):
        if len(score) != len([gt]):
            reward_list.append(0.0)
        else:
            gt_is_real = gt >= threshold
            pred_is_real = score[0] >= threshold
            if pred_is_real == gt_is_real:
                # Correct classification: Basic reward=0.7, plus regression accuracy bonus (up to+0.3)
                regression_accuracy = 1 - abs(score[0] - gt) / 9.0      # ∈ [0,1]
                reward_list.append(0.7 + 0.3 * regression_accuracy)  # ∈ [0.7, 1.0]
            else:
                # Classification error: Basic reward=0.3, minus error severity penalty (maximum -0.3)
                error_severity = abs(score[0] - gt)              # The farther you go wrong, the greater the punishment
                penalty = min(error_severity / 5.0, 1.0)      # Normalize to [0,1]
                reward_list.append(0.3 - 0.3 * penalty)       # ∈ [0.0, 0.3]
    
    return reward_list

def eval_single_sample(prompt, dataset_name, api_key, base_url):
    from openai import OpenAI
    
    question = prompt[0]
    gold_standard_answers = prompt[1]
    generated_answer = prompt[2]
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        if dataset_name == 'Wikipedia':
            res = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "user", "content": f"""Title: {question}
                                                        Real Wikipedia passage: {gold_standard_answers}
                                                        Generated Wikipedia passage: {generated_answer}"""},
                        {"role": "system", "content": """You are tasked with evaluating whether a generated Wikipedia passage is consistent with the provided real Wikipedia passage. Use a 10-point scoring scale (1-2: Completely inconsistent/hallucinated, 3-4: Mostly inconsistent with minor, 5-6: Partially consistent, mixed quality, 7-8: Mostly consistent with minor errors, 9-10: Completely consistent, accurate, comprehensive) to score the generated content. Provide a clear justification for your scoring.
                                                        Response Format:
                                                        - Score: [1/2/3/4/5/6/7/8/9/10]
                                                        - Justification: [Briefly explain why the generated content is consistent or inconsistent with the real Wikipedia passage.]"""}
                    ],
                    max_tokens=50
                )
        else:
            res = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "user", "content": f"""Question: {question}
                                                        Gold Standard Answers: {gold_standard_answers}
                                                        Generated Answer: {generated_answer}"""},
                        {"role": "system", "content": """You are tasked with evaluating whether a generated answer is truthful compared to a provided list of gold standard answers. Use a 10-point scoring scale (1-2: Completely inconsistent/hallucinated, 3-4: Mostly inconsistent with minor, 5-6: Partially consistent, mixed quality, 7-8: Mostly consistent with minor errors, 9-10: Completely consistent, accurate, comprehensive) to score the generated content. Provide a clear justification for your scoring.
                                                        Response Format:
                                                        - Score: [1/2/3/4/5/6/7/8/9/10]
                                                        - Justification: [Explain briefly why the answer is correct or incorrect.]"""}
                    ],
                    max_tokens=50
                )
        return res.choices[0].message.content
    except Exception as e:
        return f"Error evaluating sample: {str(e)}"

def deepseek_eval(prompts, dataset_name, api_key, base_url, max_workers=10, custom_system_prompt=None):
    from tqdm import tqdm
    import concurrent.futures
    results = [None] * len(prompts) 
    base_url = base_url.strip()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(eval_single_sample, prompt, dataset_name, api_key, base_url): idx
            for idx, prompt in enumerate(tqdm(prompts))
        }
        
        for future in tqdm(concurrent.futures.as_completed(futures)):
            idx = futures[future]
            try:
                results[idx] = future.result() 
            except Exception as e:
                results[idx] = f"Error in future processing: {str(e)}"
    return results

def plot_histogram(results_true, results_false, args):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.labelweight'] = 'bold'    # label bold
    plt.rcParams['axes.titleweight'] = 'bold'    # title bold
    plt.rcParams['font.weight'] = 'bold'         # global bold

    plt.figure(figsize=(7, 5))
    ax = plt.axes(polar=False)
    plt.grid(linestyle='--', linewidth=1)
    ax.spines['bottom'].set_color('#BEBEBE')
    ax.spines['top'].set_color('#BEBEBE')
    ax.spines['left'].set_color('#BEBEBE')
    ax.spines['right'].set_color('#BEBEBE')

    correct = np.array(results_true)
    # correct = np.array(random.sample(results_best+results_true, 100))
    label_correct = 'Correct'

    incorrect = np.array(results_false)
    label_incorrect = 'Incorrect'

    fea_correct = correct[~np.isnan(correct)]
    fea_incorrect = incorrect[~np.isnan(incorrect)]

    sns.distplot(fea_correct, hist = True, kde = True,
                kde_kws = {'shade': True, 'linewidth': 1},
                label = label_correct)
    sns.distplot(fea_incorrect, hist = True, kde = True,
                kde_kws = {'shade': True, 'linewidth': 1},
                label = label_incorrect)
    
    plt.legend()
    plt.tight_layout()

    log_config = args.lora_path.split('/')

    x = np.concatenate((fea_incorrect, fea_correct), 0)
    y = np.zeros(x.shape[0])
    y[fea_incorrect.shape[0]:] = 1

    ap = metrics.roc_auc_score(y, x)
    fpr, tpr, thresholds = metrics.roc_curve(y, x)
    accs = {th: tpr[np.argwhere(fpr <= th).max()] for th in [0.01, 0.05, 0.1]}
    thre = {tpr[np.argwhere(fpr <= th).max()]: thresholds[np.argwhere(fpr <= th).max()] for th in [0.01, 0.05, 0.1]}

    print("auroc: {:.4f}; ".format(ap) + "; ".join(["TPR: {:.4f} @ thresholds={:.4f}".format(k, v) for k, v in thre.items()]))
    if not os.path.exists(f'./logs/seed_{args.seed}/{log_config[-3]}'):
        os.makedirs(f'./logs/seed_{args.seed}/{log_config[-3]}')

    with open(os.path.join(f'./logs/seed_{args.seed}/{log_config[-3]}', f'log_{log_config[-2]}.json'), 'a') as f:
        f.write(f"# Dataset: {args.dataset_name}; Model: {args.lora_path}; Num Generations: {args.num_generations}" + '\n')
        f.write("auroc: {:.4f}; ".format(ap) + "; ".join(["TPR: {:.4f} @ thresholds={:.4f}".format(k, v) for k, v in thre.items()]) + '\n')