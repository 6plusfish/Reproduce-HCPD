import argparse
from open_r1.utils.haldet_utils import load_hal_datasets

def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA-finetuned model with vLLM")
    parser.add_argument("--dataset_name", type=str, default="CoQA_qwen3-8B", help="Dataset name or path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset loading")
    
    args = parser.parse_args()
    
    dataset_dict = load_hal_datasets(args.dataset_name, args.seed)
    print(dataset_dict)
    # print(len(dataset_dict['train']), len(dataset_dict['test']))
    # for item in dataset_dict['train']:
    #     print(item['ground_truth'])

if __name__ == '__main__':
    main()