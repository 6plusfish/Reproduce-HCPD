## Datasets using BLEURT as metric in main experiments
for dataset in TriviaQA SciQtrain NQOpen CoQA;do
    for model in llama2_chat_7B llama2_chat_13B llama3.1-8B qwen2.5-7B qwen2.5-14B qwen3-8B;do
        python src/open_r1/hal_dataset_generation.py --gen_model_name $model --dataset $dataset --metric_type bleurt
        python src/open_r1/hal_dataset_generation.py --gen_model_name $model --dataset $dataset --metric_type bleurt --most_likely
    done
done

## Datasets using ROUGE as alternative metric
for dataset in TriviaQA SciQtrain NQOpen CoQA;do
    for model in llama3.1-8B qwen3-8B;do
        python src/open_r1/hal_dataset_generation.py --gen_model_name $model --dataset $dataset --metric_type rouge
        python src/open_r1/hal_dataset_generation.py --gen_model_name $model --dataset $dataset --metric_type rouge --most_likely
    done
done

## Datasets using DeepSeek-V3 as alternative metric
for dataset in TriviaQA;do
    for model in llama3.1-8B qwen3-8B;do
        python src/open_r1/hal_dataset_generation.py --gen_model_name $model --dataset $dataset --metric_type deepseek
        python src/open_r1/hal_dataset_generation.py --gen_model_name $model --dataset $dataset --metric_type deepseek --most_likely
    done
done