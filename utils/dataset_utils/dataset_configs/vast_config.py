class VASTConfig():
    dataset_name = 'VAST'
    is_multi_label = True
    apply_cleaning = False
    test_stance = [0, 1, 2]
    label2idx = {'pro': 0, 'con': 1, 'neutral': 2}
    zero_shot_target_names = ['zero-shot']
    short_target_names = {
        'zero-shot': 'zero-shot'
    }

    data_dir = 'datasets/VAST'
    zero_shot_data_dir = f'{data_dir}'

    llm_rationale_data_dir = {
        'gpt': 'datasets/gpt_analysis_datasets/VAST',
        'llama': 'datasets/llama_analysis_datasets/VAST'
    }
    zero_shot_llm_rationale_data_dir = {
        'gpt': f'{llm_rationale_data_dir["gpt"]}',
        'llama': f'{llm_rationale_data_dir["llama"]}'
    }

    zero_shot_cad_data_dir = {
        'causal': {
            'gpt': 'datasets/conterfactual_datasets/gpt_analysis_datasets/gpt_causal_counterfactural_dataset/VAST',
            'llama': 'datasets/conterfactual_datasets/llama_analysis_datasets/gpt_causal_counterfactural_dataset/VAST'
        },
        'non_causal': {
            'gpt': 'datasets/conterfactual_datasets/gpt_analysis_datasets/gpt_non_causal_counterfactural_dataset/VAST',
            'llama': 'datasets/conterfactual_datasets/llama_analysis_datasets/gpt_non_causal_counterfactural_dataset/VAST'
        }
    }