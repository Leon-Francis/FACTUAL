class PStanceConfig():
    dataset_name = 'P-Stance'
    is_multi_label = True
    apply_cleaning = False
    test_stance = [0, 1]
    label2idx = {'FAVOR': 0, 'AGAINST': 1}
    in_target_target_names = ['Donald Trump', 'Joe Biden', 'Bernie Sanders']
    zero_shot_target_names = ['Donald Trump', 'Joe Biden', 'Bernie Sanders']
    short_target_names = {
        'Donald Trump': 'DT',
        'Joe Biden': 'JB',
        'Bernie Sanders': 'BS'
    }
    data_dir = 'datasets/P-Stance'
    in_target_data_dir = f'{data_dir}/in-target'
    zero_shot_data_dir = f'{data_dir}/zero-shot'

    llm_rationale_data_dir = {
        'gpt': 'datasets/gpt_analysis_datasets/P-Stance',
        'llama': 'datasets/llama_analysis_datasets/P-Stance'
    }
    in_target_llm_rationale_data_dir = {
        'gpt': f'{llm_rationale_data_dir["gpt"]}/in-target',
        'llama': f'{llm_rationale_data_dir["llama"]}/in-target'
    }
    zero_shot_llm_rationale_data_dir = {
        'gpt': f'{llm_rationale_data_dir["gpt"]}/zero-shot',
        'llama': f'{llm_rationale_data_dir["llama"]}/zero-shot'
    }

    in_target_cad_data_dir = {
        'causal': {
            'gpt': 'datasets/conterfactual_datasets/gpt_analysis_datasets/gpt_causal_counterfactural_dataset/P-Stance/in-target',
            'llama': 'datasets/conterfactual_datasets/llama_analysis_datasets/gpt_causal_counterfactural_dataset/P-Stance/in-target'
        },
        'non_causal': {
            'gpt': 'datasets/conterfactual_datasets/gpt_analysis_datasets/gpt_non_causal_counterfactural_dataset/P-Stance/in-target',
            'llama': 'datasets/conterfactual_datasets/llama_analysis_datasets/gpt_non_causal_counterfactural_dataset/P-Stance/in-target'
        }
    }
    zero_shot_cad_data_dir = {
        'causal': {
            'gpt': 'datasets/conterfactual_datasets/gpt_analysis_datasets/gpt_causal_counterfactural_dataset/P-Stance/zero-shot',
            'llama': 'datasets/conterfactual_datasets/llama_analysis_datasets/gpt_causal_counterfactural_dataset/P-Stance/zero-shot'
        },
        'non_causal': {
            'gpt': 'datasets/conterfactual_datasets/gpt_analysis_datasets/gpt_non_causal_counterfactural_dataset/P-Stance/zero-shot',
            'llama': 'datasets/conterfactual_datasets/llama_analysis_datasets/gpt_non_causal_counterfactural_dataset/P-Stance/zero-shot'
        }
    }