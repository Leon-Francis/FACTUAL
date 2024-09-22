class Sem16Config():
    dataset_name = 'SemEval2016Task6'
    is_multi_label = True
    apply_cleaning = False
    test_stance = [0, 1]
    label2idx = {'FAVOR': 0, 'AGAINST': 1, 'NONE': 2}
    in_target_target_names = ['Hillary Clinton', 'Feminist Movement', 'Legalization of Abortion', 'Atheism', 'Climate Change is a Real Concern']
    zero_shot_target_names = ['Hillary Clinton', 'Feminist Movement', 'Legalization of Abortion', 'Atheism', 'Climate Change is a Real Concern', 'Donald Trump']
    short_target_names = {
        'Feminist Movement': 'FM',
        'Hillary Clinton': 'HC',
        'Legalization of Abortion': 'LA',
        'Atheism': 'A',
        'Climate Change is a Real Concern': 'CC',
        'Donald Trump': 'DT'
    }
    data_dir = 'datasets/Semeval16'
    in_target_data_dir = f'{data_dir}/in-target'
    zero_shot_data_dir = f'{data_dir}/zero-shot'

    llm_rationale_data_dir = {
        'gpt': 'datasets/gpt_analysis_datasets/Semeval16',
        'llama': 'datasets/llama_analysis_datasets/Semeval16'
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
            'gpt': 'datasets/conterfactual_datasets/gpt_analysis_datasets/gpt_causal_counterfactural_dataset/Semeval16/in-target',
            'llama': 'datasets/conterfactual_datasets/llama_analysis_datasets/gpt_causal_counterfactural_dataset/Semeval16/in-target'
        },
        'non_causal': {
            'gpt': 'datasets/conterfactual_datasets/gpt_analysis_datasets/gpt_non_causal_counterfactural_dataset/Semeval16/in-target',
            'llama': 'datasets/conterfactual_datasets/llama_analysis_datasets/gpt_non_causal_counterfactural_dataset/Semeval16/in-target'
        }
    }
    zero_shot_cad_data_dir = {
        'causal': {
            'gpt': 'datasets/conterfactual_datasets/gpt_analysis_datasets/gpt_causal_counterfactural_dataset/Semeval16/zero-shot',
            'llama': 'datasets/conterfactual_datasets/llama_analysis_datasets/gpt_causal_counterfactural_dataset/Semeval16/zero-shot'
        },
        'non_causal': {
            'gpt': 'datasets/conterfactual_datasets/gpt_analysis_datasets/gpt_non_causal_counterfactural_dataset/Semeval16/zero-shot',
            'llama': 'datasets/conterfactual_datasets/llama_analysis_datasets/gpt_non_causal_counterfactural_dataset/Semeval16/zero-shot'
        }
    }
