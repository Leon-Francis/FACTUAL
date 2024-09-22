read -p "input training dataset: [sem16, p_stance, vast]: " trainDataset
read -p "input train dataset mode: [in_target, zero_shot]: " trainData
read -p "input model framework: [baseline, adv_baseline]: " framework
read -p "input model name: [bert_base, roberta_base, bertweet_base, robert_base_sentiment, kebert]: " trainModel
read -p "input running mode: [sweep, wandb, normal]: " runMode
read -p "input training cuda idx: " cudaIdx

currTime=$(date +"%Y-%m-%d_%T")
fileName="baseline/run_baseline.py"
outputDir="logs/baseline/${trainData}"

if [ ! -d ${outputDir} ]; then
    mkdir -p ${outputDir}
fi

outputName="${outputDir}/${trainDataset}_${framework}_${trainModel}_${currTime}.log"
nohup python ${fileName} --cuda_idx ${cudaIdx} --dataset_name ${trainDataset} --model_name ${trainModel} --${trainData} --framework_name ${framework} --${runMode} > ${outputName} 2>&1 &