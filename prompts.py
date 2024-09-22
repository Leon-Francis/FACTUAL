# PROMPTS

prompt_normal = '''Stance detection is to determine the attitude or tendency towards a certain target through a given sentence, including favor, against and neutral.
{sentence}
Question: What is the attitude of the sentence toward "{target}"? Please select the correct answer from "favor", "against" and "neutral".
Answer this question with JSON format:
```json
{{
    "stance": "favor" | "against" | "neutral",
}}
```'''

prompt_shuffling_labels = '''Stance detection is to determine the attitude or tendency towards a certain target through a given sentence, including against, neutral and favor.
{sentence}
Question: What is the attitude of the sentence toward "{target}"? Please select the correct answer from "against", "neutral" and "favor".
Answer this question with JSON format:
```json
{{
    "stance": "against" | "neutral" | "favor",
}}
```'''

prompt_debiasing_sentiment = '''Stance detection is to determine the attitude or tendency towards a certain target through a given sentence, including favor, against and neutral. **Note that the sentiment of the sentence is not necessarily consistent with the author's attitude on the target, and avoid directly using emotion as the only basis for judging the attitude.**
{sentence}
Question: What is the attitude of the sentence toward "{target}"? Please select the correct answer from "favor", "against" and "neutral".
Answer this question with JSON format:
```json
{{
    "stance": "favor" | "against" | "neutral",
}}
```'''

prompt_debiasing_target = '''Stance detection is to determine the attitude or tendency towards a certain target through a given sentence, including favor, against and neutral. **Be careful to only judge the author's attitude on the target based on the content in the sentence, and do not include your inherent attitude towards the target.**
{sentence}
Question: What is the attitude of the sentence toward "{target}"? Please select the correct answer from "favor", "against" and "neutral".
Answer this question with JSON format:
```json
{{
    "stance": "favor" | "against" | "neutral",
}}
```'''

def generate_zero_shot_prompt(dataset, sentence, target):
    return prompt_normal.format(sentence=sentence, target=target)

def generate_shuffling_labels_prompt(dataset, sentence, target):
    return prompt_shuffling_labels.format(sentence=sentence, target=target)

def generate_debiasing_sentiment_prompt(dataset, sentence, target):
    return prompt_debiasing_sentiment.format(sentence=sentence, target=target)

def generate_debiasing_target_prompt(dataset, sentence, target):
    return prompt_debiasing_target.format(sentence=sentence, target=target)


pstance_few_shot = [
    {
        'sentence': "I love everyone in this movement. So proud to fight another day in the Twitter trenches for compassion, love, and common sense with you all. #BigUs #BernieSanders #NotMeUs #Receipts To the rest of you, what has your candidate been saying for 30 years?",
        'target': 'Bernie Sanders',
        'answer': 'The sentence expresses a positive sentiment towards Bernie Sanders and the movement he represents. The use of words like "love," "proud," and "fight" indicates a strong support and admiration for Bernie Sanders and the values he stands for. The hashtags used, such as #BigUs and #NotMeUs, further emphasize the positive attitude towards the target. Therefore, the attitude of the sentence towards \"Bernie Sanders\" is favor.',
        'stance': 'favor'
    },
    {
        'sentence': "Just did FaceTime interview with author @andrewtshaffer fun guy and great writer who penned a book that has @JoeBiden and @BarackObama as crime fighters in @wilmingtondegov - story airs next week on @NBCPhiladelphia",
        'target': 'Joe Biden',
        'answer': 'The sentence describes Joe Biden as a crime fighter in a book written by the author Andrew Shaffer. The use of positive words such as "fun guy" and "great writer" suggests a favorable attitude towards both the author and the book. Additionally, the mention of Joe Biden and Barack Obama as crime fighters implies a positive portrayal of Joe Biden\'s character. Therefore. the attitude of the sentence towards \"Joe Biden\" is favor.',
        'stance': 'favor'
    },
    {
        'sentence': "Actually Joe Biden has had one hell of a long groping spree, some might think the fact that nobody has kicked his ass yet is an accomplishment.",
        'target': 'Joe Biden',
        'answer': "The sentence uses derogatory terms like \"hell\" and makes negative assumptions about Joe Biden's actions, suggesting he has been engaging in a \"groping spree.\" Additionally, the sentence implies that the lack of consequences for Biden's supposed actions is seen as an accomplishment by some. These elements indicate a critical or negative attitude toward Joe Biden in the sentence. Therefore, the attitude of the sentence toward \"Joe Biden\" is against.",
        'stance': 'against'
    },
    {
        'sentence': "It is insane how much they hate @realDonaldTrump. They dont care about this country or the people. Theyd rather see him fail. #insane #americanpeopleneedtowakeup",
        'target': 'Donald Trump',
        'answer': 'This can be inferred from the negative words and phrases used, such as "insane," "hate," "don\'t care," and "rather see him fail." These words indicate a negative or unfavorable attitude towards Donald Trump. Additionally, the use of hashtags like "#insane" and "#americanpeopleneedtowakeup" further suggests a critical or opposing viewpoint. Therefore, the attitude of the sentence toward \"Donald Trump\" is against.',
        'stance': 'against'
    }
]

vast_few_shot = [
    {
        'sentence': "I totally agree with this premise. As a younger person I was against Nuclear power (I was in college during 3 mile island) but now it seems that nuclear should be in the mix. Fission technology is better, and will continue to get better if we actively promote its development. The prospect of fusion energy also needs to be explored. If it's good enough for the sun and the stars, it's good enough for me.",
        'target': "nuclear power",
        'answer': 'The reason for this is that the sentence expresses agreement with the premise that nuclear power should be included in the energy mix. The speaker states that they were initially against nuclear power but have now changed their opinion, believing that fission technology is improving and should be actively promoted. Additionally, the speaker mentions the need to explore fusion energy, suggesting a positive attitude towards nuclear power as a potential energy source. Therefore, the attitude of the sentence toward \"nuclear power\" is favor.',
        'stance': 'favor'
    },
    {
        'sentence': "Community colleges tend to have standards much lower than 'regular' colleges and universities. They should be free, but we should lower our expectations to the level of the tuition. We cannot make regular colleges and universities free because of their very expensive athletic programs. US colleges and universities provide 'farm' teams for the big football and basketball leagues by adding the cost of these activities to the tuition of students who do no participate in sports. I do not have an figures for this, but if US colleges and universities provided ONLY education, far more graduates would graduate debt-free.",
        'target': "college",
        'answer': 'The sentence expresses a negative attitude towards community colleges, stating that they have lower standards compared to "regular" colleges and universities. It also suggests that community colleges should be free, but with lowered expectations to match the level of tuition. Additionally, it criticizes regular colleges and universities for not being free due to the expense of their athletic programs, implying that this negatively affects the education provided. The sentence concludes by suggesting that if colleges and universities focused solely on education, more graduates would be able to graduate debt-free.\n\nOverall, the sentence portrays a negative view towards the current state of colleges and universities, indicating a critical attitude towards the target "college." Therefore, the attitude of the sentence towards \"college\" is against.',
        'stance': 'against'
    },
    {
        'sentence': "Having a viewpoint on a person's candidacy is not necessarily a problem. if a case arises out of the election, she is expected to set those opinions aside to evaluate the law and facts. Justices Scalia and Thomas famously had direct and close ties to the Bush campaign. They deemed the ties to not be problematic in Bush v. Gore. Did anyone really wonder if those Justices supported Bush or Gore (or McCain, or Romney...)? In matters other than elections, Justices certainly and routinely have opinions on matters before a case arises before the court -- for example, many of the justices have written memos or articles or even represented clients on subjects such as affirmative action, or the requisite amount of deference to give an agency.",
        'target': "3d printing",
        'answer': "The reason for choosing this answer is that the sentence does not express any positive or negative sentiment towards 3D printing. It simply provides examples and explanations related to the topic of justices having opinions on matters before a case arises before the court. The sentence does not indicate any bias or preference towards 3D printing. Therefore, the attitude of the sentence towards \"3D printing\" is neutral.",
        'stance': 'neutral'
    },
    {
        'sentence': "The liberals' contributions to this section are basically saying that two Articles and four Amendments will fix what two Articles and three Amendments haven't. Yet, the states do a good job of running elections and the required IDs are easy to get, in a process already overseen by our heavily politicized DOJ. Regarding the extravagant demand for transportation, both major parties have local people who drive you to get IDs or to vote already. We should keep that effort in the private sector where it arose. If you are frantic enough not to trust the judicial system already, one more amendment won't cure what ails you. What would help all of us is more knowledge of history - which we won't find on the Soros blogs. Force people to vote? When all they are interested in is celebrity and sports buzz or their iPhones, absolutely not. They'll end up simply voting for the first name, and when that is a GOPer, we'll have to keep watch over the bridges in the big-city liberal-statist philosophical ghettos/campuses.",
        'target': "voting",
        'answer': "The attitude of the sentence is against the idea of making changes to the voting system. The sentence expresses skepticism towards the liberals' contributions, stating that their proposed amendments will not fix the existing issues. It also criticizes the idea of transportation assistance and forcing people to vote, suggesting that people are not interested in voting and may make uninformed choices. Additionally, the sentence uses derogatory language to refer to certain political groups and implies that their ideas are not trustworthy. Overall, the tone of the sentence is negative and dismissive towards the concept of voting. Therefore, the attitude of the sentence towards \"voting\" is against.",
        'stance': 'against'
    }
]

semeval_few_shot = [
    {
        'sentence': "Feminists are so stupid they think they're the only ones who can have freedom of speech Lmaooo",
        'target': "Feminist Movement",
        'answer': "The sentence uses the word \"stupid\" to describe feminists and implies that they have a misguided belief that they are the only ones entitled to freedom of speech. This negative attitude suggests that the author is against the feminist movement. Therefore, the attitude of the sentence towards \"Feminist Movement\" is against.",
        'stance': 'against'
    },
    {
        'sentence': "Of mothers advising their daughter's to abort are insane! It's unacceptable!! #Condoms #ItsNotOnItsNotSafe",
        'target': "Legalization of Abortion",
        'answer': "The sentence expresses a negative attitude towards mothers advising their daughters to abort, which implies that the author is against the act of abortion. Therefore, the attitude of the sentence towards the target of legalization of abortion is also against. Therefore, the attitude of the sentence towards \"Legalization of Abortion\" is against.",
        'stance': 'against'
    },
    {
        'sentence': "@realDonaldTrump Don't let the @GOP tell you how to run!  #America doesn't need another politician.  #MakeAmericaGreatAgain only",
        'target': "Donald Trump",
        'answer': "The sentence is directed at Donald Trump and encourages him to not let the GOP (his own political party) tell him how to run. It also expresses support for Trump's campaign slogan \"Make America Great Again.\" Therefore, the attitude of the sentence towards the target (Donald Trump) is favorable. Therefore, the attitude of the sentence towards \"Donald Trump\" is favor.",
        'stance': 'favor'
    },
    {
        'sentence': "Pittsburgh had a Regatta without boats. River full of debris and swift currents. Too dangerous",
        'target': "Climate Change is a Real Concern",
        'answer': "he sentence is not expressing any attitude towards climate change. It is simply stating a fact about a specific event in Pittsburgh. The sentence does not indicate whether the event was caused by climate change or whether the author believes climate change is a real concern. Therefore, the attitude of the sentence towards \"Climate Change is a Real Concern\" is neutral.",
        'stance': 'neutral'
    },   
]

prompt_few_shot_cot = {
    'p_stance': pstance_few_shot,
    'vast': vast_few_shot,
    'sem16': semeval_few_shot,
}

def generate_few_shot_prompt(dataset, sentence, target):
    prompt_list = prompt_few_shot_cot[dataset]
    task_instruction = '''Stance detection is to determine the attitude or tendency towards a certain target through a given sentence, including favor, against and neutral. **Please read the following examples carefully and use them as references to judge the attitude of the sentence towards the target.**\n\n'''
    few_shot_prompt = task_instruction
    for idx, example in enumerate(prompt_list):
        few_shot_prompt += f'Example {idx + 1}:\n'
        few_shot_prompt += f'{example["sentence"]}\n'
        few_shot_prompt += f'What is the attitude of the sentence toward "{example["target"]}"? Please select the correct answer from "favor", "against" and "neutral".\n'
        few_shot_prompt += '''Answer this question with JSON format:
```json
{{
    "answer": "{answer}",
    "stance": '{stance}'
}}
```'''.format(answer=example['answer'], stance=example['stance'])
        few_shot_prompt += '\n\n'
    few_shot_prompt += f'Your sentence:\n{sentence}\n'
    few_shot_prompt += f'Question: What is the attitude of the sentence toward "{target}"? Please select the correct answer from "favor", "against" and "neutral".\n'
    few_shot_prompt += '''Answer this question with JSON format:
```json
{{
    "answer": "your answer",
    "stance": "favor" | "against" | "neutral"
}}
```'''
    return few_shot_prompt

CONSTRUCT_NON_CAUSAL_COUNTERFACTUAL_PROMPT = {
    "system_prompt": "You are an intelligent system that can rewrite sentences according to my requirements.",
    "stance_prompt": """I will give you a <Sentence> about a <Topic>, and known that the <Sentence> expresses a %s attitude to the <Topic>. Please rephrase the <Sentence> using different words and emotions, and rephrase the <Topic> using different words. Make sure the <Rephrased Sentence> and <Rephrased Topic> preserve the same meaning as the original, and after rephrasing the <Rephrased Sentence> still expresses a %s attitude to the <Rephrased Topic>. The <Sentence> is "%s". The <Topic> is "%s". Answer this question with JSON format:
```json
{{
    "Rephrased Sentence": "Your rephrased sentence.",
    "Rephrased Topic": "Your rephrased topic."
}}
```""",
    "unrelated_prompt": """I will give you a <Sentence> and a <Topic>, and known that the <Sentence> is unrelated to the <Topic>. Please rephrase the <Sentence> using different words and emotions, and rephrase the <Topic> using different words. Make sure the <Rephrased Sentence> and <Rephrased Topic> preserve the same meaning as the original, and after rephrasing the <Rephrased Sentence> is still unrelated to the <Rephrased Topic>. The <Sentence> is "%s". The <Topic> is "%s". Answer this question with JSON format:
```json
{{
    "Rephrased Sentence": "Your rephrased sentence.",
    "Rephrased Topic": "Your rephrased topic."
}}
```"""
}

CONSTRUCT_CAUSAL_COUNTERFACTUAL_PROMPT = {
    "system_prompt": "You are an intelligent system that can complete the task according to my requirements.",
    "stance2stance_prompt": """I will give you a <Sentence> about a <Topic>, and known that the <Sentence> expresses a %s attitude to the <Topic>. Please make minimal changes to the <Sentence> so that the <Revised Sentence> expresses a %s attitude to the <Topic>. The <Sentence> is "%s". The <Topic> is "%s". Please answer this question directly without any explanation with JSON format:
```json
{{
    "Revised Sentence": "Your revised sentence."
}}
```""",
    "stance2unrelated_prompt": """I will give you a <Sentence> about a <Topic>, and known that the <Sentence> expresses a %s attitude to the <Topic>. Please make minimal changes to the <Sentence> so that the <Revised Sentence> is unrelated to the <Topic>. The <Sentence> is "%s". The <Topic> is "%s". Please answer this question directly without any explanation with JSON format:
```json
{{
    "Revised Sentence": "Your revised sentence."
}}
```""",
    "unrelated2stance_prompt": """I will give you a <Sentence> about a <Topic>, and known that the <Sentence> is unrelated to the <Topic>. Please make minimal changes to the <Sentence> so that the <Revised Sentence> expresses a %s attitude to the <Topic>. The <Sentence> is "%s". The <Topic> is "%s". Please answer this question directly without any explanation with JSON format:
```json
{{
    "Revised Sentence": "Your revised sentence."
}}
```"""
}

if __name__ == '__main__':
    print(generate_few_shot_prompt('pstance', '[Sentence]', '[Target]'))
