import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
import random
import time as time_module
from datasets import load_dataset, Dataset
import numpy as np
import os
from google.colab import userdata

# Default configuration
DEFAULT_CONFIG = {
    "model_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "model_type": "llama",
    "dataset_name": "sst2",
    "data_path": "./data",
    "num_samples": 500,
    "output_dir": "outputs",
    "verbose": False,
    "few_shot_number": 0,
    "rate": 1.0,
    "rate_min": None,
    "rate_max": None,
    "rate_step": None,
    "heads": None,
    "layers": None,
    "exploring_mode": None,
    "use_8bit": False  
}

# Dataset configurations
DATASET_CONFIGS = {
    "hellaswag": {
        "instruction1": "Complete the following sentence with an appropriate ending.\n",
        "instruction2": "\nAnswer:",
        "template": "{ctx}\nA. {endings[0]}\nB. {endings[1]}\nC. {endings[2]}\nD. {endings[3]}",
        "dataset_path": "hellaswag",
        "text_key": ["ctx", "endings"],
        "split_names": {"train": "train", "validation": "validation"},
        "answer_key": "label",
        "answer_map": {"0": 0, "1": 1, "2": 2, "3": 3}
    },
    "ARCE": {
        "instruction1": "",
        "instruction2": "\nAnswer:",
        "template": "{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}",
        "dataset_path": "ai2_arc",
        "dataset_name": "ARC-Easy",
        "text_key": ["question", "choices"],
        "split_names": {"train": "train", "validation": "validation"},
        "answer_key": "answerKey",
        "answer_map": {"A": 0, "B": 1, "C": 2, "D": 3}
    },
    "ARCC": {
        "instruction1": "",
        "instruction2": "\nAnswer:",
        "template": "{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}",
        "dataset_path": "ai2_arc",
        "dataset_name": "ARC-Challenge",
        "text_key": ["question", "choices"],
        "split_names": {"train": "train", "validation": "validation"},
        "answer_key": "answerKey",
        "answer_map": {"A": 0, "B": 1, "C": 2, "D": 3}
    },
    "PIQA": {
        "instruction1": "Generate the correct solution to accomplish the following goal.\n",
        "instruction2": "\nAnswer:",
        "template": "{goal}\nA. {sol1}\nB. {sol2}",
        "dataset_path": "piqa",
        "text_key": ["goal", "sol1", "sol2"],
        "split_names": {"train": "train", "validation": "validation"},
        "answer_key": "label"
    },
    "OB": {
        "instruction1": "",
        "instruction2": "\nAnswer: ",
        "template": "{question_stem}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}",
        "dataset_path": "openbookqa",
        "dataset_name": "main",
        "text_key": ["question_stem", "choices"],
        "split_names": {"train": "train", "validation": "validation"},
        "answer_key": "answerKey",
        "answer_map": {"A": 0, "B": 1, "C": 2, "D": 3}
    },
    "COPA": {
        "instruction1": "",
        "instruction2": "\nAnswer:",
        "template": "{premise} What is the {question}?\nA. {choice1}\nB. {choice2}",
        "dataset_path": "super_glue",
        "dataset_name": "copa",
        "text_key": ["premise", "choice1", "choice2", "question"],
        "split_names": {"train": "train", "validation": "validation"},
        "answer_key": "label",
        "answer_map": {"0": 0, "1": 1}
    },
    "CQA": {
        "instruction1": "Generate the correct answer to the following question.\n",
        "instruction2": "\nAnswer:",
        "template": "{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nE. {choices[4]}",
        "dataset_path": "commonsense_qa",
        "text_key": ["question", "choices"],
        "split_names": {"train": "train", "validation": "validation"},
        "answer_key": "answerKey",
        "answer_map": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    },
    "AQUA": {
        "instruction1": "",
        "instruction2": "\nAnswer:",
        "template": "{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nE. {choices[4]}",
        "dataset_path": "aqua",
        "text_key": ["question", "options"],
        "split_names": {"train": "train", "validation": "validation"},
        "answer_key": "correct"
    },
    "MMLU": {
        # "instruction1": "",
        "instruction1": "Generate the correct answer to the following question.\n",
        "instruction2": "\nAnswer:",
        "template": "{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}",
        "dataset_path": "cais/mmlu",
        "dataset_name": "all",
        "text_key": ["question", "choices"],
        "split_names": {"train": "train", "validation": "validation"},
        "answer_key": "answer",
        "answer_map": {"A": 0, "B": 1, "C": 2, "D": 3}
    },
    "MathQA": {
        "instruction1": "",
        # "instruction1": "Generate the correct answer to the following question.\n",
        "instruction2": "\nAnswer:",
        "template": "{Problem}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nE. {choices[4]}",
        "dataset_path": "allenai/math_qa",
        "text_key": ["Problem", "options"],
        "split_names": {"train": "train", "validation": "validation"},
        "answer_key": "correct"
    },
    "LogiQA": {
        "instruction1": "",
        "instruction2": "\nAnswer:",
        "template": "{context}\n{query}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}",
        "dataset_path": "logiqa",
        "text_key": ["context", "query", "options"],
        "split_names": {"train": "train", "validation": "validation"},
        "answer_key": "correct_option"
    },
    # classification
    "sst2": {
        "instruction1": "Classify the sentence into one of the following categories: positive or negative.\n",
        "template": "Sentence: {text}",
        "instruction2": "\nAnswer:",
        "labels": [' negative', ' positive'],
        "dataset_path": "sst2",
        "text_key": "text",
        "split_names": {"train": "train", "validation": "validation"}
    },
    "sst5": {
        "instruction1": "Classify the sentence into one of the following categories: terrible, negative, neutral, positive, or great.\n",
        "template": "Sentence: {text}",
        "instruction2": "\nAnswer:",
        "labels": [' terrible', " negative", " neutral", " positive", " great"],
        "dataset_path": "SetFit/sst5",
        "text_key": "text",
        "split_names": {"train": "train", "validation": "validation"}
    },
    "MR": {
        "instruction1": "Classify the review into one of the following categories: positive or negative.\n",
        "template": "Review: {text}",
        "instruction2": "\n</think></think>\nAnswer:",
        "labels": [' negative', ' positive'],
        "dataset_path": "rotten_tomatoes",
        "text_key": "text",
        "split_names": {"train": "train", "validation": "validation"}
    },
    "SUBJ": {
        "instruction1": "Classify the input into one of the following categories: subjective or objective.\n",
        "template": "Input: {text}",
        "instruction2": "\nType:",
        "labels": [" objective", " subjective"],
        "dataset_path": "SetFit/subj",
        "text_key": "text",
        "split_names": {"train": "train", "validation": "test"}
    },
    "DBPedia": {
        "instruction1": "Classify the input into one of the following categories: company, school, artist, sport, politics, transportation, building, nature, village, animal, plant, album, film, or book.\n",
        "template": "Input: {content}",
        "instruction2":"\nCategory:",
        "labels": [" company", " school", " artist", " sport", " politics", " transportation", " building", " nature", " village", " animal", " plant", " album", " film", " book"],
        "dataset_path": "dbpedia_14",
        "text_key": "content",
        "split_names": {"train": "train", "validation": "test"}
    },
    "AGNews": {
        "instruction1": "Classify the news articles into one of the following categories: World, Sports, Business, or Technology.\n",
        "template": "Article: {text}",
        "instruction2": "\nCategories:",
        "labels": [" World", " Sports", " Business", " Technology"],
        "dataset_path": "ag_news",
        "text_key": "text",
        "split_names": {"train": "train", "validation": "test"}
    },
    "TREC": {
        "instruction1": "",
        "template": "What category does the question '{text}' belong to? Choose from Description, Entity, Expression, Person, Number, or Location.",
        "instruction2": "\nAnswer:",
        "labels": [" Description", " Entity", " Expression", " Person", " Number", " Location"],
        "dataset_path": "trec",
        "text_key": "text",
        "split_names": {"train": "train", "validation": "test"},
        "label_key": "coarse_label"
    },
    "CB": {
        "instruction1": "Read the premise and decide if it supports the hypothesis.\n",
        "template": "Premise: {premise} Hypothesis: {hypothesis}. Respond with Yes if it supports, No if it contradicts, or Maybe if it is neutral.",
        "instruction2": "\nAnswer:",
        "labels": ["Yes", "No", "Maybe"],
        "label_map": {
            "entailment": "Yes",
            "contradiction": "No",
            "neutral": "Maybe"
        },
        "dataset_path": "super_glue",
        "dataset_name": "cb",
        "text_key": ["premise", "hypothesis"],
        "split_names": {"train": "train", "validation": "validation"}
    },
    "BoolQ": {
        "instruction1": "Read the text and answer the question by True or False.\n",
        "template": "Text: {passage}\nQuestion: {question}?",
        "instruction2": "\nAnswer:",
        "labels": [" False", " True"],
        "dataset_path": "super_glue",
        "dataset_name": "boolq",
        "text_key": ["passage", "question"],
        "split_names": {"train": "train", "validation": "validation"}
    }
}

def set_random_seed(seed=42):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_mc_dataset(data_path, dataset_name, split='validation', num_samples=-1):
    """Load multiple-choice dataset from cache directory.
    
    Args:
        data_path: Path to the dataset cache directory
        dataset_name: Name of the dataset to load
        split: Dataset split to use (default: 'validation')
        num_samples: Number of samples to load (-1 for all)
    
    Returns:
        Dataset object containing the loaded data
    """
    config = DATASET_CONFIGS[dataset_name]
    
    if dataset_name == "LogiQA":
        try:
            dataset_list = []
            with open(os.path.join(data_path, "LogiQA/logiqa_val.json"), 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    processed_item = {
                        'context': item['context'],
                        'query': item['query'],
                        'choices': item['options'],  
                        'correct_option': item['correct_option'] 
                    }
                    dataset_list.append(processed_item)
            
            if not dataset_list:
                raise ValueError("No data loaded from LogiQA dataset")
            
            if num_samples > 0 and num_samples < len(dataset_list):
                dataset_list = random.sample(dataset_list, num_samples)
                
            return Dataset.from_list(dataset_list)
        except Exception as e:
            print(f"Error loading LogiQA dataset: {e}")
            print(f"Trying to read file: {os.path.join(data_path, 'LogiQA/logiqa_val.json')}")
            print(f"Current working directory: {os.getcwd()}")
            return None
    
    elif dataset_name == "MathQA":
        try:
            dataset_list = []
            with open(os.path.join(data_path, "MATHQA/math_qa_val.json"), 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    #  "a ) 238 sec , b ) 190 sec , c ) 667 sec , d ) 167 sec , e ) 176 sec"
                    options = [opt.strip() for opt in item['options'].split(',')]
                    choices = []
                    for opt in options:
                        # "a ) 238 sec" -> "238 sec"
                        choice = ' '.join(opt.split(')')[1:]).strip()
                        choices.append(choice)
                    
                    processed_item = {
                        'Problem': item['Problem'],
                        'choices': choices,
                        'correct': item['correct'],  # 'a', 'b', 'c', 'd', 'e'
                        'options': item['options'],
                        'Rationale': item.get('Rationale', '')  
                    }
                    dataset_list.append(processed_item)
            
            if not dataset_list:
                raise ValueError("No data loaded from MathQA dataset")
            
            if num_samples > 0 and num_samples < len(dataset_list):
                dataset_list = random.sample(dataset_list, num_samples)
                
            return Dataset.from_list(dataset_list)
        except Exception as e:
            print(f"Error loading MathQA dataset: {e}")
            print(f"Trying to read file: {os.path.join(data_path, 'MATHQA/math_qa_val.json')}")
            print(f"Current working directory: {os.getcwd()}")
            return None
    
    elif dataset_name == "MMLU":
        try:
            dataset = load_dataset(
                'cais/mmlu',
                'all',
                cache_dir=os.path.join(data_path, "MMLU/all"),
                trust_remote_code=True
            )
            dataset_list = []
            for item in dataset[split]:
                item['subject'] = item['subject'] 
                dataset_list.append(item)
            
            if num_samples > 0 and num_samples < len(dataset_list):
                dataset_list = random.sample(dataset_list, num_samples)
                
            return Dataset.from_list(dataset_list)
        except Exception as e:
            print(f"Error loading MMLU dataset: {e}")
            return None
    
    elif dataset_name == "AQUA":
        dataset_list = []
        with open(os.path.join(data_path, "AQUA/aqua.jsonl"), 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                dataset_list.append(item)
        
        if num_samples > 0 and num_samples < len(dataset_list):
            dataset_list = random.sample(dataset_list, num_samples)
            
        return Dataset.from_list(dataset_list)
    
    elif dataset_name == "PIQA":
        try:
            dataset = load_dataset(
                config["dataset_path"],
                cache_dir=os.path.join(data_path, dataset_name),
                trust_remote_code=True
            )
            split_name = config["split_names"][split]
            dataset = dataset[split_name]
            
            if "goal" not in dataset.features or "sol1" not in dataset.features or "sol2" not in dataset.features or "label" not in dataset.features:
                raise ValueError("PIQA dataset missing required fields")
            
            if num_samples > 0 and num_samples < len(dataset):
                dataset_indices = random.sample(range(len(dataset)), num_samples)
                dataset = dataset.select(dataset_indices)
            
            return dataset
            
        except Exception as e:
            print(f"Error loading PIQA dataset: {e}")
            print("Attempting to load from default path...")
            
            try:
                dataset = load_dataset(
                    config["dataset_path"],
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
                split_name = config["split_names"][split]
                dataset = dataset[split_name]
                
                if num_samples > 0 and num_samples < len(dataset):
                    dataset_indices = random.sample(range(len(dataset)), num_samples)
                    dataset = dataset.select(dataset_indices)
                
                return dataset
            except Exception as e:
                print(f"Failed to load PIQA dataset from both paths: {e}")
                return None
    
    cache_mapping = {
        # multi-choice
        "hellaswag": "data/hellaswag",
        "ARCE": "data/ARCE",
        "ARCC": "data/ARCC",
        "PIQA": "data/PIQA",
        "OB": "data/OB",
        "COPA": "data/COPA",
        "CQA": "data/CQA",
        
        # classification
        "sst2": "./data/sst2",
        "sst5": "./data/sst5",
        "MR": "./data/MR",
        "SUBJ": "./data/SUBJ",
        "DBPedia": "./data/DBPedia",
        "AGNews": "./data/AGNews",
        "TREC": "./data/TREC",
        "CB": "./data/CB",
        "BoolQ": "./data/BoolQ"
    }
    
    # Use specific cache directory for each dataset
    cache_dir = cache_mapping[dataset_name]
    print(f"Loading from cache directory: {cache_dir}")
    dataset = load_dataset(cache_dir)
    split_name = config["split_names"][split]
    dataset = dataset[split_name]
        
    if num_samples > 0 and num_samples < len(dataset):
        dataset_indices = random.sample(range(len(dataset)), num_samples)
        dataset = dataset.select(dataset_indices)
        
    return dataset
    
def format_mc_prompt(item, dataset_name):
    """Format prompt for multiple-choice or classification tasks.
    
    Args:
        item: Dataset item containing question and choices
        dataset_name: Name of the dataset being used
    
    Returns:
        Tuple of (instruction1, formatted_text, instruction2)
    """
    config = DATASET_CONFIGS[dataset_name]
    instruction1 = config["instruction1"]
    instruction2 = config["instruction2"]
    
    is_classification = "labels" in config
    
    if is_classification:
        if isinstance(config["text_key"], list):
            formatted_text = config["template"].format(**item)
        else:
            formatted_text = config["template"].format(**{config["text_key"]: item[config["text_key"]]})
        return instruction1, formatted_text, instruction2
    
    if dataset_name in ["ARCE", "ARCC"]:
        question = item['question']
        choice_texts = item['choices']['text'] 
        prompt = question
        for i, choice in enumerate(choice_texts):
            prompt += '\n' + chr(65+i) + '. ' + choice
        return instruction1, prompt, instruction2
    
    elif dataset_name == "MMLU":
        question = item['question']
        choices = item['choices']
        prompt = question
        for i, choice in enumerate(choices):
            prompt += '\n' + chr(65+i) + '. ' + choice
        return instruction1, prompt, instruction2
    
    elif dataset_name == "OB":
        question = item['question_stem']
        choice_texts = item['choices']['text']
        prompt = question
        for i, choice in enumerate(choice_texts):
            prompt += '\n' + chr(65+i) + '. ' + choice
        return instruction1, prompt, instruction2
    
    elif dataset_name == "hellaswag":
        question = item['ctx']
        choice_texts = item['endings']
        prompt = question
        for i, choice in enumerate(choice_texts):
            prompt += '\n' + chr(65+i) + '. ' + choice
        return instruction1, prompt, instruction2
    
    elif dataset_name == "AQUA":
        question = item['question']
        choice_texts = []
        for option in item['options']:
            choice = option.split(')', 1)[1].strip()
            choice_texts.append(choice)
        
        prompt = question
        for i, choice in enumerate(choice_texts):
            prompt += '\n' + chr(65+i) + '. ' + choice
        return instruction1, prompt, instruction2
    
    elif dataset_name == "MathQA":
        question = item['Problem']
        choice_texts = item['choices']
        prompt = question
        for i, choice in enumerate(choice_texts):
            prompt += '\n' + chr(65+i) + '. ' + choice
        return instruction1, prompt, instruction2
    
    elif dataset_name == "COPA":
        prompt = config["template"].format(**item)
        return instruction1, prompt, instruction2
    
    elif dataset_name == "PIQA":
        prompt = config["template"].format(**item)
        return instruction1, prompt, instruction2
    
    elif dataset_name == "CQA":
        question = item['question']
        choice_texts = item['choices']['text']
        prompt = question
        for i, choice in enumerate(choice_texts):
            prompt += '\n' + chr(65+i) + '. ' + choice
        return instruction1, prompt, instruction2
    
    elif dataset_name == "LogiQA":
        context = item['context']
        query = item['query']
        choice_texts = item['choices']
        prompt = context + '\n' + query
        for i, choice in enumerate(choice_texts):
            prompt += '\n' + chr(65+i) + '. ' + choice
        return instruction1, prompt, instruction2
    
    else:
        if isinstance(config["text_key"], list):
            prompt = config["template"].format(**item)
        else:
            prompt = config["template"].format(**{config["text_key"]: item[config["text_key"]]})
        return instruction1, prompt, instruction2

def get_formatted_evaluation_dataset(dataset, dataset_name, num_samples=-1):
    """Format dataset for evaluation with standardized structure.
    
    Args:
        dataset: Raw dataset to format
        dataset_name: Name of the dataset being used
        num_samples: Number of samples to include (-1 for all)
    
    Returns:
        List of formatted dataset items with standardized fields
    """
    formatted_dataset = []
    config = DATASET_CONFIGS[dataset_name]
    is_classification = "labels" in config  
    
    if num_samples > 0 and num_samples < len(dataset):
        dataset_indices = random.sample(range(len(dataset)), num_samples)
        dataset = [dataset[i] for i in dataset_indices]
    
    for item in dataset:
        instruction1, text, instruction2 = format_mc_prompt(item, dataset_name)
        
        if is_classification:
            label_key = config.get("label_key", "label")
            if dataset_name == "CB":
                # Yes/No/Maybe
                raw_label = item[label_key]
                label = config["label_map"][raw_label]  
            else:
                label = item[label_key]
            label_choices = config["labels"]
        else:
            if dataset_name == "MMLU":
                label = item[config["answer_key"]]
            elif dataset_name == "AQUA":
                answer_key = item['correct']
                label = ord(answer_key) - ord('A')
            elif dataset_name in ["ARCE", "ARCC", "OB"]:
                answer_key = item[config["answer_key"]]
                label = ord(answer_key) - ord('A')
            elif dataset_name in ["hellaswag", "PIQA", "COPA"]:
                if dataset_name == "PIQA":
                    label = item[config["answer_key"]]
                    label_choices = ["A", "B"] 
                else:
                    answer_key = str(item[config["answer_key"]])
                    label = config["answer_map"][answer_key]
                    label_choices = [chr(65 + i) for i in range(5)]  # A, B, C, D, E
            elif dataset_name == "MathQA":
                # 'a', 'b', 'c', 'd', 'e'
                answer_key = item['correct']
                label = ord(answer_key.upper()) - ord('A')
            elif dataset_name == "LogiQA":
                label = item['correct_option']  
            elif dataset_name == "CQA":
                answer_key = item['answerKey']
                label = ord(answer_key) - ord('A')
            else:
                # Default handling
                label = item[config["answer_key"]]
            
            if dataset_name == "PIQA" or dataset_name == "COPA":
                label_choices = ["A", "B"]  
            else:
                label_choices = [chr(65 + i) for i in range(5)]  # A, B, C, D, E
            
        formatted_dataset.append({
            'name': dataset_name,
            'instruction1': instruction1,
            'text': text,
            'instruction2': instruction2,
            'label': label,
            'label_choices': label_choices
        })
    return formatted_dataset

def extract_label_from_response(response, label_choices):
    """Extract classification label from model response.
    
    Args:
        response: Raw model output text
        label_choices: List of possible label values
    
    Returns:
        Index of the matched label, or None if no match found
    """
    response = response.strip()
    
    if label_choices == ["Yes", "No", "Maybe"]:
        response = response.lower()
        if response == "yes":
            return 0
        elif response == "no":
            return 1
        elif response == "maybe":
            return 2
        return None
    
    response = response.lower()
    for i, label in enumerate(label_choices):
        if label.lower().strip() in response:
            return i
    return None

def extract_answer_from_response(response):
    """Extract answer option (A-E) from model response.
    
    Args:
        response: Raw model output text
    
    Returns:
        Integer index (0-4) corresponding to answer A-E, or None if no match found
    """
    # Clean and standardize the response
    response = response.strip().lower()
    
    if response.startswith('a'):
        return 0
    elif response.startswith('b'):
        return 1
    elif response.startswith('c'):
        return 2
    elif response.startswith('d'):
        return 3
    elif response.startswith('e'):
        return 4
    
    if response.startswith('1') or response == '0':
        return 0
    elif response.startswith('2'):
        return 1
    elif response.startswith('3'):
        return 2
    elif response.startswith('4'):
        return 3
    elif response.startswith('5'):
        return 4
    
    for letter, idx in [('a', 0), ('b', 1), ('c', 2), ('d', 3), ('e', 4)]:
        if letter in response:
            return idx
            
    for num, idx in [('1', 0), ('2', 1), ('3', 2), ('4', 3), ('5', 4)]:
        if num in response:
            return idx
    
    return None

def prepare_input(tokenizer, prompts):
    """Prepare model input"""
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')
    return input_tokens

def load_model(model_path, model_type="llama", hftoken=None, use_8bit=False):
    """Load model and tokenizer with specified configuration
    
    Args:
        model_path: Path to the model
        model_type: Type of model (llama or qwen)
        hftoken: HuggingFace access token
        use_8bit: Whether to use 8-bit quantization
    
    Returns:
        Tuple of (model, tokenizer)
    """
    if 'llama' in model_type:
        from transformers import AutoTokenizer, BitsAndBytesConfig
        from modelling_llama_open import LlamaForCausalLM
        
        # Check if token is provided
        if hftoken is None:
            raise ValueError("Please provide HuggingFace token")

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            padding_side="left",
            use_auth_token=hftoken,
            trust_remote_code=True
        )
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
        
        # Configure quantization if needed
        quantization_config = None
        if use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        # Load the model
        model = LlamaForCausalLM.from_pretrained(
            model_path, 
            use_auth_token=hftoken,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device) # This might still be useful depending on your GPU

    elif 'qwen' in model_type:
        from transformers import AutoTokenizer, BitsAndBytesConfig
        from models.modeling_qwen2 import Qwen2ForCausalLM

        # Check if token is provided
        if hftoken is None:
            raise ValueError("Please provide HuggingFace token")

        print("Loading Qwen2 model with attention enhancement...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
            use_auth_token=hftoken
        )

        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1

        # Configure quantization if needed
        quantization_config = None
        if use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        # Load model using modified Qwen2ForCausalLM
        model = Qwen2ForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=False,  # Use locally modified code
            torch_dtype=torch.float16,
            attn_implementation="sdpa",  # flash_attention_2, sdpa
            quantization_config=quantization_config,
            use_auth_token=hftoken
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device) # This might still be useful depending on your GPU

        print("Qwen2 model loaded successfully with attention enhancement")

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model, tokenizer

def get_few_shot_examples(dataset, dataset_name, num_shots, is_ins1=False):
    """Generate few-shot examples from the dataset
    
    Args:
        dataset: Dataset to sample from
        dataset_name: Name of the dataset
        num_shots: Number of examples to generate
        is_ins1: Whether to include instruction1 in the prompt
    
    Returns:
        String containing formatted few-shot examples
    """
    if num_shots == 0:
        return ""
    
    # Set fixed random seed
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Randomly select examples from the dataset
    selected_examples = random.sample(list(dataset), num_shots)
    
    # Format examples
    few_shot_prompt = ""
    config = DATASET_CONFIGS[dataset_name]
    is_classification = "labels" in config  # Check if it's a classification task
    
    for example in selected_examples:
        # Use the same formatting logic as format_mc_prompt
        instruction1, text, instruction2 = format_mc_prompt(example, dataset_name)
        
        # Get correct answer
        if is_classification:
            # Classification task: use predefined labels
            label_key = config.get("label_key", "label")
            if dataset_name == "CB":
                # CB dataset label mapping
                label_map = {
                    "entailment": "Yes",
                    "contradiction": "No",
                    "neutral": "Maybe"
                }
                answer = label_map[example[label_key]]
            else:
                label_idx = example[label_key]
                answer = config["labels"][label_idx]
        else:
            # Multiple choice task: get answer based on dataset type
            if dataset_name == "MMLU":
                answer = chr(65 + example['answer'])
            elif dataset_name == "AQUA":
                answer = example['correct']
            elif dataset_name in ["ARCE", "ARCC", "OB"]:
                answer = example['answerKey']
            elif dataset_name in ["hellaswag", "PIQA", "COPA"]:
                answer = chr(65 + example['label'])
            elif dataset_name == "MathQA":
                answer = example['correct'].upper()
            elif dataset_name == "LogiQA":
                answer = chr(65 + example['correct_option'])
            elif dataset_name == "CQA":
                answer = example['answerKey']
        
        few_shot_prompt += (instruction1 if is_ins1 else "") + text + instruction2 + answer + "\n\n"
    
    return few_shot_prompt

def evaluate_mc(model, tokenizer, dataset, dataset_name, rate=1.0, few_shot_number=0, 
               verbose=False, output_path=None,
               target_layers=None, target_heads=None, num_samples=-1, fixed_few_shot_prompt=""):
    """Evaluate multiple-choice and classification datasets
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Dataset to evaluate on
        dataset_name: Name of the dataset
        rate: Attention enhancement rate
        few_shot_number: Number of few-shot examples
        verbose: Whether to print detailed information
        output_path: Path to save evaluation results
        target_layers: Target layers for attention enhancement
        target_heads: Target heads for attention enhancement
        num_samples: Number of samples to evaluate (-1 for all)
        fixed_few_shot_prompt: Pre-generated few-shot examples
    
    Returns:
        Accuracy of the model on the dataset
    """
    config = DATASET_CONFIGS[dataset_name]
    is_classification = "labels" in config

    eval_dataset = get_formatted_evaluation_dataset(dataset, dataset_name, num_samples)
    correct_counts = 0
    total_samples = len(eval_dataset)
    invalid_predictions = 0

    # Ensure consistent format for target_layers and target_heads
    if target_layers is not None and not isinstance(target_layers, list):
        target_layers = [target_layers]
    if target_heads is not None and not isinstance(target_heads, list):
        target_heads = [target_heads]

    print(f"\nTesting with attention enhancement rate: {rate}")
    print(f"Few-shot examples: {few_shot_number}")
    if target_layers is not None:
        print(f"Target layers: {target_layers}")
    if target_heads is not None:
        print(f"Target heads: {target_heads}")

    # Generate new few-shot examples if not provided but few_shot_number > 0
    if not fixed_few_shot_prompt and few_shot_number > 0:
        fixed_few_shot_prompt = get_few_shot_examples(dataset, dataset_name, few_shot_number)
        print(f"\nGenerated few-shot examples:\n{fixed_few_shot_prompt}")

    # Handle output file
    if output_path and output_path.strip():  # Ensure output_path is not empty
        # Generate output filename with timestamp
        timestamp = time_module.strftime("%Y%m%d_%H%M%S")
        output_filename = f"eval_{dataset_name}_{timestamp}.json"
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Create full output path
        full_output_path = os.path.join(output_path, output_filename)
        
        # Create experiment configuration dictionary
        experiment_config = {
            "dataset": dataset_name,
            "task_type": "Classification" if is_classification else "Multiple Choice",
            "rate": rate,
            "few_shot_examples": few_shot_number,
            "target_layers": target_layers,
            "target_heads": target_heads,
            "timestamp": timestamp,
            "results": []  # Store results for each sample
        }

    batch_size = 1
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(eval_dataset), batch_size), 
                     desc=f"Evaluating {dataset_name}", 
                     leave=True,  
                     position=0, 
                     ncols=100):  
            item = eval_dataset[i]
            
            # Build prompt
            few_shot_prompt = item['instruction1'] + fixed_few_shot_prompt + item['text'] + item['instruction2']
            
            # Prepare input
            encode_inputs = prepare_input(tokenizer, [few_shot_prompt])
            
            # Set text_start and text_end to 0,0
            text_start = 0
            text_end = 0

            # Generate answer
            outputs = model.generate(
                **encode_inputs,
                max_new_tokens=1,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                input_len=(text_start, text_end, rate, target_layers, target_heads)
            )

            # Decode model output
            pred_text = tokenizer.batch_decode(outputs[:, encode_inputs['input_ids'].shape[1]:], 
                                             skip_special_tokens=True)[0]
            
            # Extract predicted label
            if is_classification:
                pred_label = extract_label_from_response(pred_text, config["labels"])
            else:
                pred_label = extract_answer_from_response(pred_text)
            
            # Get true label
            true_label = item['label']
            if isinstance(true_label, str):
                if dataset_name == "CB":
                    # CB dataset: directly find mapped label in labels list
                    true_label = config["labels"].index(true_label)
                else:
                    # Other datasets: keep existing processing
                    try:
                        true_label = config["labels"].index(true_label)
                    except ValueError:
                        true_label = config["labels"].index(f" {true_label}")
            else:
                true_label = int(true_label)
            
            # Check if answer is correct
            if pred_label is not None:
                is_correct = pred_label == true_label
                correct_counts += int(is_correct)
            else:
                invalid_predictions += 1
                is_correct = False

            if verbose:
                print(f"\nInput: {few_shot_prompt}")
                print(f"Model output: {pred_text}")
                print(f"Extracted answer: {pred_label}")
                print(f"True answer: {true_label}")
                print(f"Is correct: {is_correct}")

            # Record results
            if output_path and output_path.strip():  # Ensure output_path is not empty
                result = {
                    "sample_id": i,
                    "text": item["text"],
                    "prompt": few_shot_prompt,
                    "model_output": pred_text,
                    "predicted_label": pred_label,
                    "true_label": true_label,
                    "is_correct": is_correct
                }
                experiment_config["results"].append(result)
    
    # Write complete JSON file after loop
    if output_path and output_path.strip():  # Ensure output_path is not empty
        # Add final statistics
        experiment_config["statistics"] = {
            "total_samples": total_samples,
            "valid_predictions": total_samples - invalid_predictions,
            "invalid_predictions": invalid_predictions,
            "correct_predictions": correct_counts,
            "accuracy": correct_counts / total_samples
        }
        
        # Write JSON file
        with open(full_output_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_config, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {full_output_path}")
    
    # Calculate and print accuracy
    accuracy = correct_counts / total_samples
    print(f"\n{dataset_name} Statistics:")
    print(f"Rate: {rate}")
    print(f"Total samples: {total_samples}")
    print(f"Valid predictions: {total_samples - invalid_predictions}")
    print(f"Invalid predictions: {invalid_predictions}")
    print(f"Correct predictions: {correct_counts}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy

def evaluate_all_layers_heads(model, tokenizer, dataset, dataset_name, rate, mode, args):
    """Evaluate performance across all layers and heads
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Dataset to evaluate on
        dataset_name: Name of the dataset
        rate: Attention enhancement rate
        mode: Evaluation mode (single_head, both, all_layers, all_heads)
        args: Additional arguments
    
    Returns:
        Best performing layer-head combination or overall results
    """
    num_layers = len(model.model.layers)
    num_heads = model.config.num_attention_heads
    
    # Create results file
    timestamp = time_module.strftime("%Y%m%d_%H%M%S")
    result_file = f"results/layer_head_test_{dataset_name}_{timestamp}.txt"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    # Write experiment configuration
    with open(result_file, 'w') as f:
        f.write(f"=== Layer-Head Analysis Configuration ===\n")
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Rate: {rate}\n")
        f.write(f"Number of layers: {num_layers}\n")
        f.write(f"Number of heads per layer: {num_heads}\n")
        f.write(f"Number of samples: {args.num_samples}\n")
        if mode == "single_head":
            f.write(f"Target head: {args.target_head}\n")
        f.write(f"Analysis mode: {mode}\n")
        f.write(f"Time: {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        f.write("=== Results ===\n")
        
        if mode == "both":
            f.write("Accuracy\tTime(s)\n")
        elif mode in ["all_layers", "single_head"]:
            f.write("Layer\tAccuracy\tTime(s)\n")
        elif mode == "all_heads":
            f.write("Layer\tAccuracy\tTime(s)\n")
        else:
            f.write("Layer\tHead\tAccuracy\tTime(s)\n")
    
    # Store all results for sorting
    all_results = []
    
    if mode == "single_head":
        # Test specific head across all layers
        head_idx = args.target_head
        print(f"\nTesting head {head_idx} across all layers")
        
        for layer_idx in range(num_layers):
            print(f"  Testing layer {layer_idx}")
            start_time = time_module.time()
            
            accuracy = evaluate_mc(
                model, tokenizer, dataset, dataset_name,
                rate=rate,
                few_shot_number=args.few_shot_number,
                verbose=args.verbose,
                output_path=None,
                target_layers=None,
                target_heads=[head_idx],
                num_samples=args.num_samples
            )
            
            end_time = time_module.time()
            run_time = end_time - start_time
            
            result = (layer_idx, accuracy, run_time)
            all_results.append(result)
            
            with open(result_file, 'a') as f:
                f.write(f"{layer_idx}\t{accuracy:.4f}\t{run_time:.2f}\n")
    
    elif mode == "both":
        # Test all layers and heads together
        print("\nTesting all layers and heads together")
        start_time = time_module.time()
        
        accuracy = evaluate_mc(
            model, tokenizer, dataset, dataset_name,
            rate=rate,
            few_shot_number=args.few_shot_number,
            verbose=args.verbose,
            output_path=None,
            target_layers=None,  # None means use all layers
            target_heads=None,   # None means use all heads
            num_samples=args.num_samples
        )
        
        end_time = time_module.time()
        run_time = end_time - start_time
        
        result = (accuracy, run_time)
        all_results.append(result)
        
        with open(result_file, 'a') as f:
            f.write(f"{accuracy:.4f}\t{run_time:.2f}\n")
            
    elif mode == "all_layers":
        # Test each head across all layers
        for head_idx in range(num_heads):
            print(f"\nTesting head {head_idx} across all layers")
            start_time = time_module.time()
            
            accuracy = evaluate_mc(
                model, tokenizer, dataset, dataset_name,
                rate=rate,
                few_shot_number=args.few_shot_number,
                verbose=args.verbose,
                output_path=None,
                target_layers=None,
                target_heads=[head_idx],
                num_samples=args.num_samples
            )
            
            end_time = time_module.time()
            run_time = end_time - start_time
            
            result = (head_idx, accuracy, run_time)
            all_results.append(result)
            
            with open(result_file, 'a') as f:
                f.write(f"{head_idx}\t{accuracy:.4f}\t{run_time:.2f}\n")
                
    elif mode == "all_heads":
        # Test each layer with all heads
        for layer_idx in range(num_layers):
            print(f"\nTesting layer {layer_idx} with all heads")
            start_time = time_module.time()
            
            accuracy = evaluate_mc(
                model, tokenizer, dataset, dataset_name,
                rate=rate,
                few_shot_number=args.few_shot_number,
                verbose=args.verbose,
                output_path=None,
                target_layers=None,
                target_heads=None,
                num_samples=args.num_samples
            )
            
            end_time = time_module.time()
            run_time = end_time - start_time
            
            result = (layer_idx, accuracy, run_time)
            all_results.append(result)
            
            with open(result_file, 'a') as f:
                f.write(f"{layer_idx}\t{accuracy:.4f}\t{run_time:.2f}\n")
    else:
        # Original logic: test each layer-head combination
        for layer_idx in range(num_layers):
            print(f"\nTesting layer {layer_idx}")
            for head_idx in range(num_heads):
                print(f"  Testing head {head_idx}")
                start_time = time_module.time()
                
                accuracy = evaluate_mc(
                    model, tokenizer, dataset, dataset_name,
                    rate=rate,
                    few_shot_number=args.few_shot_number,
                    verbose=args.verbose,
                    output_path=None,
                    target_layers=None,
                    target_heads=[head_idx],
                    num_samples=args.num_samples
                )
                
                end_time = time_module.time()
                run_time = end_time - start_time
                
                result = (layer_idx, head_idx, accuracy, run_time)
                all_results.append(result)
                
                with open(result_file, 'a') as f:
                    f.write(f"{layer_idx}\t{head_idx}\t{accuracy:.4f}\t{run_time:.2f}\n")
    
    # Write summary results
    with open(result_file, 'a') as f:
        f.write("\n=== Summary ===\n")
        if mode == "single_head":
            f.write(f"Results for Head {args.target_head}:\n")
            f.write("Layers Ranked by Accuracy:\n")
            f.write("Rank\tLayer\tAccuracy\tTime(s)\n")
            sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)
            for i, (layer, acc, time) in enumerate(sorted_results, 1):
                f.write(f"{i}\t{layer}\t{acc:.4f}\t{time:.2f}\n")
            
            # Find best layer
            best_layer, best_accuracy, _ = sorted_results[0]
            f.write(f"\nBest Layer: {best_layer}")
            f.write(f"\nBest Accuracy: {best_accuracy:.4f}\n")
        elif mode == "both":
            f.write("Overall Accuracy:\n")
            accuracy, run_time = all_results[0]
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Time: {run_time:.2f}s\n")
        elif mode in ["all_layers", "all_heads"]:
            f.write("Heads Ranked by Accuracy:\n")
            f.write("Rank\tHead\tAccuracy\tTime(s)\n")
            sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)
            for i, (head, acc, time) in enumerate(sorted_results, 1):
                f.write(f"{i}\t{head}\t{acc:.4f}\t{time:.2f}\n")
        else:
            f.write("Top 10 Best Layer-Head Combinations:\n")
            f.write("Rank\tLayer\tHead\tAccuracy\tTime(s)\n")
            sorted_results = sorted(all_results, key=lambda x: x[2], reverse=True)
            for i, (layer, head, acc, time) in enumerate(sorted_results[:10], 1):
                f.write(f"{i}\t{layer}\t{head}\t{acc:.4f}\t{time:.2f}\n")
        
        f.write("=" * 50 + "\n")
    
    print(f"\nLayer-head analysis results saved to: {result_file}")
    
    if mode == "single_head":
        # Return best layer and current head combination
        best_layer, best_accuracy, _ = sorted_results[0]
        return (best_layer, args.target_head, best_accuracy, 0)
    else:
        return all_results[0]

def explore_all_combinations(model, tokenizer, dataset, dataset_name, rate, args):
    """Explore performance of all combinations of specified layers and heads
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Dataset to evaluate on
        dataset_name: Name of the dataset
        rate: Attention enhancement rate
        args: Additional arguments containing layer and head specifications
    
    Returns:
        List of results for each layer-head combination
    """
    results = []
    
    # Use specified layers and heads, or all if not specified
    target_layers = args.layers if args.layers is not None else list(range(len(model.model.layers)))
    target_heads = args.heads if args.heads is not None else list(range(model.config.num_attention_heads))
    
    print(f"\nExploring combinations of specified layers and heads:")
    print(f"Target layers: {target_layers}")
    print(f"Target heads: {target_heads}")
    
    for layer_idx in target_layers:
        print(f"\nTesting layer {layer_idx}")
        for head_idx in target_heads:
            print(f"  Testing head {head_idx}")
            accuracy = evaluate_mc(
                model, tokenizer, dataset, dataset_name,
                rate=rate,
                few_shot_number=args.few_shot_number,
                verbose=args.verbose,
                output_path=None,
                target_layers=[layer_idx],
                target_heads=[head_idx],
                num_samples=args.num_samples
            )
            results.append({
                'layer': layer_idx,
                'head': head_idx,
                'accuracy': accuracy
            })
            
            # Print current result
            print(f"    Accuracy: {accuracy:.4f}")
    
    return results

def explore_heads_performance(model, tokenizer, dataset, dataset_name, rate, args):
    """Explore average performance of each head across specified layers"""
    results = []
    # If layers not specified, use all layers
    target_layers = args.layers if args.layers is not None else list(range(len(model.model.layers)))
    # If heads not specified, use all heads
    target_heads = args.heads if args.heads is not None else list(range(model.config.num_attention_heads))
    
    print(f"\nExploring performance across {len(target_heads)} heads with {len(target_layers)} layers")
    print(f"Target heads: {target_heads}")
    print(f"Target layers: {target_layers}")
    
    for head_idx in tqdm(target_heads, desc="Testing heads"):
        accuracy = evaluate_mc(
            model, tokenizer, dataset, dataset_name,
            rate=rate,
            few_shot_number=args.few_shot_number,
            verbose=args.verbose,
            output_path=None,
            target_layers=target_layers,
            target_heads=[head_idx],  # Test one head at a time
            num_samples=args.num_samples
        )
        results.append({
            'head': head_idx,
            'layers': target_layers,
            'accuracy': accuracy
        })
        
        # Print current head's result
        print(f"\nHead {head_idx} accuracy: {accuracy:.4f}")
    
    return results

def explore_layers_performance(model, tokenizer, dataset, dataset_name, rate, args):
    """Explore average performance of each layer across specified heads"""
    results = []
    # If layers not specified, use all layers
    target_layers = args.layers if args.layers is not None else list(range(len(model.model.layers)))
    # If heads not specified, use all heads
    target_heads = args.heads if args.heads is not None else list(range(model.config.num_attention_heads))
    
    print(f"\nExploring performance across {len(target_layers)} layers with {len(target_heads)} heads")
    print(f"Target layers: {target_layers}")
    print(f"Target heads: {target_heads}")
    
    for layer_idx in tqdm(target_layers, desc="Testing layers"):
        accuracy = evaluate_mc(
            model, tokenizer, dataset, dataset_name,
            rate=rate,
            few_shot_number=args.few_shot_number,
            verbose=args.verbose,
            output_path=None,
            target_layers=[layer_idx],  # Test one layer at a time
            target_heads=target_heads,  # Use all specified heads
            num_samples=args.num_samples
        )
        results.append({
            'layer': layer_idx,
            'heads': target_heads,
            'accuracy': accuracy
        })
        
        # Print current layer's result
        print(f"\nLayer {layer_idx} accuracy: {accuracy:.4f}")
    
    return results

def save_exploration_results(results, result_file, args):
    """Save exploration results to file"""
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    with open(result_file, 'w') as f:
        # Write configuration information
        f.write(f"=== Exploration Results ===\n")
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Rate: {args.rate}\n")
        f.write(f"Exploring mode: {args.exploring_mode}\n")
        f.write(f"Target heads: {args.heads}\n")
        f.write(f"Target layers: {args.layers}\n")
        f.write(f"Time: {time_module.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        # Write results
        if args.exploring_mode == 0:
            # Write results for all combinations
            f.write("=== All Combinations Results ===\n")
            f.write("Layer\tHead\tAccuracy\n")
            sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
            for result in sorted_results:
                f.write(f"{result['layer']}\t{result['head']}\t{result['accuracy']:.4f}\n")
            
            # Add summary section
            f.write("\n=== Summary ===\n")
            best_result = max(results, key=lambda x: x['accuracy'])
            worst_result = min(results, key=lambda x: x['accuracy'])
            avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
            
            f.write(f"Best combination: Layer {best_result['layer']}, Head {best_result['head']} (accuracy: {best_result['accuracy']:.4f})\n")
            f.write(f"Worst combination: Layer {worst_result['layer']}, Head {worst_result['head']} (accuracy: {worst_result['accuracy']:.4f})\n")
            f.write(f"Average accuracy across all combinations: {avg_accuracy:.4f}\n")
            
            # Add distribution analysis
            accuracies = [r['accuracy'] for r in results]
            quartiles = np.percentile(accuracies, [25, 50, 75])
            f.write(f"\nAccuracy Distribution:\n")
            f.write(f"25th percentile: {quartiles[0]:.4f}\n")
            f.write(f"Median: {quartiles[1]:.4f}\n")
            f.write(f"75th percentile: {quartiles[2]:.4f}\n")
            
            # Statistics by layer
            f.write("\nPer-Layer Statistics:\n")
            layer_stats = {}
            for r in results:
                layer = r['layer']
                if layer not in layer_stats:
                    layer_stats[layer] = []
                layer_stats[layer].append(r['accuracy'])
            
            for layer, accs in sorted(layer_stats.items()):
                avg = sum(accs) / len(accs)
                best = max(accs)
                f.write(f"Layer {layer}: Avg = {avg:.4f}, Best = {best:.4f}\n")
            
            # Statistics by head
            f.write("\nPer-Head Statistics:\n")
            head_stats = {}
            for r in results:
                head = r['head']
                if head not in head_stats:
                    head_stats[head] = []
                head_stats[head].append(r['accuracy'])
            
            for head, accs in sorted(head_stats.items()):
                avg = sum(accs) / len(accs)
                best = max(accs)
                f.write(f"Head {head}: Avg = {avg:.4f}, Best = {best:.4f}\n")
        
        elif args.exploring_mode == 1:
            f.write("Head\tLayers\tAccuracy\n")
            sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
            for result in sorted_results:
                f.write(f"{result['head']}\t{len(result['layers'])} layers\t{result['accuracy']:.4f}\n")
            
            # Add summary section
            f.write("\n=== Summary ===\n")
            best_result = max(results, key=lambda x: x['accuracy'])
            worst_result = min(results, key=lambda x: x['accuracy'])
            avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
            
            f.write(f"Best performing head: {best_result['head']} (accuracy: {best_result['accuracy']:.4f})\n")
            f.write(f"Worst performing head: {worst_result['head']} (accuracy: {worst_result['accuracy']:.4f})\n")
            f.write(f"Average accuracy across all heads: {avg_accuracy:.4f}\n")
            
            # Add distribution analysis
            accuracies = [r['accuracy'] for r in results]
            quartiles = np.percentile(accuracies, [25, 50, 75])
            f.write(f"\nAccuracy Distribution:\n")
            f.write(f"25th percentile: {quartiles[0]:.4f}\n")
            f.write(f"Median: {quartiles[1]:.4f}\n")
            f.write(f"75th percentile: {quartiles[2]:.4f}\n")
        
        else:  # args.exploring_mode == 2
            f.write("Layer\tHeads\tAccuracy\n")
            sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
            for result in sorted_results:
                f.write(f"{result['layer']}\t{len(result['heads'])} heads\t{result['accuracy']:.4f}\n")
            
            # Add summary section
            f.write("\n=== Summary ===\n")
            best_result = max(results, key=lambda x: x['accuracy'])
            worst_result = min(results, key=lambda x: x['accuracy'])
            avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
            
            f.write(f"Best performing layer: {best_result['layer']} (accuracy: {best_result['accuracy']:.4f})\n")
            f.write(f"Worst performing layer: {worst_result['layer']} (accuracy: {worst_result['accuracy']:.4f})\n")
            f.write(f"Average accuracy across all layers: {avg_accuracy:.4f}\n")
            
            # Add distribution analysis
            accuracies = [r['accuracy'] for r in results]
            quartiles = np.percentile(accuracies, [25, 50, 75])
            f.write(f"\nAccuracy Distribution:\n")
            f.write(f"25th percentile: {quartiles[0]:.4f}\n")
            f.write(f"Median: {quartiles[1]:.4f}\n")
            f.write(f"75th percentile: {quartiles[2]:.4f}\n")


def get_context_text(tokenizer, target_token_length):
    """Generate padding tokens as context with specified length"""
    try:
        # Generate specified number of padding tokens
        padding_tokens = [tokenizer.pad_token_id] * target_token_length
        
        # Decode back to text
        context = tokenizer.decode(padding_tokens, skip_special_tokens=False)
        
        # Verify token length
        final_tokens = tokenizer.encode(context, add_special_tokens=False)
        actual_length = len(final_tokens)
        print(f"Context token length: {actual_length}")
        
        # Adjust padding tokens if length doesn't match
        if actual_length != target_token_length:
            # Retry until correct length is achieved
            while actual_length != target_token_length:
                if actual_length < target_token_length:
                    padding_tokens.append(tokenizer.pad_token_id)
                else:
                    padding_tokens.pop()
                context = tokenizer.decode(padding_tokens, skip_special_tokens=False)
                final_tokens = tokenizer.encode(context, add_special_tokens=False)
                actual_length = len(final_tokens)
        
        return context
    except Exception as e:
        print(f"Error creating padding context: {e}")
        return ""

def parse_args_from_config(config_path=None, config_dict=None):
    """Parse arguments from config file or config dictionary"""
    if config_dict is not None:
        config = config_dict
    elif config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG

    # Flatten nested configuration
    flattened_config = {}

    # Process model configuration
    if 'model' in config:
        flattened_config['model_path'] = config['model'].get('path', DEFAULT_CONFIG['model_path'])
        flattened_config['model_type'] = config['model'].get('type', DEFAULT_CONFIG['model_type'])
        flattened_config['use_8bit'] = config['model'].get('use_8bit', DEFAULT_CONFIG['use_8bit'])
        
    # Process dataset configuration
    if 'dataset' in config:
        flattened_config['dataset_name'] = config['dataset'].get('name', DEFAULT_CONFIG['dataset_name'])
        flattened_config['data_path'] = config['dataset'].get('path', DEFAULT_CONFIG['data_path'])
        flattened_config['num_samples'] = config['dataset'].get('num_samples', DEFAULT_CONFIG['num_samples'])

    # Process evaluation configuration
    if 'evaluation' in config:
        flattened_config['rate'] = config['evaluation'].get('rate', DEFAULT_CONFIG['rate'])
        flattened_config['rate_min'] = config['evaluation'].get('rate_min', DEFAULT_CONFIG['rate_min'])
        flattened_config['rate_max'] = config['evaluation'].get('rate_max', DEFAULT_CONFIG['rate_max'])
        flattened_config['rate_step'] = config['evaluation'].get('rate_step', DEFAULT_CONFIG['rate_step'])
        flattened_config['few_shot_number'] = config['evaluation'].get('few_shot_number', DEFAULT_CONFIG['few_shot_number'])
        flattened_config['verbose'] = config['evaluation'].get('verbose', DEFAULT_CONFIG['verbose'])

    # Process attention configuration
    if 'attention' in config:
        flattened_config['heads'] = config['attention'].get('heads', DEFAULT_CONFIG['heads'])
        flattened_config['layers'] = config['attention'].get('layers', DEFAULT_CONFIG['layers'])
        flattened_config['exploring_mode'] = config['attention'].get('exploring_mode', DEFAULT_CONFIG['exploring_mode'])

    # Process output configuration
    if 'output' in config:
        # Support both dir and output_dir configuration
        output_dir = config['output'].get('dir') or config['output'].get('output_dir', DEFAULT_CONFIG['output_dir'])
        flattened_config['output_dir'] = output_dir

    config = flattened_config

    # Create argument object
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    # Convert configuration to argument object
    args = Args(**config)

    # Handle special parameters
    if hasattr(args, 'heads') and args.heads == 'None':
        args.heads = None
    elif hasattr(args, 'heads') and isinstance(args.heads, str):
        args.heads = [int(x) for x in args.heads.split(',')]
        
    if hasattr(args, 'layers') and args.layers == 'None':
        args.layers = None 
    elif hasattr(args, 'layers') and isinstance(args.layers, str):
        args.layers = [int(x) for x in args.layers.split(',')]
    
    return args

def ZeroTuning(config=None, config_path=None, hftoken=None, model=None, tokenizer=None, **kwargs):
    """ZeroTuning function for model evaluation and exploration.
    
    This function can be used both as a standalone script and as an importable interface.
    It supports running evaluations with either a configuration object, configuration file, or direct parameter passing.
    
    Args:
        config (dict, optional): Configuration dictionary. If provided, will be used instead of config_path.
        config_path (str, optional): Path to the configuration file. If None and config is None, uses default config.
        hftoken (str, optional): HuggingFace access token for model loading.
        model (optional): Pre-loaded model. If provided, will use this model instead of loading a new one.
        tokenizer (optional): Pre-loaded tokenizer. If provided, will use this tokenizer instead of loading a new one.
        **kwargs: Additional parameters that will override config values.
            Supported parameters:
            - model_path: Path to the model
            - model_type: Type of model (llama or qwen)
            - dataset_name: Name of the dataset to evaluate
            - data_path: Path to the dataset
            - num_samples: Number of samples to evaluate
            - rate: Attention enhancement rate
            - rate_min: Minimum rate for interval testing
            - rate_max: Maximum rate for interval testing
            - rate_step: Step size for rate interval testing
            - few_shot_number: Number of few-shot examples
            - verbose: Whether to print detailed information
            - heads: Target attention heads
            - layers: Target model layers
            - exploring_mode: Mode for layer-head exploration
            - output_dir: Directory for saving results
            - use_8bit: Whether to use 8-bit quantization
    
    Returns:
        float: Accuracy score for single evaluation
        tuple: (best_rate, best_accuracy) for rate interval testing
        list: Results for exploration modes
        None: If running as a script
    """
    # Parse arguments from config file or config dict
    if config is not None:
        args = parse_args_from_config(config_dict=config)
    else:
        args = parse_args_from_config(config_path=config_path)

    # Update with directly passed parameters
    for key, value in kwargs.items():
        setattr(args, key, value)

    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Load model and tokenizer if not provided
    if model is None or tokenizer is None:
        model, tokenizer = load_model(
            args.model_path, 
            args.model_type, 
            hftoken,
            use_8bit=args.use_8bit
        )
    
    # Load dataset
    dataset = load_mc_dataset(args.data_path, args.dataset_name, num_samples=args.num_samples)
    
    if args.rate_min is not None and args.rate_max is not None:
        # Rate interval testing
        print(f"Running rate interval testing from {args.rate_min} to {args.rate_max} with step {args.rate_step}")
        
        # Round rate values to 2 decimal places
        rate_min = round(args.rate_min, 2)
        rate_max = round(args.rate_max, 2)
        rate_step = round(args.rate_step, 2)
        
        # Generate rate sequence using numpy
        rates = np.arange(rate_min, rate_max + rate_step/2, rate_step)
        rates = [round(rate, 2) for rate in rates]
        
        results = []
        
        # Create results file
        timestamp = time_module.strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join("results", f"rate_interval_{args.dataset_name}_{timestamp}.txt")
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        
        # Get fixed few-shot examples if needed
        if args.few_shot_number > 0:
            fixed_few_shot_prompt = get_few_shot_examples(dataset, args.dataset_name, args.few_shot_number)
            print(f"\nFixed few-shot examples:\n{fixed_few_shot_prompt}")
        else:
            fixed_few_shot_prompt = ""
        
        # Write configuration information
        with open(result_file, 'w') as f:
            f.write(f"=== Rate Interval Testing Configuration ===\n")
            f.write(f"Dataset: {args.dataset_name}\n")
            f.write(f"Rate range: {rate_min:.2f} to {rate_max:.2f} (step: {rate_step:.2f})\n")
            f.write(f"Target heads: {args.heads}\n")
            f.write(f"Target layers: {args.layers}\n")
            f.write(f"Few-shot number: {args.few_shot_number}\n")
            if args.few_shot_number > 0:
                f.write(f"Fixed few-shot examples:\n{fixed_few_shot_prompt}\n")
            f.write(f"Time: {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            f.write("=== Results ===\n")
            f.write("Rate\tAccuracy\n")
        
        # Test each rate
        for rate in tqdm(rates, desc="Testing rates"):
            accuracy = evaluate_mc(
                model, tokenizer, dataset, args.dataset_name,
                rate=rate,
                few_shot_number=args.few_shot_number,
                verbose=args.verbose,
                output_path=args.output_dir,
                target_layers=args.layers,
                target_heads=args.heads,
                num_samples=args.num_samples,
                fixed_few_shot_prompt=fixed_few_shot_prompt
            )
            results.append((rate, accuracy))
            
            # Write current result
            with open(result_file, 'a') as f:
                f.write(f"{rate:.2f}\t{accuracy:.4f}\n")
        
        # Find best rate
        best_rate, best_accuracy = max(results, key=lambda x: x[1])
        
        # Write summary
        with open(result_file, 'a') as f:
            f.write("\n=== Summary ===\n")
            f.write(f"Best rate: {best_rate:.2f}\n")
            f.write(f"Best accuracy: {best_accuracy:.4f}\n")
            
        print(f"\nRate interval testing results saved to: {result_file}")
        print(f"Best rate: {best_rate:.2f}")
        print(f"Best accuracy: {best_accuracy:.4f}")
        
        return best_rate, best_accuracy
        
    elif args.exploring_mode is not None:
        # Exploration mode
        timestamp = time_module.strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join("results", f"exploration_{args.dataset_name}_{timestamp}.txt")
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        
        if args.exploring_mode == 0:
            # Test all layer-head combinations
            results = explore_all_combinations(
                model, tokenizer, dataset, args.dataset_name,
                rate=args.rate, args=args
            )
        elif args.exploring_mode == 1:
            # Test average performance of each head across specified layers
            results = explore_heads_performance(
                model, tokenizer, dataset, args.dataset_name,
                rate=args.rate, args=args
            )
        else:  # args.exploring_mode == 2
            # Test average performance of each layer across specified heads
            results = explore_layers_performance(
                model, tokenizer, dataset, args.dataset_name,
                rate=args.rate, args=args
            )
            
        # Save results
        save_exploration_results(results, result_file, args)
        return results
        
    else:
        # Single evaluation run
        print("Running single evaluation...")
        
        # Generate fixed few-shot examples if needed
        fixed_few_shot_prompt = ""
        if args.few_shot_number > 0:
            fixed_few_shot_prompt = get_few_shot_examples(dataset, args.dataset_name, args.few_shot_number)
            print(f"\nFixed few-shot examples:\n{fixed_few_shot_prompt}")
        
        accuracy = evaluate_mc(
            model, tokenizer, dataset, args.dataset_name,
            rate=args.rate,
            few_shot_number=args.few_shot_number,
            verbose=args.verbose,
            output_path=args.output_dir,
            target_layers=args.layers,
            target_heads=args.heads,
            num_samples=args.num_samples,
            fixed_few_shot_prompt=fixed_few_shot_prompt
        )
        return accuracy

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Model evaluation and exploration script")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--hftoken", type=str, help="HuggingFace access token")
    args = parser.parse_args()
    ZeroTuning(config_path=args.config, hftoken=args.hftoken)
