import json
import random
import torch
import numpy as np
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from experiments import llm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModel
from LlamaForMLPRegression import DoubleTS, LinearDBDiag, NeuralDBDiag
from experiments import config
import re

import argparse
from experiments.evaluation.instruction_induction.utility import set_all_seed
import datetime
import torch.nn.functional as F
from datasets import load_dataset


SMOKE_TEST = os.environ.get("SMOKE_TEST")
tkwargs = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}
    
model_name = "vicuna"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
api_model = 'chatgpt'
alpha = 1
sigma = 1

def extract_sub_sentence(long_sentence):
    matches = re.findall('<prompt>(.*?)</prompt>', long_sentence)
    return matches

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_sen_embedding(model, tokenizer, sentences):
    # Tokenize sentences
    # print(sentences)
    # raise NotImplementedError
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    return sentence_embeddings

class LMForwardAPI:
    def __init__(self, conf=None, base_conf=None, magnitude=None, norm_method=None):
        p = torch.ones(10)
        
        kwargs={'torch_dtype': torch.float16}
        self.count = 0
                
        ## eval preparation
        self.conf = config.update_config(conf, base_conf)
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_last_perf = 10
        self.best_prompt = None
        self.num_call = 0
        self.best_instruction = None
        self.prompts_set = dict()
        self.prompts_list = []
        self.parents = []
        self.best_score = 0
        self.tokenizer = AutoTokenizer.from_pretrained('Ray2333/gpt2-large-helpful-reward_model')
        self.model = AutoModelForSequenceClassification.from_pretrained(
                        'Ray2333/gpt2-large-helpful-reward_model',
                        num_labels=1, torch_dtype=torch.bfloat16,
                        device_map=0,
                        )
        self.score_mean = None
        self.score_std = None
        self.score_min = None
        self.score_max = None
        self.magnitude = magnitude
        self.norm_method = norm_method
        
    def initialize_prompts(self, prompt, n_domain):
        ini_prompts_his = {}
        print(self.conf['evaluation']['model'])
        model = llm.model_from_config(self.conf['evaluation']['model'])
        batch_size = 50
        # print(f"Prompt: {prompt}")
        while len(ini_prompts_his) < n_domain:
            model_outputs = model.generate_text([prompt for tmp in range(batch_size)], 1, 1, use_seed=False)
            # if model_outputs[0] not in ini_prompts_his:
            for model_output_ in model_outputs:
                ini_prompts_his[model_output_] = 0
            # print(len(ini_prompts_his))
        return list(ini_prompts_his.keys())[:n_domain]
    
    def selection(self, num_next_gen):
        scores = np.array([self.prompts_set[tmp] for tmp in self.parents])
        num_parents = len(self.parents)
        probability = []
        if np.sum(scores) == 0:
            probability = np.ones(num_parents)/ num_parents
        else:
            probability = scores / np.sum(scores)
        
        all_parents = []
        for i in range(num_next_gen):
            try:
                parent_pair = np.random.choice(self.parents, size=2, replace=False, p=probability)
            except:
                parent_pair = np.random.choice(self.parents, size=2, replace=True, p=probability)
            all_parents += [parent_pair]
        return all_parents
    
    def evolution(self, all_parents):

        next_gens = []
        model = llm.model_from_config(self.conf['evaluation']['model'])
        
        template = "Please follow the instruction step-by-step to generate a better prompt.\n1. Cross over the following prompts and generate a new prompt:\nPrompt 1: [prompt_id1].\nPrompt 2: [prompt_id2].\n2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>."
        for parents_ in all_parents:
            template_ = template.replace('[prompt_id1]', parents_[0])
            template_ = template_.replace('[prompt_id2]', parents_[1])
            model_outputs = model.generate_text(template_, 1, 0)
            model_outputs_ = extract_sub_sentence(model_outputs[0])
            if len(model_outputs_) != 0:
                model_outputs = model_outputs_[0]
                print(f"EVOL: {model_outputs}")
            else:
                model_outputs = model_outputs[0]
            next_gens += [model_outputs]
        return next_gens
    
    def update(self, next_gens):
        next_gens_scores = []
        for gen_ in next_gens:
            score_ = self.eval([gen_])
            next_gens_scores += [score_]
        self.this_iter_best = np.max(next_gens_scores) 
        num_parents = len(self.parents)
        parents_next_gen = self.parents + next_gens
        all_scores = [self.prompts_set[tmp] for tmp in parents_next_gen]
        idx_rank = np.argsort(all_scores)
        selected_idx = idx_rank[-num_parents:]
        new_parents = []
        for idx_ in selected_idx:
            new_parents += [parents_next_gen[idx_]]
        self.parents = new_parents
    
    def eval(self, prompt, response, test=False):
        input_ = prompt + response
        if input_ in self.prompts_set.keys():
            score = self.prompts_set[input_]
        else:
            token_ids = self.tokenizer(prompt, response, return_tensors='pt', truncation=True)
            with torch.no_grad():
                score = self.model(**(token_ids.to(0))).logits[0].cpu().detach().item()
            if not test:
                if self.norm_method == 'standard':
                    score = self.magnitude * (score - self.score_mean) / self.score_std
                elif self.norm_method == 'minmax':
                    score = self.magnitude * (score - self.score_min) / (self.score_max - self.score_min)
                self.prompts_set[input_] = score
        return score

    def return_best_prompt(self):
        return self.best_instruction

    def return_prompts_set(self):
        return self.prompts_set

    def return_prompts_list(self):
        return self.prompts_list

def run(nu, lamdba, n_init, n_domain, total_iter, local_training_iter, gpt, args):
    # assert task in TASKS, 'Task not found!'
    dataset = load_dataset(path="../hh-rlhf", data_dir="helpful-base")

    n_prompts = args.n_prompts
    selected_idx = random.sample(range(len(dataset['train']['chosen'])), n_prompts)
    selected_prompt_ = [dataset['train']['chosen'][idx] for idx in selected_idx]
    selected_prompt = []
    for prompt_ in selected_prompt_:
        matches = re.finditer(r'Assistant: ', prompt_)
        position = list(matches)[-1].start()
        selected_prompt += [prompt_[:(position + 11)]]
    print(set_all_seed(args.trial))
    conf = {
        'evaluation': {
            'model': {
                'gpt_config': {
                    'model': gpt
                }
            }
        }
    }
    base_conf = '../experiments/configs/instruction_induction.yaml'
    model_forward_api = LMForwardAPI(conf=conf, base_conf=base_conf, magnitude=args.magnitude, norm_method=args.norm_method)

    # check whether a certain file exists
    if os.path.exists(f"./query/response_selection.json"):
        with open(f"./query/response_selection.json", 'r') as fp:
            all_responses = json.load(fp)
    else:
        all_responses = {prompt_:[] for prompt_ in selected_prompt}
        for prompt_ in selected_prompt:
            all_responses[prompt_] = model_forward_api.initialize_prompts(prompt_, n_domain)
        # create the folder if it does not exist
        if not os.path.exists("./query"):
            os.mkdir("./query")
        with open(f"./query/response_selection.json", 'x') as fp:
            json.dump(all_responses, fp, indent=4)
    

    sen_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    sen_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    
    # obtaining the embedding for all prompt-response pairs
    all_sentences = []
    sentence_id = {}
    id_sentence = {}
    id_sentence_pair = {}
    prompt_domain_id = {prompt_:[] for prompt_ in all_responses.keys()}
    for i_, prompt_ in enumerate(all_responses.keys()):
        for j_, response_ in enumerate(all_responses[prompt_]):
            all_sentences += [prompt_ + response_]
            sentence_id[prompt_ + response_] = i_ * len(all_responses[prompt_]) + j_
            id_sentence[i_ * len(all_responses[prompt_]) + j_] = prompt_ + response_
            id_sentence_pair[i_ * len(all_responses[prompt_]) + j_] = [prompt_, response_]
            prompt_domain_id[prompt_] += [i_ * len(all_responses[prompt_]) + j_]
    
    sen_embeddings = get_sen_embedding(sen_model, sen_tokenizer, all_sentences)
    sen_embeddings = sen_embeddings.to(**tkwargs)

    # Randomly eval scores to normalize the scores
    test_num = 50
    all_tmp_scores = []
    for tmp in range(test_num):
        prompt_tmp = np.random.choice(selected_prompt)
        score_tmp = model_forward_api.eval(prompt_tmp, all_responses[prompt_tmp][np.random.choice(args.n_domain)], test=True)
        all_tmp_scores += [score_tmp]
    model_forward_api.score_mean = np.mean(all_tmp_scores)
    model_forward_api.score_std = np.std(all_tmp_scores)
    model_forward_api.score_min = np.min(all_tmp_scores)
    model_forward_api.score_max = np.max(all_tmp_scores)
    
    # select the first m pairs for the first m rounds
    X_train = []
    y_train = []
    select_idx_history = {prompt_:[] for prompt_ in all_responses.keys()}
    select_idx_history_global = []
    instruction_select_history = []
    prompt_id = 0
    for i in range(n_init):
        prompt_ = list(all_responses.keys())[prompt_id]
        sen_1_id, sen_2_id = np.random.choice(args.n_domain, 2, replace=False)
        score_1 = model_forward_api.eval(prompt_, all_responses[prompt_][sen_1_id])
        score_2 = model_forward_api.eval(prompt_, all_responses[prompt_][sen_2_id])
        global_sen_1_id = sentence_id[prompt_ + all_responses[prompt_][sen_1_id]]
        global_sen_2_id = sentence_id[prompt_ + all_responses[prompt_][sen_2_id]]
        instruction_select_history += [(prompt_ + all_responses[prompt_][sen_1_id], score_1, prompt_ + all_responses[prompt_][sen_2_id], score_2)]
        p_ = 1/(1 + np.exp(-(score_1 - score_2)/args.noisy_parameter))
        y_ = np.random.binomial(1, p_)
        X_train += [torch.cat([sen_embeddings[global_sen_1_id].reshape(1,1,-1), sen_embeddings[global_sen_2_id].reshape(1,1,-1)])]
        y_train += [y_] 
        select_idx_history[prompt_] += [[global_sen_1_id, global_sen_2_id]]
        select_idx_history_global += [[global_sen_1_id, global_sen_2_id]]
        prompt_id += 1
        if prompt_id >= len(all_responses.keys()):
            prompt_id = 0


    X_train = torch.cat(X_train, dim=1)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int32)
    if args.func == 'neural':
        l = NeuralDBDiag(input_dim=X_train.shape[-1], lamdba=lamdba, nu=nu, init_x=X_train, init_y=y_train, style='ucb', diagonalize=False)
        l.train(local_training_iter=local_training_iter)
    elif args.func == 'linear':
        l = LinearDBDiag(input_dim=X_train.shape[-1], lamdba=lamdba, nu=nu, init_x=X_train, init_y=y_train, style='ucb', diagonalize=False)
        l.train(local_training_iter=local_training_iter)
    elif args.func == 'doublets':
        l = DoubleTS(input_dim=X_train.shape[-1], lamdba=lamdba, nu=nu, init_x=X_train, init_y=y_train, style='ucb', diagonalize=False)
        l.train(local_training_iter=local_training_iter)
    elif args.func == 'random':
        pass

    max_iter = total_iter - n_init
    print("Max iter: ", max_iter)
    best_r = -np.infty
    best_values = []
    now_best_values = []
    
    best_response_all_prompt = {prompt_:None for prompt_ in all_responses.keys()}
    best_r_all_prompt = {prompt_:-np.infty for prompt_ in all_responses.keys()}
    best_avg_r = []
    best_values_all_prompt = {prompt_:[] for prompt_ in all_responses.keys()}
    best_response_over_iter_all_prompt = {prompt_:[] for prompt_ in all_responses.keys()}
    now_response_over_iter_all_prompt = {prompt_:[] for prompt_ in all_responses.keys()}
    now_avg_r = []
    for t in range(max_iter):
        prompt_ = list(all_responses.keys())[prompt_id]
        print("Start selecting...")
        if args.func == 'random':
            arm_select1_, arm_select2_ = np.random.choice(args.n_domain, 2, replace=False)
            arm_select1 = sentence_id[prompt_ + all_responses[prompt_][arm_select1_]]
            arm_select2 = sentence_id[prompt_ + all_responses[prompt_][arm_select2_]]
        else:
            arm_select1, arm_select2 = l.select(sen_embeddings, select_idx_history_global, prompt_domain_id[prompt_])
            arm_select1, arm_select2 = arm_select1.item(), arm_select2.item()
        select_idx_history[prompt_] += [[arm_select1, arm_select2]]
        score_1 = model_forward_api.eval(id_sentence_pair[arm_select1][0], id_sentence_pair[arm_select1][1])
        score_2 = model_forward_api.eval(id_sentence_pair[arm_select2][0], id_sentence_pair[arm_select2][1])
        instruction_select_history += [(id_sentence[arm_select1], score_1, id_sentence[arm_select2], score_2)]
        p_ = 1/(1 + np.exp(-(score_1 - score_2)/args.noisy_parameter))
        y_ = np.random.binomial(1, p_)

        if args.func == 'random':
            best_arm = arm_select1 if y_ == 1 else arm_select2
        else:
            best_arm = l.find_best(sen_embeddings, select_idx_history[prompt_])
            best_arm = best_arm.item()
        
        r = model_forward_api.eval(id_sentence_pair[best_arm][0], id_sentence_pair[best_arm][1])
        now_best_values += [r]
        best_r = max(r, best_r)
        best_values.append(best_r)
        
        # find the best for all prompts
        now_all_r = []
        for prompt_1 in all_responses.keys():
            if args.func == 'random':
                random_arm1 = sentence_id[prompt_1 + all_responses[prompt_1][np.random.choice(args.n_domain)]]
                random_arm2 = sentence_id[prompt_1 + all_responses[prompt_1][np.random.choice(args.n_domain)]]
                score_1_ = model_forward_api.eval(id_sentence_pair[random_arm1][0], id_sentence_pair[random_arm1][1])
                score_2_ = model_forward_api.eval(id_sentence_pair[random_arm2][0], id_sentence_pair[random_arm2][1])
                p_1 = 1/(1 + np.exp(-(score_1_ - score_2_)/args.noisy_parameter))
                y_1 = np.random.binomial(1, p_1)
                best_arm_ = random_arm1 if y_1 == 1 else random_arm2
            else:
                if len(select_idx_history[prompt_1]) ==0:
                    best_arm_ = sentence_id[prompt_1 + all_responses[prompt_1][0]]
                else:
                    best_arm_ = l.find_best(sen_embeddings, select_idx_history[prompt_1]).item()
            r_ = model_forward_api.eval(id_sentence_pair[best_arm_][0], id_sentence_pair[best_arm_][1])
            now_response_over_iter_all_prompt[id_sentence_pair[best_arm_][0]] += [(t, id_sentence_pair[best_arm_][1], r_)]
            now_all_r.append(r_)
            if r_ > best_r_all_prompt[prompt_1]:
                best_r_all_prompt[prompt_1] = r_
                best_response_all_prompt[prompt_1] = id_sentence_pair[best_arm_][1]
            best_values_all_prompt[prompt_1] += [best_r_all_prompt[prompt_1]]
            best_response_over_iter_all_prompt[prompt_1] += [(t, best_response_all_prompt[prompt_1], r_)]
        best_avg_r += [np.mean([best_r_all_prompt[prompt_1] for prompt_1 in all_responses.keys()])]
        now_avg_r += [np.mean(now_all_r)]
        print("Start training...")
        new_x_ = torch.cat([sen_embeddings[arm_select1].reshape(1,1,-1), sen_embeddings[arm_select2].reshape(1,1,-1)]).to(**tkwargs)
        if args.func != 'random':
            l.train(new_x_, y_, local_training_iter)

        print("Selected arm: ", arm_select1, arm_select2)
        print("iter {}".format(t))
        print(f"Best value found till now: {best_avg_r[-1]}")
        
        prompt_id += 1
        if prompt_id >= len(all_responses.keys()):
            prompt_id = 0

    return best_r, best_values, now_best_values, now_avg_r, best_response_all_prompt, best_r_all_prompt, best_avg_r, best_values_all_prompt, best_response_over_iter_all_prompt, now_response_over_iter_all_prompt, select_idx_history, all_responses
    # print(f'Test score on ChatGPT: {test_score}')

def parse_args():
    parser = argparse.ArgumentParser(description="InstructZero pipeline")
    parser.add_argument(
        "--nu",
        type=float,
        default=0.1,
        help="Set the parameter nu."    
    )
    parser.add_argument(
        "--lamdba",
        type=float,
        default=0.1,
        help="Set the lamdba parameter."    
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=40,
        help="Set the number of initialization points."    
    )
    parser.add_argument(
        "--n_domain",
        type=int,
        default=50,
        help="Set the number of domain."    
    )
    parser.add_argument(
        "--total_iter",
        type=int,
        default=165,
        help="Set the number of total queries."    
    )
    parser.add_argument(
        "--local_training_iter",
        type=int,
        default=30,
        help="Set the number of total queries."    
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Set the name of the experiments."    
    )
    parser.add_argument(
        "--gpt",
        type=str,
        default="gpt-3.5-turbo",
        help="Which version of gpt to use."    
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="last",
        help="Which pooling method to use."    
    )
    parser.add_argument(
        "--func",
        type=str,
        default="neural",
        help="Which model to use, can be linear, neural."    
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=0,
        help="Trial ID."
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=10,
        help="The number of prompts."
    )
    parser.add_argument(
        "--magnitude",
        type=int,
        default=10,
        help="The magnitude of the scores."
    )
    parser.add_argument(
        "--norm_method",
        type=str,
        default='standard',
        help="The way to transform the value, standard, minmax."
    )
    parser.add_argument(
        "--noisy_parameter",
        type=float,
        default=1,
        help="Parameter that control the noisy in preferential feedback."
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(set_all_seed(0))
    best_r, best_values, now_best_values, now_avg_r, best_response_all_prompt, best_r_all_prompt, best_avg_r, best_values_all_prompt, best_response_over_iter_all_prompt, now_response_over_iter_all_prompt, select_idx_history, all_responses = run(
        nu=args.nu,
        lamdba=args.lamdba,
        n_init=args.n_init,
        n_domain=args.n_domain,
        total_iter=args.total_iter,
        local_training_iter = args.local_training_iter,
        gpt=args.gpt,
        args=args
    )
    
    args_dict = vars(args)
    args_dict['final_best_avg_r'] = best_avg_r[-1]
    args_dict['best_r'] = best_r
    args_dict['best_values'] = best_values
    args_dict['now_best_values'] = now_best_values
    args_dict['best_response_all_prompt'] = best_response_all_prompt
    args_dict['best_r_all_prompt'] = best_r_all_prompt
    args_dict['best_avg_r'] = best_avg_r
    args_dict['best_values_all_prompt'] = best_values_all_prompt
    args_dict['now_response_over_iter_all_prompt'] = now_response_over_iter_all_prompt
    args_dict['best_response_over_iter_all_prompt'] = best_response_over_iter_all_prompt
    args_dict['select_idx_history'] = select_idx_history
    args_dict['all_responses'] = all_responses
    args_dict['now_avg_r'] = now_avg_r

    
    save_dir = "./results/" + args.name
    
    # if no folder create one
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # get a path with the current time
    path = os.path.join(save_dir, 'response_selection' + datetime.datetime.now().strftime("-%Y-%m-%d_%H-%M-%S") + "_trial{}".format(args.trial) +".json")

    with open(path, 'x') as fp:
        json.dump(args_dict, fp, indent=4)
    
    print("Finished!!!")
    print(f'Best score: {best_avg_r[-1]}')


