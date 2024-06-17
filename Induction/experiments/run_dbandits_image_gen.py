import json
import torch
import numpy as np
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from automatic_prompt_engineer import llm
from transformers import AutoTokenizer, pipeline
from transformers import AutoTokenizer, AutoModel
from LlamaForMLPRegression import DoubleTS, LinearDBDiag, NeuralDBDiag
from automatic_prompt_engineer import config

import argparse
from experiments.evaluation.instruction_induction.utility import set_all_seed
import datetime
import torch.nn.functional as F
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO
import hashlib, base64
from tqdm import tqdm
from numpyencoder import NumpyEncoder
from transformers import CLIPProcessor, CLIPModel

SMOKE_TEST = os.environ.get("SMOKE_TEST")
tkwargs = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_sen_embedding(model, tokenizer, sentences):
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    return sentence_embeddings

def gpt_image_response(image_path, prompt):

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    with open('key', 'r') as f:
        api_key = f.readline()
        api_key = api_key.split('=')[1].strip('\n')
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4-turbo-2024-04-09",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 200
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

class LMForwardAPI:
    def __init__(self, conf, base_conf):

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
        self.cosine_scores = None

    def repharse_prompts(self, prompt, n_domain):
        ini_prompts_his = {}
        print(self.conf['evaluation']['model'])
        model = llm.model_from_config(self.conf['evaluation']['model'])
        # print(f"Prompt: {prompt}")
        repharse_prompt = "Rephrase the following description: {} \n The rephrased description is: ".format(prompt)
        repharse_ratio = 0.02
        batch_size = 10
        if repharse_ratio != 0:
            ini_prompts_his_ = {}
            while len(ini_prompts_his_) < int(repharse_ratio * n_domain):
                model_outputs = model.generate_text([repharse_prompt for tmp in range(batch_size)], 1, 1, use_seed=False)
                for model_output_ in model_outputs:
                    ini_prompts_his_[model_output_] = 0
            ini_prompts_his = {prompt_: 0 for prompt_ in list(ini_prompts_his_.keys())[:int(repharse_ratio * n_domain)]}
        else:
            ini_prompts_his = {}
        batch_size = 50
        while len(ini_prompts_his) < n_domain:
            all_sentences = prompt.split('.')
            # pick a subset of sentence in the above, every sentence will be picked with a prob of 0.5, and idenpendently
            all_query_prompt = []
            for _ in range(batch_size):
                random_pick_sentences = [np.random.choice(all_sentences)]
                random_pick_sentences += [tmp for tmp in all_sentences if np.random.binomial(1, 0.3) == 1]
                prompt_ = '.'.join(random_pick_sentences)
                # modify_prompt = "Modify the following description by changing the objects included and their corresponding descriptions: {} \n The modified description is: ".format(prompt_) # 5 sentence -> use 1-2 sentence to get new ones, change the object, alter the high level semantics 
                modify_prompt = "Modify the following description by rephrasing and changing some information: {} \n The modified description is: ".format(prompt_) # 5 sentence -> use 1-2 sentence to get new ones, change the object, alter the high level semantics 
                all_query_prompt += [modify_prompt]
            model_outputs = model.generate_text(all_query_prompt, 1, 1, use_seed=False)
            for model_output_ in model_outputs:
                ini_prompts_his[model_output_] = 0
        return list(ini_prompts_his.keys())[:n_domain]
    
    def eval(self, image_id):
        return self.cosine_scores[image_id].item()


def image_gen(prompt, number):
    client = OpenAI()

    response = client.images.generate(
    model="dall-e-3",
    prompt=prompt,
    size="1024x1024",
    quality="standard",
    n=number,
    )
    image_url = [tmp.url for tmp in response.data]
    all_images = []
    for i in range(len(image_url)):
        response = requests.get(image_url[i])
        img = Image.open(BytesIO(response.content))
        all_images.append(img)
    return all_images


def run(nu, lamdba, n_init, n_domain, total_iter, local_training_iter, gpt, args):
    base_conf = '../experiments/configs/instruction_induction.yaml'
    conf = {
        'evaluation': {
            'model': {
                'gpt_config': {
                    'model': gpt
                }
            }
        }
    }    
    # load the image comparison model
    pipe = pipeline(task="image-feature-extraction", model_name="google/vit-base-patch16-384", device='cuda:0', pool=True)
    img_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    img_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    if args.image_path != '':
        image_md5 = hashlib.shake_256(args.image_path.encode('utf-8')).hexdigest(5)
        if os.path.exists(f'./query/image-prompt/{image_md5}'):
            with open(f'./query/image-prompt/{image_md5}', 'r') as f:
                init_prompt = f.readline()
        else:
            # create the folder if it does not exist
            if not os.path.exists(f"./query/image-prompt"):
                os.mkdir(f"./query/image-prompt")
            with open(f'./query/image-prompt/{image_md5}', 'x') as f:
                init_prompt = gpt_image_response(args.image_path, "Please provide a detailed description of this image with a few short sentences which each sentence describing a single object.")
                f.write(init_prompt)    
    else:
        scene_prompt = {'garden': "In a vibrant garden, a grand marble fountain gushes clear water, dazzling in the sunlight. Nearby, a centuries-old oak tree stands with sprawling, gnarled branches. A vintage wrought iron bench with floral patterns offers a quaint seat. Beside the path, a whimsical, brightly painted gnome statue holds a fishing rod towards a small pond. In the pond, lily pads float with blooming white lilies.",
                        'street': "On a lively city street, a striking vintage red telephone booth pops against the muted city colors. Nearby, a vibrant graffiti mural adds color to a plain brick wall, featuring an abstract mix of urban elements. A futuristic bicycle with a shiny, aerodynamic silver frame is locked to a lamppost. A small vendor's stall on the sidewalk displays handmade, colorful beaded jewelry, glistening in the afternoon sun. In the background, an ornate old-fashioned street lamp emits a warm glow as dusk approaches.",
                        'cafe': "In a quaint cafe corner, a vintage espresso machine with polished brass fixtures and a matte black body gleams under an antique lamp. A rustic wooden bookshelf, brimming with well-worn books, stands against a distressed cream wall. A marble table at the room's center holds a delicate porcelain teapot with intricate blue flowers, from which steam gently rises. Beside the table, a colorful glass mosaic cat sculpture perches on a mismatched velvet chair, casting playful reflections around.",
                        'sports': "A sleek grand piano with a glossy black surface speckled with white spots stands at the room's center. On the wall, a colorful clock features a face marked by vibrant, multicolored spots for each hour. Beside it, a tall floor lamp sports a leopard-spot patterned lampshade in black and gold. A plush armchair in the corner showcases bold red polka dots on a white background. On a nearby table, a delicate glass vase captivates with swirling, iridescent spots that shimmer in the light."}
        init_prompt = scene_prompt[args.init_prompt]
    
    # load the eval and API module
    model_forward_api = LMForwardAPI(conf=conf, base_conf=base_conf)

    # check whether a certain file exists
    md5=hashlib.shake_256(init_prompt.encode('utf-8')).hexdigest(5)
    print(set_all_seed(args.trial))
    if os.path.exists(f"./query/image-gen-{md5}/prompts.json"):
        with open(f"./query/image-gen-{md5}/prompts.json", 'r') as fp:
            data = json.load(fp)
        rephrased_prompts = data['rephrased-prompt']
        all_images = []
        for i in range(n_domain):
            all_images.append(Image.open(f"./query/image-gen-{md5}/{i}.png"))
        gt_image = Image.open(f"./query/image-gen-{md5}/gt.png")
    else:
        # create the folder if it does not exist
        if not os.path.exists(f"./query/image-gen-{md5}"):
            os.mkdir(f"./query/image-gen-{md5}")
        data = {'gt-prompt': init_prompt}

        if args.image_path != '':
            gt_image_prompt = gpt_image_response(args.image_path, "Please provide a detailed description of this image with a few short sentences.")
        else:
            gt_image_prompt = init_prompt
        data['gt_image_prompt'] = gt_image_prompt
        # generating the ground-true image using this prompt
        gt_image = image_gen(gt_image_prompt, 1)[0]

        # rephrasing the prompt to generate domains
        rephrased_prompts = model_forward_api.repharse_prompts(init_prompt, (n_domain+20))

        # use the rephrased prompts to generate the images
        all_images = []
        rephrased_prompts_ = []
        for i in tqdm(range(len(rephrased_prompts)), desc="Generating images"):
            try:
                img_ = image_gen(rephrased_prompts[i], 1)[0]
                all_images += [img_]
                rephrased_prompts_ += [rephrased_prompts[i]]
            except Exception as e:
                print(f"Error: {e}")
                pass
            if len(rephrased_prompts_) == n_domain:
                break

        data['rephrased-prompt'] = rephrased_prompts_
        rephrased_prompts = rephrased_prompts_
        
        # store the images
        for i in range(len(all_images)):
            all_images[i].save(f"./query/image-gen-{md5}/{i}.png")
        gt_image.save(f"./query/image-gen-{md5}/gt.png")

        with open(f"./query/image-gen-{md5}/prompts.json", 'x') as fp:
            json.dump(data, fp, indent=4)
        print('Finished preparing the images and prompts.')

    with torch.no_grad():
        gt_image_embed = img_model.get_image_features(**img_processor(images=gt_image, return_tensors="pt"))
        all_images_embed = torch.vstack([img_model.get_image_features(**img_processor(images=all_images[i], return_tensors="pt")) for i in range(n_domain)])
    
    cosine_scores = torch.tensor([torch.cosine_similarity(gt_image_embed, all_images_embed[tmp]).item() for tmp in range(n_domain)])    
    # standardize the cosine scores
    if args.norm_method == 'standard':
        cosine_scores = args.magnitude * (cosine_scores - cosine_scores.mean()) / cosine_scores.std()
    elif args.norm_method == 'minmax':
        cosine_scores = args.magnitude * (cosine_scores - cosine_scores.min()) / (cosine_scores.max() - cosine_scores.min())
    model_forward_api.cosine_scores = cosine_scores

    # obtain the prompt embeddings
    sen_embeddings = torch.tensor(pipe(all_images))
    sen_embeddings = torch.squeeze(sen_embeddings, dim=1)
    sen_embeddings = sen_embeddings.to(**tkwargs)
    
    # select the first m pairs for the first m rounds
    X_train = []
    y_train = []
    select_idx_history = []
    instruction_select_history = []
    for i in range(n_init):
        sen_1_id, sen_2_id = np.random.choice(args.n_domain, 2, replace=False)
        score_1 = model_forward_api.eval(sen_1_id)
        score_2 = model_forward_api.eval(sen_2_id)
        instruction_select_history += [(rephrased_prompts[sen_1_id], score_1, rephrased_prompts[sen_2_id], score_2)]
        p_ = 1/(1 + np.exp(-(score_1 - score_2)))
        y_ = np.random.binomial(1, p_)
        X_train += [torch.cat([sen_embeddings[sen_1_id].reshape(1,1,-1), sen_embeddings[sen_2_id].reshape(1,1,-1)])]
        y_train += [y_] 
        select_idx_history += [[sen_1_id, sen_2_id]]


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
    now_values = []
    now_images = []
    best_values = []
    best_instruction_over_iter = []
    best_images = []
    for t in range(max_iter):
        print("Start selecting...")
        if args.func == 'random':
            arm_select1, arm_select2 = np.random.choice(args.n_domain, 2, replace=False)
        else:
            arm_select1, arm_select2 = l.select(sen_embeddings, select_idx_history)
            arm_select1, arm_select2 = arm_select1.item(), arm_select2.item()
        select_idx_history += [[arm_select1, arm_select2]]
        score_1 = model_forward_api.eval(arm_select1)
        score_2 = model_forward_api.eval(arm_select2)
        instruction_select_history += [(rephrased_prompts[arm_select1], score_1, rephrased_prompts[arm_select2], score_2)]
        p_ = 1/(1 + np.exp(-(score_1 - score_2)))
        y_ = np.random.binomial(1, p_)

        if args.func == 'random':
            best_arm = arm_select1 if y_ == 1 else arm_select2
        else:
            best_arm = l.find_best(sen_embeddings, select_idx_history).item()
        
        r = model_forward_api.eval(best_arm)
        now_values += [r]
        now_images += [best_arm]
        best_instruction_over_iter += [(t, rephrased_prompts[best_arm], r)]

        if r > best_r:
            best_images_over_iter = best_arm
            best_r = r
        print("Start training...")
        new_x_ = torch.cat([sen_embeddings[arm_select1].reshape(1,1,-1), sen_embeddings[arm_select2].reshape(1,1,-1)]).to(**tkwargs)
        if args.func != 'random':
            l.train(new_x_, y_, local_training_iter)

        print("Selected arm: ", arm_select1, arm_select2)
        print("iter {0} --- reward: {1}".format(t, r))
        print(f"Best value found till now: {best_r}")
        best_values.append(best_r)
        best_images.append(best_images_over_iter)

    return best_values, now_values, best_images, now_images, best_instruction_over_iter, instruction_select_history, select_idx_history, md5

def parse_args():
    parser = argparse.ArgumentParser(description="Image gen pipeline")
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
        default=500,
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
        "--image_path",
        type=str,
        default='',
        help="The path to the original image."
    )
    parser.add_argument(
        "--init_prompt",
        type=str,
        default='',
        help="The initial prompt."
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(set_all_seed(0))
    best_values, now_values, best_images, now_images, best_instruction_over_iter, instruction_select_history, select_idx_history, md5 = run(
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
    args_dict['final_best_values'] = best_values[-1]
    args_dict['best_values'] = best_values
    args_dict['now_values'] = now_values
    args_dict['best_images'] = best_images
    args_dict['now_images'] = now_images
    args_dict['best_instruction_over_iter'] = best_instruction_over_iter
    args_dict['instruction_select_history'] = instruction_select_history
    args_dict['select_idx_history'] = select_idx_history
    args_dict['md5'] = md5

    save_dir = "./results/" + args.name
    
    # if no folder create one
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # get a path with the current time
    path = os.path.join(save_dir, 'image-gen' + datetime.datetime.now().strftime("-%Y-%m-%d_%H-%M-%S") + "_trial{}".format(args.trial) +".json")

    print(args_dict)
    with open(path, 'x') as fp:
        json.dump(args_dict, fp, indent=4, ensure_ascii=False, cls=NumpyEncoder)
    
    print("Finished!!!")
    print(f'Finial similarity score: {best_values[-1]}')


