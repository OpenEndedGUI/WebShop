"""
Test the site gym environment.

TODO: move to testing dir for more rigorous tests
"""
import gym
from rich import print
from rich.markup import escape

import sys
import json
import random
from os.path import join, dirname, abspath
from collections import defaultdict

MODEL_PATH = dirname(abspath(__file__))
SITE_PATH = join(MODEL_PATH, '../')
sys.path.insert(0, SITE_PATH)

from web_agent_site.envs import WebAgentImageEnv
from web_agent_site.models import (
    HumanPolicy,
    RandomPolicy,
)
from web_agent_site.utils import DEBUG_PROD_SIZE
from train_choice_il import *
from transformers import BartForConditionalGeneration, BartTokenizer
from models.vlnbert import VisualBertModelForWebshop, BertConfigForWebshop

bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

import logging
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

def bart_predict(input, model, skip_special_tokens=True, **kwargs):
    input_ids = bart_tokenizer(input)['input_ids']
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    output = model.generate(input_ids, max_length=512, **kwargs)
    return bart_tokenizer.batch_decode(output.tolist(), skip_special_tokens=skip_special_tokens)

def predict(obs, info, model, softmax=False, rule=False, bart_model=None):
    valid_acts = info['valid']
    instruction_obs = obs['instruction_text']
    image_obs = obs['image_feat']
    text_obs = obs['text_data']
    if valid_acts[0].startswith('search['):
        if bart_model is None:
            return valid_acts[-1]
        else:
            goal = process_goal(instruction_obs)
            query = bart_predict(goal, bart_model, num_return_sequences=5, num_beams=5)
            # query = random.choice(query)  # in the paper, we sample from the top-5 generated results.
            query = query[0]  #... but use the top-1 generated search will lead to better results than the paper results.
            return f'search[{query}]'
    try:        
        if rule:
            item_acts = [act for act in valid_acts if act.startswith('click[item - ')]
            if item_acts:
                return item_acts[0]
            else:
                assert 'click[buy now]' in valid_acts
                return 'click[buy now]'
    except:
        print('Valid actions:',valid_acts)
        assert False
                
    #text_obs, image_obs = obs
    
    state_encodings = tokenizer(process(text_obs), max_length=512, truncation=True, padding='max_length')
    action_encodings = tokenizer(list(map(process, valid_acts)), max_length=512, truncation=True,  padding='max_length')
    batch = {
        'state_input_ids': state_encodings['input_ids'],
        'state_attention_mask': state_encodings['attention_mask'],
        'action_input_ids': action_encodings['input_ids'],
        'action_attention_mask': action_encodings['attention_mask'],
        'sizes': len(valid_acts),
        'images': image_obs,
        'labels': 0
    }
    batch = data_collator([batch])
    # make batch cuda
    batch = {k: v.cuda() for k, v in batch.items()}
    #outputs = model(state_input_ids=state_encodings['input_ids'], state_attention_mask=state_encodings['attention_mask'], action_input_ids=action_encodings['input_ids'], action_attention_mask= action_encodings['attention_mask'], sizes=len(valid_acts), images=image_obs, labels=None)
    outputs = model(**batch)
    if softmax:
        idx = torch.multinomial(F.softmax(outputs.logits[0], dim=0), 1)[0].item()
    else:
        idx = outputs.logits[0].argmax(0).item()
    return valid_acts[idx]

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--model_path", type=str, default="./ckpts/web_click/epoch_9/model.pth", help="Where to store the final model.")
    parser.add_argument("--mem", type=int, default=0, help="State with memory")
    parser.add_argument("--bart_path", type=str, default='./ckpts/web_search/checkpoint-800', help="BART model path if using it")
    parser.add_argument("--bart", type=bool, default=True, help="Flag to specify whether to use bart or not (default: True)")
    parser.add_argument("--image", type=bool, default=True, help="Flag to specify whether to use image or not (default: True)")
    parser.add_argument("--softmax", type=bool, default=True, help="Flag to specify whether to use softmax sampling or not (default: True)")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    #env = gym.make('WebAgentSite-v0')
    #env = WebAgentSiteEnv(render=True, pause=2.0)
    #env = WebAgentSiteEnv(observation_mode='html', render=False)
    #env = WebAgentImageEnv(observation_mode='image', render=False, num_products=DEBUG_PROD_SIZE)
    from env import WebEnvImageOnly 
    from train_rl import parse_args as webenv_args
    env_args = webenv_args()[0]
    env_args.state_format = 'image'
    env = WebEnvImageOnly(env_args, split='test')

    args = parse_args()
    print(args)

    bart_model = BartForConditionalGeneration.from_pretrained(args.bart_path)
    print('bart model loaded', args.bart_path)
    config = BertConfigForWebshop(image=args.image)
    model = VisualBertModelForWebshop(config)
    model.cuda()
    print('bert model loaded', args.model_path)
    
    
    try:
        #policy = HumanPolicy()
        policy = RandomPolicy()
        info={}
        #observation = env.env.observation
        total_episodes = 50
        total_reward = 0
        total_harsh_reward = 0
        for _ in range(total_episodes):
            global_step = 0
            obs, info = env.reset(global_step)
            while True:
                #print(observation)
                # available_actions = env.get_available_actions()
                # print('Available actions:', available_actions)
                # action = policy.forward(observation, available_actions)
                # info['valid'] = env.get_valid_actions()
                action = predict(obs, info, model, softmax=args.softmax, rule=False, bart_model=bart_model)
                # print("Action:",action)
                obs, reward, done, info = env.step(action)
                print(f'Taking action "{escape(action)}"')
                if done or global_step > 100:
                    print(f'Reward = {reward}')
                    total_reward += reward
                    if reward == 10:
                        total_harsh_reward += 10
                    break
                global_step += 1
        print("Avg Reward:", total_reward/total_episodes)
        print("Avg Harsh Reward:", total_harsh_reward/total_episodes)
    finally:
        env.close()
