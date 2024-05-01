from model import Foundation
from dataset import TextDataset
import torch
import tqdm
from tokenizers import Tokenizer
from torch.profiler import profile, ProfilerActivity
from ignite.metrics.nlp import Bleu, Rouge
from torch.utils.tensorboard import SummaryWriter
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
import os
import torch._dynamo


os.environ["TOKENIZERS_PARALLELISM"] = 'false'

def init():
    print(f"Using PyTorch version {torch.__version__}")
    
    #use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    
    #use tensor cores
    torch.set_float32_matmul_precision('high')
    
    #use flash attention
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    
    return device



def collate_fn(batch):
    x = [s[0] for s in batch]
    y = [s[1] for s in batch]
    article = [s[2] for s in batch]
    x = torch.stack(x)
    y = torch.stack(y)
    article = torch.stack(article)
    return x, y, article

def create_dataloader(params, split):
    dataset = TextDataset(max_context_length_abstract = params['max_context_length_abstract'],
                        max_context_lenght_article = params['max_context_length_article'],
                        split = split)
    if split != 'test':
        dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size = params['batch_size'], 
                                            shuffle = True, 
                                            num_workers = params['num_workers'],
                                            collate_fn = collate_fn)
    else:    
        dataloader = dataset
    return dataloader

def train(params, model):
    train_dataloader = create_dataloader(params, split = "train")
    model.train()
    optimizer = model.config_optimizer()
    scaler = torch.cuda.amp.GradScaler() #scaler needed for mixed precision training
    
    early_stop_check=0
    prev_bleu_score=-10**10

    for epoch in range(params['epochs']):

        progress_bar = tqdm.tqdm(train_dataloader, desc = f"Epoch {epoch}/{params['epochs']}")
        for i, data in enumerate(progress_bar):
            x, y, article = data
            x, y, article = x.to(device), y.to(device), article.to(device)
            
            #train an iteration
            with torch.amp.autocast(device_type = "cuda", dtype = torch.float16): #use mixed precision training
                loss, acc = model(x = x, article = article, y = y, return_loss = True)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none = True)
            scaler.update()
            
            #save the model every 1000 iterations
            if i % params["save_freq"] == 0:
                print(f"Saving model")
                torch.save(model, params['experiment_path'] + f"/Experiment{params['experiment_id']}_model.pt")
                
                #validation and early stopping
                validation(params,model)                
                if(prev_bleu_score<params['bleu_score'][-1] and early_stop_check<params['early_stop_sensitivity']): #sensitivity in iterations not epochs
                    early_stop_check=0
                else:
                    early_stop_check+=1

            #save the loss and accuracy    
            progress_bar.postfix = f"Loss: {loss.item()}, acc: {acc.item()}"

def validation(params, model):
    model.eval()
    val_dataloader = create_dataloader(params, split = "validation")
    tokenizer = Tokenizer.from_file("src/notebooks/tokenizer.json")
    progress_bar = tqdm.tqdm(val_dataloader, desc = "Validating")
    bleu = Bleu(ngram = 4, smooth = "smooth1")
    rouge = Rouge()

    with torch.inference_mode():
        for data in progress_bar:
            x, y, article = data
            x, y, article = x.to(device), y.numpy().tolist(), article.to(device)

            #get the prediction
            with torch.amp.autocast(device_type = "cuda", dtype = torch.float16):
                y_pred = model(x = x, article = article, return_loss = False)
            y_pred = y_pred.argmax(dim = -1).cpu().numpy().tolist()         

            for i in range(len(y)):
                for j in range(len(y), 0, -1):
                    if y[i][j] != 4:
                        y_pred[i] = y_pred[i][:j]
                        y[i] = y[i][:j]
                        break
   
            y_pred, y = tokenizer.decode_batch(y_pred), tokenizer.decode_batch(y)
            bleu.update(([y_pred_indv.split() for y_pred_indv in y_pred], [y_indv.split() for y_indv in y]))
            rouge.update(([y_pred_indv.split() for y_pred_indv in y_pred], [y_indv.split() for y_indv in y]))


    bleu_score=bleu.compute().item()
    rouge_score=rouge.compute()['Rouge-L-P']
    print(f"Rouge score: {rouge_score}")
    print(f"Bleu score: {bleu_score}")
    model.train()


def generate_examples(params, model):
    tokenizer = Tokenizer.from_file("src/notebooks/tokenizer.json")
    model.eval()
    dataloader = create_dataloader(params, split = "test")
    progress_bar = tqdm.tqdm(dataloader, desc = "Generating examples")
    start = 0
    with torch.inference_mode():
        for i, data in enumerate(progress_bar):
            x = torch.tensor([start], dtype = torch.int64).to(device)
            _, y, article = data
            y, article = y.numpy().tolist(), article.to(device)
            article = article.unsqueeze(0)
            x = x.unsqueeze(0)
            for _ in range(params['max_context_length_abstract']):
                with torch.inference_mode():
                    with torch.amp.autocast(device_type = "cuda", dtype = torch.float16):
                        y_pred = model.generate(x =  x, article = article)
                    y_pred = y_pred.argmax(dim = -1)
                    if y_pred[0][-1].item() == 1:
                        break
                    x = torch.cat((x[0], y_pred[0][-1].unsqueeze(0)), dim = 0).unsqueeze(0)
            
            with open(f"{params['experiment_path']}"+"/generated_test.txt" ,"a") as txt:
                txt.write(f"Generated example {i}:\n\n\n")
                txt.write(tokenizer.decode_batch(x.cpu().numpy().tolist())[0])
                txt.write("\n\n\n")
                txt.write(f"Actual ground truth {i}:\n\n\n")
                txt.write(tokenizer.decode_batch([y])[0])
                txt.write("\n\n\n")
                txt.close()
            if i == params["num_examples_generate"]:
                break

def run_experiment():
    experiments=pd.read_json("src/Experiments.json")
    for i in range(len(experiments.columns)):
        #experiment parmams and experiment id
        params=experiments[f"experiment{i+1}"]
        params['experiment_id']=i+1

        #create the experiment folder
        params["experiment_path"] = f"logs/Experiment_{params['experiment_id']}"
        metrics_log_path=params["experiment_path"]+"/metrics"
        os.makedirs(params["experiment_path"],exist_ok=True)
        os.makedirs(metrics_log_path,exist_ok=True)

        params['hyper_params']['pad_char']=4
        #create the model
        model = Foundation(params['hyper_params']).to(device)
        print(f"The number of parameters is {model.get_num_params()}")

        #create the loss and accuracy lists
        params["train_loss"]=[]
        params["train_acc"]=[]
        params["bleu_score"]=[]
        params["rouge_score"]=[]
        
        #profile and train model
        train(params, model)
        generate_examples(params, model)
    
        params["max_mem_allocated"]=torch.cuda.max_memory_allocated(device="cuda")

        torch.save(model, params['experiment_path'] + f"/Experiment{params['experiment_id']}_model.pt")
        
        #save the loss and accuracy
        data=pd.DataFrame(data=[params["train_loss"],params['train_acc'], params['bleu_score'], params['rouge_score']])
        data=data.transpose()
        data.to_csv(params["experiment_path"]+'/loss_acc_data.csv')

        plt.plot(data=data)
        plt.savefig(params['experiment_path']+"/loss_acc_curves.png")

        
if __name__ == "__main__":
    device = init()
    run_experiment()