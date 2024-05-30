from model import Translator
from dataset import TextDataset
import torch
import tqdm
from tokenizers import Tokenizer
import os
from nltk.translate.bleu_score import corpus_bleu
os.environ["TOKENIZERS_PARALLELISM"] = 'false'
import pandas as pd

EPOCHS = 10
LEARNING_RATE = 5e-4
DROPOUT = 0.1
VOCAB_SIZE = 1000
CONTEXT_LENGTH = 500
NUM_ENCODER_BLOCKS = 8
NUM_DECODER_BLOCKS = 8
NUM_HEADS = 16
EMBED_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 4

lang=[["gl","en"],
      ["glpt","en"],
      ["pt","en"],
      ['tr','en']]

def collate_fn(batch):
    x = [s[0] for s in batch]
    y = [s[1] for s in batch]
    originalText = [s[2] for s in batch]
    x = torch.stack(x)
    y = torch.stack(y)
    originalText = torch.stack(originalText)
    return x, y, originalText

def eval_collate_fn(batch):
    engText = [s[0] for s in batch]
    hilliText = [s[1] for s in batch]
    return engText, hilliText

print(f"Using PyTorch version {torch.__version__}")

# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# use tensor cores
torch.set_float32_matmul_precision('high')

# use flash attention
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

for cur,target in lang:
    name=cur+"_to_"+target
    path = "dataset/comparisions/"

    bleu_scores=[]
    train_loss = []
    train_acc = []
    
    model = Translator(engVocabSize=VOCAB_SIZE, hilliVocabSize=VOCAB_SIZE, embed_size=EMBED_SIZE,
                    num_encoder_blocks=NUM_ENCODER_BLOCKS, num_decoder_blocks=NUM_DECODER_BLOCKS, num_heads=NUM_HEADS, dropout=DROPOUT, pad_char=2).to(device)
    print(f"The number of parameters is {model.get_num_params()}")
    
    dataset = TextDataset(path = path+cur+"_to_"+target+"train.csv", engContextLength=CONTEXT_LENGTH, hilliContextLength=CONTEXT_LENGTH ,cur=cur,target=target,)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    
    val_dataset = TextDataset(path = path+cur+"_to_"+target+"val.csv", engContextLength=CONTEXT_LENGTH, hilliContextLength=CONTEXT_LENGTH, cur=cur,target=target, isTrain=False)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=eval_collate_fn, num_workers=NUM_WORKERS)

    test_dataset = TextDataset(path = path+cur+"_to_"+target+"test.csv", engContextLength=CONTEXT_LENGTH, hilliContextLength=CONTEXT_LENGTH , cur=cur,target=target, isTrain=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=eval_collate_fn, num_workers=NUM_WORKERS)
    
    model.train()
    optimizer = model.config_optimizer(LEARNING_RATE)

    hilliTokenizer = Tokenizer.from_file(f"models/{cur}_tokeniser.json")
    engTokenizer = Tokenizer.from_file(f"models/{cur}s_{target}_tokeniser.json")

    def generate(batched_sentence):
        batched_sentence = hilliTokenizer.encode(batched_sentence)
        batched_sentence = torch.tensor(batched_sentence, dtype=torch.int64).unsqueeze(0).to(device)
        currentOutput = [0]
        model.eval()
        for i in range(100):
            x = torch.tensor(
                currentOutput, dtype=torch.int64).unsqueeze(0).to(device)
            output = model(x=x, originalText=batched_sentence, return_loss=False)
            output = torch.argmax(output[0][-1]).item()
            currentOutput.append(output)
            if (output == 1):
                break
        currentOutput = engTokenizer.decode(currentOutput)
        model.train()
        return currentOutput

    for epoch in range(EPOCHS):

        progress_bar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
        for i, data in enumerate(progress_bar):
            x, y, originalText = data
            x, y, originalText = x.to(device), y.to(
                device), originalText.to(device)

            loss, acc = model(x=x, originalText=originalText,
                            y=y, return_loss=True)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            progress_bar.postfix = f"Loss: {loss.item()}, acc: {acc.item()}"
            train_loss.append(loss.item())
            train_acc.append(acc.item())
            
        with torch.no_grad():
            #get validation BLEU score
            print("Calculating BLEU score")
            generated = []
            referenced = []
            for data in val_dataloader:
                engText, hilliText = data
                generated_text = generate(hilliText)
                generated.append(generated_text.split(" "))
                referenced.append([engText.split(" ")])
                # print(f"Generated: {generated_text}")
                # print(f"Referenced: {engText}\n\n")
            bleu = corpus_bleu(referenced, generated)
            bleu_scores.append(bleu)
            print(f"BLEU score: {bleu}")
            break
            
        torch.save(model, "models/model.pt")

    #testing the model
    with torch.no_grad():
        #get validation BLEU score
        print("Calculating BLEU score")
        generated = []
        referenced = []
        for data in test_dataloader:
            engText, hilliText = data
            generated_text = generate(hilliText)
            generated.append(generated_text.split(" "))
            referenced.append([engText.split(" ")])
            # print(f"Generated: {generated_text}")
            # print(f"Referenced: {engText}\n\n")
        bleu = corpus_bleu(referenced, generated)
        print(f"BLEU score: {bleu}")
        
    bleu_scores.append(bleu)
    pd.Dataframe(data=bleu_scores).to_csv(f"Bleu scores/{name}_bleu.csv")
    pd.Dataframe(data=train_loss).to_csv(f"Bleu scores/{name}_train_loss.csv")
    pd.DataFrame(data=train_acc).to_csv(f"Bleu scores/{name}_train_acc.csv")