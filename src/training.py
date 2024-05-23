from model import Translator
from dataset import TextDataset
import torch
import tqdm
from tokenizers import Tokenizer
import os
from nltk.translate.bleu_score import corpus_bleu
os.environ["TOKENIZERS_PARALLELISM"] = 'false'
import pandas as pd

lang=[["gl","en"],
      ["glpt","en"],
      ["pt","en"],
      ['tr','en']]
for cur,target in lang:
    name=cur+"_to_"+target
    path = "dataset/comparisions/"
    EPOCHS = 10
    LEARNING_RATE = 5e-4
    DROPOUT = 0.1

    bleu_scores=[]
    train_loss = []
    train_acc = []
    def collate_fn(batch):
        x = [s[0] for s in batch]
        y = [s[1] for s in batch]
        originalText = [s[2] for s in batch]
        x = torch.stack(x)
        y = torch.stack(y)
        originalText = torch.stack(originalText)
        return x, y, originalText


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

    model = Translator(engVocabSize=1000, hilliVocabSize=1000, embed_size=256,
                    num_encoder_blocks=8, num_decoder_blocks=8, num_heads=16, dropout=DROPOUT, pad_char=2).to(device)
    print(f"The number of parameters is {model.get_num_params()}")

    dataset = TextDataset(path = path+cur+"_to_"+target+"train.csv", engContextLength=500, hilliContextLength=500,cur=cur,target=target,)

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_dataset = TextDataset(path = path+cur+"_to_"+target+"val.csv", engContextLength=500, hilliContextLength=500,cur=cur,target=target, isTrain=False)

    test_dataset = TextDataset(path = path+cur+"_to_"+target+"test.csv", engContextLength=500, hilliContextLength=500,cur=cur,target=target, isTrain=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=4)
    model.train()
    optimizer = model.config_optimizer(LEARNING_RATE)


    def generate(sentence):
        hilliTokenizer = Tokenizer.from_file(f"models/{cur}_tokeniser.json")
        engTokenizer = Tokenizer.from_file(f"models/{cur}s_{target}_tokeniser.json")
        sentence = hilliTokenizer.encode(sentence).ids
        sentence = torch.tensor(
            sentence, dtype=torch.int64).unsqueeze(0).to(device)
        currentOutput = [0]
        model.eval()
        for i in range(100):
            x = torch.tensor(
                currentOutput, dtype=torch.int64).unsqueeze(0).to(device)
            output = model(x=x, originalText=sentence, return_loss=False)
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
            for data in val_dataset:
                engText, hilliText = data
                generated_text = generate(hilliText)
                generated.append(generated_text.split(" "))
                referenced.append([engText.split(" ")])
                # print(f"Generated: {generated_text}")
                # print(f"Referenced: {engText}\n\n")
            bleu = corpus_bleu(referenced, generated)
            bleu_scores.append(bleu)
            print(f"BLEU score: {bleu}")
            
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