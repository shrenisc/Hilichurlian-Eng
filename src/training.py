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
    LEARNING_RATE = 1e-3
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
        test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=4)
    model.train()
    optimizer = model.config_optimizer(LEARNING_RATE)

    def pad_for_batches(x):
        if(len(x)>=500):
            x=x[:500]
        else:
            x=x+[2 for _ in range(500 - len(x))]
        return x

    def generate_batch(sentences):
        
        hilliTokenizer = Tokenizer.from_file(f"models/{cur}_tokeniser.json")
        engTokenizer = Tokenizer.from_file(f"models/{cur}s_{target}_tokeniser.json")
        batch_size = 128  # Adjust batch size as needed
        sentence_batches = []
        #print(len(sentences[0]))
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            #print("inside")
            #print(batch[0].iloc[0])
            encoded_batch = [pad_for_batches(hilliTokenizer.encode(sentence).ids) for sentence in batch[0]]
            
            sentence_batches.append(torch.tensor(encoded_batch, dtype=torch.int64).to(device))
        
        current_outputs = [[0] for _ in range(min(len(sentences[0]),batch_size))]
        final_outputs = []
        for i in range(10):
            
            batch_x = torch.stack([torch.tensor(output, dtype=torch.int64) for output in current_outputs]).to(device)
            #print(batch_x.size())
            batch_outputs = model(x=batch_x, originalText=sentence_batches[0], return_loss=False)  
            #print("output shape: ",batch_outputs.size())
            batch_outputs = torch.argmax(batch_outputs, dim=2).squeeze(1).tolist()
            

            #print("batch outputs: ",batch_outputs)
            j=0
            n=min(len(sentences[0]),batch_size)
            while j<n:
                """ print(j,n,len(current_outputs))
                print(current_outputs[j])
                print(current_outputs[j][-1]) """
                if(current_outputs[j][-1]==1):
                    #print("inside")
                    final_outputs.append(current_outputs.pop(j))
                    sentence_batches[0]=sentence_batches[0].tolist()
                    sentence_batches[0].pop(j)
                    sentence_batches[0]=torch.tensor(sentence_batches[0], dtype=torch.int64).to(device)
                    """ print("sentences_batch_size:",[i.shape[0] for i in sentence_batches[0]])
                    print("current output:",[len(i) for i in current_outputs]) """
                    
                elif(isinstance(batch_outputs[j],list)):
                    current_outputs[j].append(batch_outputs[j][-1])
                    j+=1
                else:
                    current_outputs[j].append(batch_outputs[j])
                    j+=1
                n=len(current_outputs)
                
                

        if(len(current_outputs)!=0):
            for out in current_outputs:
                final_outputs.append(out)
        translated_sentences = [engTokenizer.decode(output) for output in final_outputs]
        return translated_sentences
    def validate_batch(val_dataset, batch_size=128):
        generated_sentences = []
        generation_bar = tqdm.tqdm(range(0, len(val_dataset), batch_size), desc=f"Epoch {epoch}/{EPOCHS}")
        for _,i in enumerate(generation_bar):
            #print(len(val_dataset[i:i + batch_size]))
            #print(i)
            batch = [hilliText for _, hilliText in [val_dataset[i:i + batch_size]]]
            batch_translations = generate_batch(batch)
            #print(batch_translations)
            generated_sentences.extend(batch_translations)
        return generated_sentences

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
            #print(x.size())
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
            """ if(i==10):
                break """
        with torch.no_grad():
            #get validation BLEU score
            print("Calculating BLEU score")
            generated = []
            referenced = [eng for eng,_ in val_dataset]
            generated_text = validate_batch(val_dataset)
            #print(val_dataset)
            generated.extend([sen.split(" ") for sen in generated_text])
            
            #print(f"Generated: {generated_text}")
            #print(f"Referenced: {engText}\n\n")
            #print(len(referenced),len(generated))
            #print(referenced)
            #print([len(i) for i in generated])
            bleu = corpus_bleu(referenced, generated)
            bleu_scores.append(bleu)
            print(f"BLEU score: {bleu}")
            
        torch.save(model, "models/model.pt")

    #testing the model
    with torch.no_grad():
        #get validation BLEU score
        print("Calculating BLEU score")
        generated = []
        referenced = [eng for eng,_ in test_dataset]
        generated_text = validate_batch(test_dataset)
        print(test_dataset)
        generated.extend([sen.split(" ") for sen in generated_text])
        
        #print(f"Generated: {generated_text}")
        #print(f"Referenced: {engText}\n\n")
        #print(len(referenced),len(generated))
        #print(referenced)
        #print([len(i) for i in generated])
        bleu = corpus_bleu(referenced, generated)
        bleu_scores.append(bleu)
        print(f"BLEU score: {bleu}")
    bleu_scores.append(bleu)
    pd.DataFrame(data=bleu_scores).to_csv(f"Bleu scores/{name}_bleu.csv")
    pd.DataFrame(data=train_loss).to_csv(f"Bleu scores/{name}_train_loss.csv")
    pd.DataFrame(data=train_acc).to_csv(f"Bleu scores/{name}_train_acc.csv")