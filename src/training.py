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
    LEARNING_RATE = 1e-4
    DROPOUT = 0.3
    eng_context_length = 500
    hilli_churl_context_length = 500

    val_batch_size = 128
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

    dataset = TextDataset(path = path+cur+"_to_"+target+"train.csv", engContextLength=eng_context_length, hilliContextLength=hilli_churl_context_length,cur=cur,target=target,)

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_dataset = TextDataset(path = path+cur+"_to_"+target+"val.csv", engContextLength=eng_context_length, hilliContextLength=hilli_churl_context_length,cur=cur,target=target, isTrain=False)

    test_dataset = TextDataset(path = path+cur+"_to_"+target+"test.csv", engContextLength=eng_context_length, hilliContextLength=hilli_churl_context_length,cur=cur,target=target, isTrain=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=4)
    model.train()
    optimizer = model.config_optimizer(LEARNING_RATE)

    def pad_for_batches(x):
        if(len(x)>=hilli_churl_context_length):
            x=x[:hilli_churl_context_length]
        else:
            x=x+[2 for _ in range(hilli_churl_context_length - len(x))]
        return x

    def generate_batch(sentences,batch_size):
        
        engTokenizer = Tokenizer.from_file(f"models/{cur}_tokeniser.json")
        hilliTokenizer = Tokenizer.from_file(f"models/{cur}s_{target}_tokeniser.json")
        sentence_batches = [pad_for_batches(hilliTokenizer.encode(sentence).ids) for sentence in sentences]
        
        current_outputs = torch.zeros((min(batch_size,len(sentences)),1),dtype=torch.int32).to(device)
        sentence_batches=torch.tensor(sentence_batches,dtype=torch.int32).to(device)
        final_outputs = []

        for i in range(eng_context_length):
            if(len(current_outputs)==0):
                break
            
            batch_outputs = model(x=current_outputs, originalText=sentence_batches, return_loss=False)  
            batch_outputs = torch.argmax(batch_outputs, dim=2)[:,-1]
            current_outputs = torch.cat((current_outputs,batch_outputs.reshape(current_outputs.shape[0],-1)),dim=1)
            completed_sentences = current_outputs[current_outputs[:,-1] == 1,:]
            sentence_batches = sentence_batches[current_outputs[:,-1] != 1,:]
            current_outputs = current_outputs[current_outputs[:,-1] != 1,:]
            
            
            if(len(completed_sentences)):
                final_outputs.extend(completed_sentences.tolist())
                

        if(len(current_outputs)!=0):
            current_outputs=current_outputs.tolist()
            for out in current_outputs:
                final_outputs.append(out)

        translated_sentences = []
        for output in final_outputs:
            temp=[engTokenizer.decode([token]) for token in output]
            translated_sentences.append(temp)

        return translated_sentences
    
    def validate_batch(val_dataset, batch_size=64):
        generated_sentences = []
        gen_epoch=0
        generation_bar = tqdm.tqdm(range(0, len(val_dataset), batch_size), desc=f"Validating : ")
        for _,i in enumerate(generation_bar):
            
            batch = val_dataset[i:i + batch_size][1]
            batch_translations = generate_batch(batch,batch_size)
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
            referenced = [[eng.split()] for eng,_ in val_dataset]
            generated_text = validate_batch(val_dataset,val_batch_size)
            print("Referenced text: ",referenced[:5])
            print("Generated text: ",generated_text[:5])
            bleu = corpus_bleu(referenced,generated_text)
            bleu_scores.append(bleu)
            print(f"BLEU score: {bleu}")
            
        torch.save(model, "models/model.pt")

    #testing the model
    with torch.no_grad():
        #get validation BLEU score
        print("Calculating BLEU score")
        referenced = [[eng.split()] for eng,_ in test_dataset]
        generated_text = validate_batch(test_dataset)
        bleu = corpus_bleu(referenced,generated_text)
        bleu_scores.append(bleu)
        print(f"BLEU score: {bleu}")

    bleu_scores.append(bleu)
    pd.DataFrame(data=bleu_scores).to_csv(f"Bleu scores/{name}_bleu.csv")
    pd.DataFrame(data=train_loss).to_csv(f"Bleu scores/{name}_train_loss.csv")
    pd.DataFrame(data=train_acc).to_csv(f"Bleu scores/{name}_train_acc.csv")