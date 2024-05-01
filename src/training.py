from model import Translator
from dataset import TextDataset
import torch
import tqdm
from tokenizers import Tokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = 'false'

EPOCHS = 5

def collate_fn(batch):
    x = [s[0] for s in batch]
    y = [s[1] for s in batch]
    originalText = [s[2] for s in batch]
    x = torch.stack(x)
    y = torch.stack(y)
    originalText = torch.stack(originalText)
    return x, y, originalText

print(f"Using PyTorch version {torch.__version__}")

#use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

#use tensor cores
torch.set_float32_matmul_precision('high')

#use flash attention
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
    
model = Translator(engVocabSize = 1000, hilliVocabSize = 1000, embed_size = 32, 
num_encoder_blocks = 4, num_decoder_blocks = 1, num_heads = 8, dropout = 0.1, pad_char = 4).to(device)
print(f"The number of parameters is {model.get_num_params()}")

dataset = TextDataset(engContextLength = 100, hilliContextLength = 100)

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size = 4, shuffle = True, collate_fn = collate_fn, num_workers = 4)
model.train()
optimizer = model.config_optimizer(lr = 1e-2)

def generate(sentence):
    hilliTokenizer = Tokenizer.from_file("models/hilliTokenizer.json")
    engTokenizer = Tokenizer.from_file("models/englighTokenizer.json")
    sentence  = hilliTokenizer.encode(sentence).ids
    sentence = torch.tensor(sentence, dtype = torch.int64).unsqueeze(0).to(device)
    currentOutput = [0]
    model.eval()
    for i in range(100):
        x = torch.tensor(currentOutput, dtype = torch.int64).unsqueeze(0).to(device)
        output = model(x = x, originalText = sentence, return_loss = False)
        output = torch.argmax(output[0][-1]).item()
        currentOutput.append(output)
    currentOutput = engTokenizer.decode(currentOutput)
    model.train()
    return currentOutput

for epoch in range(EPOCHS):

    progress_bar = tqdm.tqdm(train_dataloader, desc = f"Epoch {epoch}/{EPOCHS}")
    for i, data in enumerate(progress_bar):
        x, y, originalText = data
        x, y, originalText = x.to(device), y.to(device), originalText.to(device)
        
        loss, acc = model(x = x, originalText = originalText, y = y, return_loss = True)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none = True)
        progress_bar.postfix = f"Loss: {loss.item()}, acc: {acc.item()}"
    print(generate("Mi mosi gusha.")) #I am ahppy
    
    torch.save(model, "models/model.pt")
