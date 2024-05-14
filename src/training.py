from model import Translator
from dataset import TextDataset
import torch
import tqdm
from tokenizers import Tokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = 'false'

EPOCHS = 30
LEARNING_RATE = 7e-4
DROPOUT = 0.1

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

model = Translator(engVocabSize=129, hilliVocabSize=129, embed_size=256,
                   num_encoder_blocks=7, num_decoder_blocks=7, num_heads=8, dropout=DROPOUT, pad_char=0).to(device)
print(f"The number of parameters is {model.get_num_params()}")

dataset = TextDataset(engContextLength=500, hilliContextLength=500)

train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4)
model.train()
optimizer = model.config_optimizer(LEARNING_RATE)


def generate(sentence):
    sentence = [ord(char) for char in sentence]
    sentence = torch.tensor(
        sentence, dtype=torch.int64).unsqueeze(0).to(device)
    currentOutput = [2]
    model.eval()
    for i in range(500):
        x = torch.tensor(
            currentOutput, dtype=torch.int64).unsqueeze(0).to(device)
        output = model(x=x, originalText=sentence, return_loss=False)
        output = torch.argmax(output[0][-1]).item()
        currentOutput.append(output)
        if (output == 1):
            break
    #convert ascii to char
    finalOutput = ""
    for char in currentOutput:
        finalOutput += chr(char)
    model.train()
    return finalOutput


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
    with torch.no_grad():
        print(generate("Mi mosi gusha."))  # I am ahppy

    torch.save(model, "models/model.pt")
