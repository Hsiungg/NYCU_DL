
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
from utils import dice_score
import gc
torch.backends.cudnn.benchmark = True

def evaluate(model, data, device, args, criterion):
    # implement the evaluation function here
    #data = valid_set
    model.eval()
    eval_loss = []
    eval_dices = []
    eval_loader = DataLoader(
        data,
        batch_size = args.batch_size,
        shuffle = False,
        drop_last = False,
        pin_memory=True
    )
    
    with torch.no_grad():
        for sample in tqdm(eval_loader):
            gc.collect
            torch.cuda.empty_cache()
            features, mask = sample["image"].to(device), sample["mask"].to(device)

            logits = model(features)

            logits = logits.view(logits.size(0), -1)
            mask = mask.view(mask.size(0), -1)
            torch.cuda.synchronize()
            print("logits shape:", logits.shape)
            print("mask shape:", mask.shape)
            
            loss = criterion(logits.to(device), mask.to(device))
            dice = dice_score(logits, mask).item()
            
            eval_loss.append(loss.item())
            eval_dices.append(dice)
            torch.cuda.synchronize()
    
    eval_loss = sum(eval_loss) / len(eval_loss)
    eval_dice = sum(eval_dices) / len(eval_dices)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return eval_loss, eval_dice