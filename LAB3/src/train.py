import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from oxford_pet import *
from utils import get_hyperparameters, DiceBCELoss, same_seeds, plot,dice_score
from models.unet import Unet
from models.resnet34_unet import Resnet34_unet
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from evaluate import evaluate
import gc

def train(args):
    # implement the training function here
    #data_path epochs batch_size learning rate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stale = 0
    best_dice = 0
    hyperparameters = get_hyperparameters(args.model)
    if args.model == 'unet':
        model = Unet(3, 1).to(device)
    else:
        model = Resnet34_unet(3, 1).to(device)
        
    train_set = load_dataset(args.data_path, 'train', transform=True)
    #criterion = torch.nn.BCEWithLogitsLoss().to(device)
    criterion = DiceBCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = hyperparameters["lr"], weight_decay = hyperparameters['weight_decay'])
    n_epochs = hyperparameters["n_epochs"]
    momentum = 0.99 if args.model == 'unet' else 0.9
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, momentum)
    writer = SummaryWriter(args.model)
    train_losses = []
    valid_losses = []
    train_dices = []
    valid_dices = []
    train_loader = DataLoader(
        train_set,
        batch_size = args.batch_size,
        shuffle = True,
        drop_last = False,
    )
    gc.collect()
    torch.cuda.empty_cache()
    for epoch in range(hyperparameters['n_epochs']):
        model.train()
        
        train_loss = []
        train_dice = []
        for sample in tqdm(train_loader):
            model.train()
            img, mask = sample["image"].to(device), sample["mask"].to(device)
            logits = model(img)
            
            loss = criterion(logits, mask)
            
            optimizer.zero_grad()
            

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            
            optimizer.step()
            model.eval()
            dice = dice_score(logits.to(device), mask).cpu().float().mean()
            train_loss.append(loss.cpu().item())
            train_dice.append(dice.cpu())
            
        scheduler.step()
        
        train_loss = sum(train_loss) / len(train_loss)
        train_dice = sum(train_dice) / len(train_dice)
        
        train_losses.append(train_loss)
        train_dices.append(train_dice)
        writer.add_scalar('Training Accuracy', train_dice, epoch)
        writer.add_scalar('Training Loss', train_loss, epoch)
        
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, dice = {train_dice:.5f}")
        
        valid_set = load_dataset(args.data_path, 'valid')
        valid_loss, valid_dice = evaluate(model, valid_set, device, args, criterion)
        
        valid_losses.append(valid_loss)
        valid_dices.append(valid_dice)
        writer.add_scalar('Validation Accuracy', valid_dice, epoch)
        writer.add_scalar('Validation loss', valid_loss, epoch)
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, dice = {valid_dice:.5f}")
        
        if valid_dice > best_dice:
            print(f"Best model found at epoch {epoch }, saving model")
            save_path = f"saved_models/{args.model}"
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), f"{save_path}/{args.model}_best.pth") # only save best to prevent output memory exceed error
            best_dice = valid_dice
            stale = 0
        else:
            stale += 1
            if stale > hyperparameters['patience']:
                print(f"No improvment {hyperparameters['patience']} consecutive epochs, early stopping")
                n_epochs = epoch + 1
                break
        gc.collect()
        torch.cuda.empty_cache()
    writer.close()
    epochs = list(range(n_epochs))
    plot_list = [
        train_losses,
        valid_losses,
        train_dices,
        valid_dices
    ]
    plot(epochs, plot_list)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default = os.path.join("dataset/oxford-iiit-pet"), help='path of the input data')
    parser.add_argument('--batch_size', '-b', type=int, default= 6, help='batch size')
    parser.add_argument("--model", "-m", type = str, default = 'unet', help = 'unet or resnet34_unet')

    return parser.parse_args()
 
if __name__ == "__main__":
    same_seeds(123)
    args = get_args()
    train(args)