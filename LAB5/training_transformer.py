import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
import random
from torch.utils.data import DataLoader
from utils import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def same_seeds(seed):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers(args.epochs, args.learning_rate)
        self.prepare_training()
        self.writer = SummaryWriter()
        
    @staticmethod
    def prepare_training(): 
        os.makedirs(os.path.join(args.save_path,"transformer_checkpoints"), exist_ok=True)

    def train_one_epoch(self, train_loader, criterion, device, epoch):
        train_loss = 0.0
        for imgs in (pbar := tqdm(train_loader, ncols=120)):
            imgs = imgs.to(device)
            logits, target = self.model(imgs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            train_loss += loss.detach().cpu()

            self.tqdm_bar('Train  |', epoch, pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])

        train_loss /= len(train_loader)
        self.scheduler.step()
        
        return train_loss


    def eval_one_epoch(self, valid_loader, criterion, device, epoch):
        valid_loss = 0.0
        for imgs in (pbar := tqdm(valid_loader, ncols=120)):
            imgs = imgs.to(args.device)
            
            logits, target = self.model(imgs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            
            valid_loss += loss.detach().cpu()
        
            self.tqdm_bar('Valid |', epoch, pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
    
        valid_loss /= len(valid_loader)
        
        return valid_loss

    def configure_optimizers(self, epochs, lr):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.96))
        scheduler = get_cosine_schedule_with_warmup(optimizer, 35, epochs)
        return optimizer,scheduler
    
    def tqdm_bar(self, mode, epochs, pbar, loss, lr):
        pbar.set_description(f"{mode} Epoch {epochs}, lr:{lr:.6f}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()

def plot(epochs, plot_list, path):
        os.makedirs(path, exist_ok=True)
        plt.figure(figsize=(14, 6))
        # for loss
        plt.plot(epochs, plot_list[0], marker='.', linestyle='-', linewidth=2, color='blue', label='Train Loss')
        plt.plot(epochs, plot_list[1], marker='.', linestyle='-', linewidth=2, color='orange', label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(path, "plot_loss.png"))
        plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="lab5_dataset/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="lab5_dataset/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--save_plt', type=str, default='./plt/1', help='Path to save plt')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum_grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train.')
    parser.add_argument('--save_per_epoch', type=int, default=10, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start_from_epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt_interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate.')
    
    #save path
    parser.add_argument('--save_path', type=str, default="/media/hsiung/sata2tb/DLP/tryrandaug", help='save models path')
    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)
    criterion = nn.CrossEntropyLoss()

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:
    same_seeds(444)
    train_losses = []
    valid_losses = []
    best_loss = float('inf')
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        #train
        train_transformer.model.train()
        train_loss = train_transformer.train_one_epoch(train_loader, criterion, args.device, epoch)
        train_losses.append(train_loss)
        train_transformer.writer.add_scalar('Train_loss', train_loss, epoch)


        #valid
        train_transformer.model.eval()
        with torch.no_grad():
            valid_loss = train_transformer.eval_one_epoch(val_loader, criterion, args.device, epoch)
            valid_losses.append(valid_loss)
            train_transformer.writer.add_scalar('Valid_loss', valid_loss, epoch)
        
        print(f'Train_loss = {train_loss:.3f}, Valid_loss = {valid_loss:.3f}')

        if epoch % args.save_per_epoch == 0:
            path = os.path.join(args.save_path,"transformer_checkpoints/", f"epoch={epoch}.ckpt")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': train_transformer.model.state_dict(),
                'optimizer_state_dict': train_transformer.optim.state_dict(),
                "lr": train_transformer.scheduler.get_last_lr()[0],
                }

            torch.save(checkpoint, path)
            print(f"save ckpt to {path}")

        if best_loss > valid_loss:
            path = os.path.join(args.save_path,"transformer_checkpoints/", f"Best_loss.ckpt")
            best_loss = valid_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': train_transformer.model.state_dict(),
                'optimizer_state_dict': train_transformer.optim.state_dict(),
                "lr": train_transformer.scheduler.get_last_lr()[0],
                }

            torch.save(checkpoint, path)
            print(f"save ckpt to {path}")

    train_transformer.writer.close()
    epochs = list(range(args.epochs))
    plot_list = [
        train_losses,
        valid_losses
        ]
    plot(epochs, plot_list, args.save_plt)
