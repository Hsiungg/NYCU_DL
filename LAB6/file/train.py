from model import ClassConditionalDDPM
from dataloader import iclevr_dataset
import torch
import torch.nn as nn
import argparse
import os
from diffusers import DDPMPipeline, DDPMScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import same_seeds
from diffusers.optimization import get_cosine_schedule_with_warmup
from evaluator import evaluation_model
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

class trainer():
    def __init__(self, args):
        self.args =args
        self.device = args.device
        train_set = iclevr_dataset(self.args.train_data_path, mode='train')
        valid_set = iclevr_dataset('_', mode='test')
        self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        self.valid_loader = DataLoader(valid_set, batch_size=32, num_workers = args.num_workers)
        self.model = ClassConditionalDDPM().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=(args.epochs),
        )
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=args.time_steps,  beta_schedule='squaredcos_cap_v2')
        self.criterion = nn.MSELoss()
        self.evaluator = evaluation_model()
        self.writer = SummaryWriter()
    
    def tqdm_bar_train(self, mode, epochs, pbar, loss, lr, grad_norm):
        pbar.set_description(f"{mode} Epoch {epochs}, lr:{lr:.6f}, grad_norm:{grad_norm:.3f}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
    
    def train_one_epoch(self, epoch):
        train_loss = 0.0
        self.model.train()
        for imgs, labels in (pbar := tqdm(self.train_loader, ncols=160)):
            clean_imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            noise = torch.randn(clean_imgs.shape).to(self.device)
            bs = clean_imgs.shape[0]
            timesteps = torch.randint(0, self.args.time_steps, (bs,), device=self.device).long()
            noisy_imgs = self.noise_scheduler.add_noise(clean_imgs, noise, timesteps)
            pred = self.model(noisy_imgs, timesteps, labels)
            #we predict noise
            
            #loss
            loss = self.criterion(pred, noise)
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf'))
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            train_loss += loss.detach().cpu()
            self.tqdm_bar_train('Train  |', epoch, pbar, loss.detach().cpu(), lr=self.lr_scheduler.get_last_lr()[0], grad_norm=grad_norm)
        
        train_loss /= len(self.train_loader)
        self.lr_scheduler.step()
        
        return train_loss
    def tqdm_bar_valid(self, mode, epochs, pbar, acc):
        pbar.set_description(f"{mode} Epoch {epochs}" , refresh=False)
        pbar.set_postfix(loss=float(acc), refresh=False)
        pbar.refresh()
        
    def eval_one_epoch(self, epoch):
        accuracy = 0.0
        output_imgs = []
        self.model.eval()
        with torch.no_grad():
            for i, label in enumerate(pbar := tqdm(self.valid_loader, ncols=120)):
                label = label.to(self.device)
                batch_size = label.shape[0]
                noise = torch.randn(batch_size, 3, 64, 64).to(self.device)
                incompleted_img = [[] for _ in range(batch_size)]
                for idx, t in tqdm(enumerate(self.noise_scheduler.timesteps)):
                    residual = self.model(noise, t, label)
                    noise = self.noise_scheduler.step(residual, t, noise).prev_sample
                    
                    if idx % 100 == 0 or idx % (self.args.time_steps - 1) == 0:
                        for b in range(batch_size):
                            incompleted_img[b].append(noise[b].detach().cpu())
                
                img = noise
                acc = self.evaluator.eval(img, label)
                accuracy += acc
                
                for b in range(batch_size):
                    output_imgs.append(img[b].detach().cpu())
                    incompleted_grid = make_grid(incompleted_img[b], normalize=True)
                    path = os.path.join(self.args.save_png, 'incompleted')
                    os.makedirs(path, exist_ok=True)
                    save_image(incompleted_grid, os.path.join(path, f'valid_{i}_epoch_{epoch}.png'))
                    
                self.tqdm_bar_valid('Valid  |', epoch, pbar, acc)
            
            output_imgs = torch.stack(output_imgs)
            out_grid = make_grid(output_imgs, normalize=True)
            path = os.path.join(self.args.save_png, 'out')
            os.makedirs(path, exist_ok=True)
            save_image(out_grid, os.path.join(path, f'valid_out_{epoch}.png'))
            accuracy /= len(self.valid_loader)
            print(f'Accuracy = {accuracy}')
            return accuracy
    
    def train(self):
        best_acc = 0
        root = os.path.join(self.args.save_path, 'checkpoint')
        os.makedirs(root, exist_ok=True)
        for epoch in range(self.args.epochs):
            loss = self.train_one_epoch(epoch)
            self.writer.add_scalar("Loss/train", loss, epoch)
            
            if epoch % self.args.save_per_epoch ==0:
                path = os.path.join(root, f"epoch_{epoch}.pth")
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                    }
                torch.save(checkpoint, path)
                print(f"save epoch{epoch}.pth to {path}")
            #eval
            if epoch >= 50:
                accuracy = self.eval_one_epoch(epoch)
                self.writer.add_scalar("Acc/valid", accuracy, epoch)
            
                if best_acc < accuracy:
                    path = os.path.join(root, f"Best_acc.pth")
                    best_acc = accuracy
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        "lr": self.lr_scheduler.get_last_lr()[0],
                        }
                    torch.save(checkpoint, path)
                    print(f"save Best_acc.pth to {path}")
                
        self.writer.close()       


def get_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--train_data_path', type=str, default="iclevr", help='Training Dataset Path')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--save_png', type=str, default='/media/hsiung/sata2tb/DLP/LAB6/out_imgs', help='Path to save plt')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
      

    #you can modify the hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--time_steps',     type=int, default=1000,     help="number of noise adding step")
    parser.add_argument('--warmup_steps', type=int, default = 20, help='Number of epochs for warmup schedular.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--save_per_epoch', type=int, default=10, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    #parser.add_argument('--beta', type=int, default=0, help='Beta for denoise')
    
    #save path
    parser.add_argument('--save_path', type=str, default="/media/hsiung/sata2tb/DLP/LAB6", help='save models path')



    args = parser.parse_args()        
    return args
        

if __name__ == '__main__':
    same_seeds(444)
    args = get_args()
    os.makedirs(os.path.join(args.save_path), exist_ok=True)
    os.makedirs(os.path.join(args.save_png), exist_ok=True)
    train = trainer(args)
    train.train()