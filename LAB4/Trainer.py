import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio
import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from math import log10

def same_seeds(seed):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    torch.clamp(mse, min=1e-10, max=None)
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
    logvar_exp_clipped = torch.clamp(logvar.exp(), min=1e-10, max=None)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar_exp_clipped)
    KLD /= batch_size  
    return KLD


class kl_annealing():
    #reference https://github.com/haofuml/cyclical_annealing
    def __init__(self, args, current_epoch = 0):
        # TODO
        self.anneal_type = args.kl_anneal_type
        self.anneal_cycle = args.kl_anneal_cycle
        self.anneal_ratio = args.kl_anneal_ratio
        self.current_epoch = current_epoch
        self.n_epochs = args.num_epoch
        assert self.anneal_type in ["Monotonic", "Cyclical", "None"], "Invalid anneal_type"
        if self.anneal_type == "Monotonic":
            self.Beta_list = self.frange_cycle_linear(self.n_epochs, n_cycle = 1, ratio = 0.25)

        elif self.anneal_type == "Cyclical":
            self.Beta_list = self.frange_cycle_linear(self.n_epochs, n_cycle = self.anneal_cycle, ratio = self.anneal_ratio)
        else:
            self.Beta_list = np.ones(self.n_epochs)
        
        self.beta = self.Beta_list[0]
        fig = plt.figure(figsize=(8,4.0))
        stride = max( int(self.n_epochs / self.anneal_cycle), 1)
        plt.plot(range(self.n_epochs), self.Beta_list, '-', label = self.anneal_type, marker= 's', color='k', markevery=stride,lw=2,  mec='k', mew=1 , markersize=10)
        os.makedirs(os.path.join("./kl_pic"), exist_ok=True)
        plt.savefig(os.path.join("./kl_pic", f"{self.anneal_type}_{self.anneal_cycle}.png"))
        plt.show()
             
    def update(self):
        # TODO
        self.current_epoch += 1
        if self.current_epoch >= self.n_epochs:
            return
        self.beta = self.Beta_list[self.current_epoch]
            
    
    def get_beta(self):
        # TODO
        return self.beta

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle = 4, ratio=0.5):
        # TODO
        L = np.ones(n_iter)
        period = n_iter / n_cycle
        step = (stop - start) / (period * ratio)
        #linear step
        for cycle in range(n_cycle):
            beta_step, current_step = start , 0
            while beta_step <= stop and (int(current_step + cycle * period) < n_iter):
                L[int(current_step + cycle*period)] = beta_step
                beta_step += step
                current_step += 1
        return L 
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.AdamW(self.parameters(), lr=self.args.lr)
        #self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[5, 15, 25], gamma=0.3)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optim, total_steps=100, max_lr=0.0002, three_phase=True )
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        self.PSNR_best = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        #for plotting result
        self.writer = SummaryWriter(self.kl_annealing.anneal_type)
        
        #for init
        self._initialize_weights()
        
    def forward(self, img, label):
        pass
    
    def _initialize_weights(self):
        #reference https://discuss.pytorch.org/t/kld-loss-goes-nan-during-vae-training/42305/2
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.uniform_(m.weight, -0.08, 0.08)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def training_stage(self):
        train_losses = []
        valid_losses = []
        valid_PSNRs = []
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            train_loss = 0
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                #adapt_TeacherForcing = True if random.random() < self.tfr else False
                # try to change to batch level
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                train_loss += loss.item()
                
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {:.3f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {:.3f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            #for train loss plot
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            self.writer.add_scalar('Train_loss', train_loss, self.current_epoch)
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))

                
            valid_loss, valid_PSNR = self.eval()
            valid_losses.append(valid_loss)
            valid_PSNRs.append(valid_PSNR)
            self.writer.add_scalar('Valid_loss', valid_loss, self.current_epoch)
            self.writer.add_scalar('valid_PSNR', valid_PSNR, self.current_epoch)
            self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], self.current_epoch)
            self.writer.add_scalar('beta', self.kl_annealing.get_beta(), self.current_epoch)
            self.writer.add_scalar('tfr', self.tfr, self.current_epoch)
            
            self.current_epoch += 1
            self.scheduler.step()
            #self.scheduler.step(valid_PSNR)
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
        self.writer.close()
        epochs = list(range(self.args.num_epoch))
        plot_list = [
        train_losses,
        valid_losses,
        valid_PSNRs
        ]
        self.plot(epochs, plot_list, self.args.save_root)
        
    @torch.no_grad()
    def eval(self):
        loss, PSNR= 0, 0
        PSNRs = []
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, PSNR, PSNRs= self.val_one_step(img, label)
            #output PSNR per frame here
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            print(f"val  | PSNR = {PSNR:.5f}")
            if PSNR > self.PSNR_best:
                self.save(os.path.join(self.args.save_root, f"PSNR_Best.ckpt"))
                self.PSNR_best = PSNR
        if args.test:
            plt.figure(figsize=(14, 6))
            frame = list(range(len(PSNRs)))
            plt.plot(frame, PSNRs, marker='.', linestyle='-', linewidth=2, color='orange', label='Train Loss')
            plt.title('PSNR_per_frame')
            plt.xlabel('Frame')
            plt.ylabel('PSNR')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.xlim(0, 630)
            plt.savefig(os.path.join("PSNR_per_frame1.png"), bbox_inches='tight')
            plt.show()
        return loss.detach().item(), PSNR.detach().item()
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        # TODO
        self.frame_transformation.train()
        self.label_transformation.train()
        self.Gaussian_Predictor.train()
        self.Decoder_Fusion.train()
        self.Generator.train()
        
        self.img = img.permute(1, 0, 2, 3, 4)
        self.label = label.permute(1, 0, 2, 3, 4)
        MSE_loss = 0
        KL_loss = 0
        pred_out = self.img[0]
        #do first image
        for idx in range(self.train_vi_len - 1):
            if idx == 0 or adapt_TeacherForcing:
                encoded_frame = self.frame_transformation(self.img[idx])
            else:
                encoded_frame = self.frame_transformation(pred_out)
            
            encoded_frame_next = self.frame_transformation(self.img[idx + 1])
            encoded_label = self.label_transformation(self.label[idx + 1])
            z, mu, logvar = self.Gaussian_Predictor(encoded_frame_next, encoded_label)
            #z is for decoder fusion , mu and logvar is for KL_loss
            decoded_pred = self.Decoder_Fusion(encoded_frame, encoded_label, z)
            pred_out = self.Generator(decoded_pred)
            MSE = self.mse_criterion(pred_out, self.img[idx + 1])
            KL = kl_criterion(mu, logvar, batch_size = self.batch_size)
            MSE_loss += MSE
            KL_loss += KL
            #calculate loss for optimizer
        
        beta = torch.tensor(self.kl_annealing.get_beta()).to(self.args.device)
        MSE_loss /= (self.train_vi_len - 1)
        KL_loss /= (self.train_vi_len - 1)
        total_loss = MSE_loss + beta * KL_loss
        
        self.optim.zero_grad()
        total_loss.backward()
        self.optimizer_step()
        return total_loss.detach()
            
            
    
    def val_one_step(self, img, label):
        # TODO
        self.frame_transformation.eval()
        self.label_transformation.eval()
        self.Gaussian_Predictor.eval()
        self.Decoder_Fusion.eval()
        self.Generator.eval()
        
        with torch.no_grad():
            self.img = img.permute(1, 0, 2, 3, 4)
            self.label = label.permute(1, 0, 2, 3, 4)
            output_frame_list = [self.img[0].cpu()]
            PSNRs = []
            MSEloss = 0
            
            for idx in range(self.val_vi_len - 1):
                encoded_frame = self.frame_transformation(output_frame_list[idx].to(self.args.device))
                encoded_label = self.label_transformation(self.label[idx + 1])
                z = torch.normal(0, 1, size=(1, self.args.N_dim, self.args.frame_H, self.args.frame_W)).to(self.args.device)
                #sample random z from normal distribution
                decoded_pred = self.Decoder_Fusion(encoded_frame, encoded_label, z)
                pred_out = self.Generator(decoded_pred)
                output_frame_list.append(pred_out.cpu())
                MSEloss += self.mse_criterion(pred_out, self.img[idx + 1]).cpu()
                PSNR = Generate_PSNR(pred_out, self.img[idx + 1])
                PSNRs.append(PSNR.detach().cpu())
            
            Avg_PSNR = sum(PSNRs) / len(PSNRs)
            MSEloss /= (self.val_vi_len - 1)
            return MSEloss, Avg_PSNR, PSNRs
            
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # TODO
        if self.current_epoch > self.tfr_sde and self.tfr >= 0:
            self.tfr -= self.tfr_d_step
            self.tfr = max(self.tfr, 0)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr:.5f}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr'] * 0.03
            #self.args.lr = 0.000001
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.AdamW(self.parameters(), lr=self.args.lr)
            #self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, "max", patience = 20, min_lr=1e-5, factor=0.3)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optim, total_steps=100, max_lr=0.0003, three_phase=True )
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()
    
    def plot(self, epochs, plot_list, path):
        plt.figure(figsize=(14, 6))
        # for loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, plot_list[0], marker='.', linestyle='-', linewidth=2, color='blue', label='Train Loss')
        plt.plot(epochs, plot_list[1], marker='.', linestyle='-', linewidth=2, color='orange', label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

    # for PSNR
        plt.subplot(1, 2, 2)
        plt.plot(epochs, plot_list[2], marker='.', linestyle='-', linewidth=2, color='orange', label='Validation PSNR')
        plt.title('PSNR')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(path, "plot_out.png"))
        plt.show()


def main(args):
    same_seeds(4444)
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=1)
    parser.add_argument('--lr',            type=float,  default=0.0001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="AdamW")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, default="LAB4_Dataset/LAB4_Dataset",  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, default= "./save_models_con",help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=6)
    parser.add_argument('--num_epoch',     type=int, default=100,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=15,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.05,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,   default= "save_models_con/epoch=36.ckpt",help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="Chose between Monotonic, Cyclical and None")
    parser.add_argument('--kl_anneal_cycle',    type=int, default = 8,               help="only for Cyclic")
    parser.add_argument('--kl_anneal_ratio',    type=float, default = 0.5,              help="ratio")
    

    

    args = parser.parse_args()
    
    main(args)
