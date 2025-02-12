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


class tester():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        test_set = iclevr_dataset('_', mode=args.test_set_type)
        self.test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers = args.num_workers)
        self.model = ClassConditionalDDPM().to(self.device)
        checkpoint = torch.load(args.model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=args.time_steps,  beta_schedule='squaredcos_cap_v2')
        self.evaluator = evaluation_model()
        
    def tqdm_bar_test(self, mode, test_set_type, pbar, acc):
        pbar.set_description(f"{mode} {test_set_type}" , refresh=False)
        pbar.set_postfix(loss=float(acc), refresh=False)
        pbar.refresh()

    def test(self):
        accuracy = 0.0
        output_imgs = []
        test_set_type = self.args.test_set_type
        self.model.eval()
        with torch.no_grad():
            for i, label in enumerate(pbar := tqdm(self.test_loader, ncols=120)):
                label = label.to(self.device)
                batch_size = label.shape[0]
                noise = torch.randn(batch_size, 3, 64, 64).to(self.device)
                incompleted_img = [[] for _ in range(batch_size)]
                for idx, t in tqdm(enumerate(self.noise_scheduler.timesteps)):
                    residual = self.model(noise, t, label)
                    noise = self.noise_scheduler.step(residual, t, noise).prev_sample
                    
                    if idx % 50 == 0 or idx == (self.args.time_steps - 1):
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
                    save_image(incompleted_grid, os.path.join(path, f'{test_set_type}_{b}.png'))
                    
                self.tqdm_bar_test('Test  |', test_set_type, pbar, acc)
            
            output_imgs = torch.stack(output_imgs)
            out_grid = make_grid(output_imgs, normalize=True)
            path = os.path.join(self.args.save_png, 'out')
            os.makedirs(path, exist_ok=True)
            save_image(out_grid, os.path.join(path, f'test_out_{test_set_type}.png'))
            accuracy /= len(self.test_loader)
            print(f'Accuracy = {accuracy}')
    
def get_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--test_set_type', type=str, default='test', help='test or new_test')
    parser.add_argument('--model_path', type=str, default='/media/hsiung/sata2tb/DLP/LAB6/checkpoint/Best_acc.pth', help='Path to checkpoint.')
    parser.add_argument('--save_png', type=str, default='/media/hsiung/sata2tb/DLP/LAB6/out_imgs', help='Path to save plt')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
      

    #you can modify the hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--time_steps',     type=int, default=1000,     help="number of noise adding step")




    args = parser.parse_args()        
    return args


if __name__ == '__main__':
    same_seeds(444)
    args = get_args()
    os.makedirs(os.path.join(args.save_png), exist_ok=True)
    test = tester(args)
    test.test()
