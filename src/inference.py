
import argparse
from oxford_pet import load_dataset
from evaluate import evaluate
from utils import DiceBCELoss
import torch
import os
from models.unet import Unet
from models.resnet34_unet import Resnet34_unet
import gc
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--data_path', type=str, default = os.path.join("dataset/oxford-iiit-pet"), help='path of the input data')
    parser.add_argument('--batch_size', '-b', type=int, default= 4, help='batch size')
    parser.add_argument("--model", "-m", type = str, default = 'unet', help = 'unet or resnet34_unet')

    return parser.parse_args()

if __name__ == '__main__':
    gc.collect
    torch.cuda.empty_cache()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    args = get_args()
    test_set = load_dataset(args.data_path, 'test')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = DiceBCELoss().to(device)
    if args.model == 'unet':
        model_best = Unet(3, 1).to(device)
    else:
        model_best = Resnet34_unet(3, 1).to(device)
    
    model_best.load_state_dict(torch.load(f"./saved_models/{args.model}/{args.model}_best.pth", weights_only=False))
    test_loss, test_dice = evaluate(model_best, test_set, device, args, criterion)
    print(f"  Test |  test_loss = {test_loss: .5f},dice = {test_dice:.5f}")