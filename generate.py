import torch 
import torchvision
import os
import argparse


from model import Generator, Generator_BN, WGAN_Generator
from utils import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    args = parser.parse_args()

    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    model = Generator_BN(g_output_dim = mnist_dim).cuda()
    model = load_model(model, 'checkpoints')
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    print('Model loaded.')


    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples < 10000:
            z = torch.randn(args.batch_size, 100).cuda()
            x_fake = model(z)
            x_fake = x_fake.view(args.batch_size, 1, 28, 28)
            for img in x_fake:
                if n_samples < 10000:
                    torchvision.utils.save_image(img, os.path.join('samples', f'{n_samples}.png'), normalize=True)
                    n_samples += 1


    
