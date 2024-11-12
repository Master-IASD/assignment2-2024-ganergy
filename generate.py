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

    # Defining the Gaussian Mixture 
    AB_means = torch.tensor([0.0, 5.0, -5.0, 2.0, -2.0])  
    AB_stdevs = torch.tensor([0.7, 0.5, 0.3, 0.8, 0.4])  

    #the mixture distribution
    AB_dist = torch.distributions.Normal(AB_means, AB_stdevs)
    mix_weight = torch.distributions.Categorical(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]))
    mix_dist = torch.distributions.MixtureSameFamily(mix_weight, AB_dist)

    # sample from the mixture distribution (1D samples)
    # mix_samp = mix_dist.sample((n_samples,))

    
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


    
