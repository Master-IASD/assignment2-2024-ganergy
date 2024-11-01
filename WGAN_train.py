import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim


from model import WGAN_Generator, WGAN_Discriminator
from utils import WGAN_D_train, WGAN_G_train, save_models




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")

    args = parser.parse_args()


    os.makedirs('chekpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')


    print('Model Loading...')
    mnist_dim = 784
    G = WGAN_Generator(g_output_dim = mnist_dim).cuda()
    D = WGAN_Discriminator(mnist_dim).cuda()


    # model = DataParallel(model).cuda()
    print('Model loaded.')
    # Optimizer 
 

    # define optimizers
    G_optimizer = optim.RMSprop(G.parameters(), lr = args.lr)
    D_optimizer = optim.RMSprop(D.parameters(), lr = args.lr)

    print('Start Training :')
    
    n_epoch = args.epochs
    for epoch in trange(1, n_epoch+1, leave=True):           
        for batch_idx, (x, _) in enumerate(train_loader):
            #Train Descriminator
            WGAN_D_train(x, G, D, D_optimizer)

            if batch_idx % 5 == 0 : 
                #Train generator 
                z = WGAN_G_train(x, G, D, G_optimizer)

        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')
                
    print('Training done')

        
