import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim


from model import Generator_BN, Discriminator_SA
from utils import D_train_SA, G_train_SA, save_models




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs for training.")
    parser.add_argument("--lr_D", type=float, default=0.0004,
                      help="The learning rate of Descriminator to use for training.")
    parser.add_argument("--lr_G", type=float, default=0.0001,
                      help="The learning rate of Generator to use for training.")
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
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    
    print('Dataset Loaded.')


    print('Model Loading...')
    mnist_dim = 784
    G = Generator_BN(g_output_dim = mnist_dim).cuda()
    D = Discriminator_SA(mnist_dim).cuda()

    # model = DataParallel(model).cuda()
    print('Model loaded.')
    # Optimizer 

    G_optimizer = optim.Adam(G.parameters(), lr=args.lr_G, betas=(0.0, 0.9))
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr_D, betas=(0.0, 0.9))

    
    # Boucle d'entraînement principale
    for epoch in trange(args.epochs, desc="Training SAGAN"):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim).cuda()

            # Entraînement du Discriminateur avec Hinge Loss
            D_train_SA(x, G, D, D_optimizer)

            # Entraînement du Générateur avec Hinge Loss
            G_train_SA(x, G, D, G_optimizer)

        # Sauvegarde des modèles tous les 5 epochs
        if epoch % 5 == 0:
            save_models(G, D, 'checkpoints', epoch)
                
    print('Training done')

        
