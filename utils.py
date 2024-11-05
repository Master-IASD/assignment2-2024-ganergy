import torch
import os



def D_train(x, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()

    D_output =  D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).cuda()
    y = torch.ones(x.shape[0], 1).cuda()
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()

def WGAN_D_train(x, G, D, D_optimizer, clip_value=0.01):
    #=======================Train the discriminator=======================#

    D_optimizer.zero_grad()

    # train discriminator on real
    x_real = x
    x_real = x_real.cuda()

  
    # train discriminator on fake
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake = G(z).detach()

    # gradient backprop & optimize ONLY D's parameters
    D_loss = -torch.mean(D(x_real)) + torch.mean(D(x_fake))
    D_loss.backward()
    D_optimizer.step()

    # Clip weights of discriminator
    for p in D.parameters():
        p.data.clamp_(-clip_value, clip_value)

    return  D_loss.data.item()


def WGAN_G_train(x, G, D, G_optimizer):
    #=======================Train the generator=======================#
    G_optimizer.zero_grad()

    z = torch.randn(x.shape[0], 100).cuda()

    G_output = G(z)

    G_loss = -torch.mean(D(G_output))

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()

# Fonctions de Perte avec Hinge Loss
def D_loss_function_SA(real_output, fake_output):
    real_loss = torch.mean(torch.relu(1.0 - real_output))
    fake_loss = torch.mean(torch.relu(1.0 + fake_output))
    return real_loss + fake_loss

def G_loss_function_SA(fake_output):
    return -torch.mean(fake_output)

# Fonction d'entraînement du Discriminateur
def D_train_SA(x, G, D, D_optimizer):
    D.zero_grad()
    x_real = x.cuda()
    real_output = D(x_real)
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake = G(z)
    fake_output = D(x_fake)
    D_loss = D_loss_function_SA(real_output, fake_output)
    D_loss.backward()
    D_optimizer.step()
    return D_loss.item()

# Fonction d'entraînement du Générateur
def G_train_SA(x, G, D, G_optimizer):
    G.zero_grad()
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake = G(z)
    fake_output = D(x_fake)
    G_loss = G_loss_function_SA(fake_output)
    G_loss.backward()
    G_optimizer.step()
    return G_loss.item()

# def save_models(G, D, folder):
#     torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
#     torch.save(D.state_dict(), os.path.join(folder,'D.pth'))

def save_models(G, D, folder, epoch):
    torch.save(G.state_dict(), os.path.join(folder, f'G_epoch_{epoch}.pth'))
    torch.save(D.state_dict(), os.path.join(folder, f'D_epoch_{epoch}.pth'))

def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'))#, map_location=torch.device('cpu')) #(when using cpu instead of gpu)
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G
