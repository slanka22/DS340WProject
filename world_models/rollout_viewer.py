import random
import time
from numpy import load, asarray, squeeze
import matplotlib.pyplot as plt
import torch
from models import VAE
from torchvision import transforms
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE

ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    4, 32, 256, 64, 256

# Same
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])

data = load('datasets/trafficenv/thread_0/rollout_0.npz', allow_pickle=True)

num = random.randint(0, 720)

observations = data['observations']
img = observations[num]


print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = VAE(3, LSIZE).to(device)
vae_state = torch.load('world_models/exp_dir/vae/best.tar', map_location={'cuda:0': str(device)})


vae.load_state_dict(vae_state['state_dict'])

for i in range(10):
    num = random.randint(0, 719)

    observations = data['observations']
    img = observations[num]

    imgIn = transform(img).unsqueeze(0).to(device)
    recon, mu, sigma = vae(imgIn.to(device))
    recon = recon.squeeze(0).cpu().detach().numpy()
    # compress image to 64 x 64
    recon = recon.transpose(1, 2, 0)
    imgIn = imgIn.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
    print(mu)


    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(imgIn)
    axes[0].set_title('Original Image')
    axes[1].imshow(recon)
    axes[1].set_title('Reconstructed Image')
    print()
    plt.show()
    plt.savefig('./world_models/views/view_'+str(i))
    plt.close()


print(data['actions'][10])