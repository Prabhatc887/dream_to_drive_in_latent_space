import gymnasium as gym
import torch
import torch.nn as nn
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Parameters
latent_dim = 512
action_dim = 3
mdn_dim = 256  # only needed if controller input expects context; here we use zero context
sequence_len = 16  # optional

input_dim = latent_dim + mdn_dim

class VAE(nn.Module):
    def __init__(self, latent_dim=512):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),# 8x8 -> 4x4
            nn.ReLU()
        )

        # Latent space (ONLY mu is used)
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, latent_dim)

        # Decoder fully connected
        self.fc_dec = nn.Linear(latent_dim, 256*4*4)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),     # 32x32 -> 64x64
            nn.Sigmoid()  # output in [0,1]
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        return mu

    def forward(self, x):
        return self.encode(x)


vae = VAE(latent_dim=512).to(device)
vae.load_state_dict(torch.load("vae_carracing.pth", map_location=device), strict=False)
vae.eval()

class Controller(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, action_dim)

    def forward(self, x):
        return torch.tanh(self.linear(x))

flat_params = np.load("controller_params_gen_25.npy", allow_pickle=True)
flat_params = torch.tensor(flat_params, dtype=torch.float32)

def set_controller_params(controller, flat_params):
    weight_shape = controller.linear.weight.shape
    bias_shape = controller.linear.bias.shape
    num_weight = np.prod(weight_shape)
    weight = flat_params[:num_weight].reshape(weight_shape)
    bias = flat_params[num_weight:num_weight + np.prod(bias_shape)]
    with torch.no_grad():
        controller.linear.weight.copy_(weight)
        controller.linear.bias.copy_(bias)


controller = Controller(input_dim, action_dim).to(device)
set_controller_params(controller, flat_params)
controller.eval()

env = gym.make("CarRacing-v3", render_mode="human") 
obs, _ = env.reset()

done = False
z_seq = []

while not done:

    obs_resized = cv2.resize(obs, (64, 64))
    obs_tensor = torch.tensor(obs_resized, dtype=torch.float32, device=device)
    obs_tensor = obs_tensor.permute(2,0,1).unsqueeze(0) / 255.0

    with torch.no_grad():
        mu = vae.encode(obs_tensor)
    z = mu

    context = torch.zeros((1, mdn_dim), device=device)

    controller_input = torch.cat([context, z], dim=1)
    with torch.no_grad():
        action = controller(controller_input).squeeze(0)
    action_np = action.cpu().numpy()

    obs, reward, terminated, truncated, _ = env.step(action_np)
    done = terminated or truncated

env.close()
print("Testing complete.")