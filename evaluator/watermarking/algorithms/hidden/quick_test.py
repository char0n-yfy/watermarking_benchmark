import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

import utils
from model.hidden import Hidden
from noise_layers.noiser import Noiser


def compute_psnr(x, y, max_val=1.0):
    # x, y in [0,1]
    mse = torch.mean((x - y) ** 2)
    if mse.item() == 0:
        return float("inf")
    psnr = 10.0 * torch.log10((max_val ** 2) / mse)
    return float(psnr.item())


def main():
    # Select device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Project root
    root = os.path.dirname(os.path.abspath(__file__))

    # Fixed selections: existing experiment options + checkpoint and the root image file
    options_file = os.path.join(root, 'experiments', 'combined-noise', 'options-and-config.pickle')
    checkpoint_file = os.path.join(root, 'experiments', 'combined-noise', 'checkpoints', 'combined-noise--epoch-400.pyt')
    image_file = os.path.join(root, 'img.jpg')

    # Load configs and model
    train_options, hidden_config, noise_config = utils.load_options(options_file)
    noiser = Noiser(noise_config, device)
    model = Hidden(hidden_config, device, noiser, tb_logger=None)

    checkpoint = torch.load(checkpoint_file, map_location=device)
    utils.model_from_checkpoint(model, checkpoint)

    # Load and prepare image
    img = Image.open(image_file).convert('RGB')
    target_size = (hidden_config.H, hidden_config.W)
    # Resize to the network input directly for simplicity
    img_resized = img.resize((target_size[1], target_size[0]))
    img_tensor = TF.to_tensor(img_resized).to(device)  # [0,1]
    img_tensor = img_tensor * 2 - 1  # to [-1,1]
    img_tensor = img_tensor.unsqueeze(0)  # add batch dim

    # Random message
    msg = torch.tensor(
        np.random.choice([0, 1], (img_tensor.shape[0], hidden_config.message_length)),
        dtype=torch.float32,
        device=device,
    )

    # Forward encoder only (through encoder_decoder) and compute PSNR on encoded vs original
    model.encoder_decoder.eval()
    with torch.no_grad():
        encoded_images, _, _ = model.encoder_decoder(img_tensor, msg)

    # Convert to [0,1] for PSNR
    original_01 = (img_tensor + 1) / 2
    encoded_01 = (encoded_images + 1) / 2
    psnr = compute_psnr(original_01, encoded_01, max_val=1.0)

    # Output
    print(f"使用权重: {checkpoint_file}")
    print(f"测试图片: {image_file}")
    print(f"嵌入容量: {hidden_config.message_length} bits")
    print(f"嵌入前后PSNR: {psnr:.2f} dB")


if __name__ == '__main__':
    main()

