import torch
import torch.nn
import argparse
import os
import numpy as np
from options import HiDDenConfiguration

import utils
from model.hidden import *
from noise_layers.noiser import Noiser
from PIL import Image
import torchvision.transforms.functional as TF


def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--options-file', '-o', default='options-and-config.pickle', type=str,
                        help='The file where the simulation options are stored.')
    parser.add_argument('--checkpoint-file', '-c', required=True, type=str, help='Model checkpoint file')
    parser.add_argument('--batch-size', '-b', default=12, type=int, help='The batch size.')
    parser.add_argument('--source-image', '-s', required=True, type=str,
                        help='The image to watermark')
    parser.add_argument('--output-image', default=None, type=str,
                        help='Optional path where the watermarked image is saved.')
    parser.add_argument('--report-file', default=None, type=str,
                        help='Optional path where decoded bits and error are written.')
    parser.add_argument('--identity-noise', action='store_true',
                        help='Use identity/no attack noise during this single-image smoke test.')
    parser.add_argument('--message', default=None, type=str,
                        help='Optional fixed bit string. Length must match the trained message length.')
    # parser.add_argument('--times', '-t', default=10, type=int,
    #                     help='Number iterations (insert watermark->extract).')

    args = parser.parse_args()

    train_options, hidden_config, noise_config = utils.load_options(args.options_file)
    if args.identity_noise:
        noise_config = []
    noiser = Noiser(noise_config, device)

    try:
        checkpoint = torch.load(args.checkpoint_file, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint_file, map_location=device)
    hidden_net = Hidden(hidden_config, device, noiser, None)
    utils.model_from_checkpoint(hidden_net, checkpoint)


    image_pil = Image.open(args.source_image).convert('RGB')
    image = randomCrop(np.array(image_pil), hidden_config.H, hidden_config.W)
    image_tensor = TF.to_tensor(image).to(device)
    image_tensor = image_tensor * 2 - 1  # transform from [0, 1] to [-1, 1]
    image_tensor.unsqueeze_(0)

    # for t in range(args.times):
    if args.message is None:
        message = torch.Tensor(np.random.choice([0, 1], (image_tensor.shape[0],
                                                        hidden_config.message_length))).to(device)
    else:
        if len(args.message) != hidden_config.message_length or set(args.message) - {'0', '1'}:
            raise ValueError(f'--message must be a {hidden_config.message_length}-bit 0/1 string.')
        message = torch.Tensor([[int(bit) for bit in args.message]]).to(device)
    losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch([image_tensor, message])
    decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
    message_detached = message.detach().cpu().numpy()
    error = np.mean(np.abs(decoded_rounded - message_detached))
    report = '\n'.join([
        'original: {}'.format(message_detached),
        'decoded : {}'.format(decoded_rounded),
        'error : {:.3f}'.format(error),
    ])
    print(report)
    utils.save_images(image_tensor.cpu(), encoded_images.cpu(), 'test', '.', resize_to=(256, 256))
    if args.output_image is not None:
        encoded_np = utils.tensor_to_image(encoded_images.detach().cpu())[0]
        Image.fromarray(encoded_np).save(args.output_image)
    if args.report_file is not None:
        with open(args.report_file, 'w', encoding='utf-8') as f:
            f.write(report + '\n')

    # bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy()))/(image_tensor.shape[0] * messages.shape[1])



if __name__ == '__main__':
    main()
