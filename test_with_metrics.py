import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from torchvision import transforms
import models.transformer as transformer
import models.StyTR as StyTR
import numpy as np
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('prefix', help="Start of names for files produced.")
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--output', type=str, default='metrics',
                    help='Directory to save the output metrics')
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
parser.add_argument('--decoder_path', type=str, default=f'output2/original_stytr2_01/decoder_iter_160000.pth')
parser.add_argument('--Trans_path', type=str, default=f'output2/original_stytr2_01/transformer_iter_160000.pth')
parser.add_argument('--embedding_path', type=str, default=f'output2/original_stytr2_01/embedding_iter_160000.pth')


parser.add_argument('--style_interpolation_weights', type=str, default="")
parser.add_argument('--a', type=float, default=1.0)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
args = parser.parse_args()

def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))
    print(type(size))
    transform_list = []    
    transform_list.append(transforms.CenterCrop((h,w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    
    transform_list = []   
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

# Advanced options
content_size=512
style_size=512
crop='store_true'
save_ext='.jpg'
output_path=args.output
preserve_color='store_true'
alpha=args.a

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Either --content or --content_dir should be given.
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --style_dir should be given.
if args.style:
    style_paths = [Path(args.style)]    
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

if not os.path.exists(output_path):
    os.mkdir(output_path)


vgg = StyTR.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])

decoder = StyTR.decoder
Trans = transformer.Transformer()
embedding = StyTR.PatchEmbed()

decoder.eval()
Trans.eval()
vgg.eval()
from collections import OrderedDict
new_state_dict = OrderedDict()
state_dict = torch.load(args.decoder_path, map_location='cpu')
for k, v in state_dict.items():
    #namekey = k[7:] # remove `module.`
    namekey = k
    new_state_dict[namekey] = v
decoder.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(args.Trans_path, map_location='cpu')
for k, v in state_dict.items():
    #namekey = k[7:] # remove `module.`
    namekey = k
    new_state_dict[namekey] = v
Trans.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(args.embedding_path, map_location='cpu')
for k, v in state_dict.items():
    #namekey = k[7:] # remove `module.`
    namekey = k
    new_state_dict[namekey] = v
embedding.load_state_dict(new_state_dict)

network = StyTR.StyTrans(vgg,decoder,embedding,Trans,args)
network.eval()
network.to(device)

content_tf = test_transform(content_size, crop)
style_tf = test_transform(style_size, crop)
lpips_loss_fn = lpips.LPIPS(net='alex').to(device)

# Prepare a DataFrame to store results
results = []

for content_path in content_paths:
    for i, style_path in enumerate(style_paths):
        content_tf1 = content_transform()       
        content = content_tf(Image.open(content_path).convert("RGB"))
        h,w,c = np.shape(content)    
        style_tf1 = style_transform(h,w)
        style = style_tf(Image.open(style_path).convert("RGB"))

        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
        
        with torch.no_grad():
            output, loss_c, loss_s, l_identity1, l_identity2 = network(content, style)
        
        # Convert output and content to numpy arrays for SSIM/PSNR
        output_img = output.squeeze(0).cpu().permute(1,2,0).numpy()
        output_img = np.clip(output_img, 0, 1)  # ensure 0-1 range
        content_img = content.squeeze(0).cpu().permute(1,2,0).numpy()
        content_img = np.clip(content_img, 0, 1)

        # SSIM (multichannel=True for RGB)
        ssim_value = ssim(content_img, output_img, multichannel=True, data_range=1.0, channel_axis=2, win_size=3)
        
        # PSNR
        psnr_value = psnr(content_img, output_img, data_range=1.0)
        
        # LPIPS
        transform = lambda x: (x * 2 - 1)  # convert [0,1] to [-1,1]
        output_tensor = transform(output).to(device)
        content_tensor = transform(content).to(device)
        lpips_value = lpips_loss_fn(output_tensor, content_tensor).item()

        # Store results
        results.append({
            "content": basename(content_path),
            "style": basename(style_path),
            "loss_c": loss_c.sum().item(),
            "loss_s": loss_s.sum().item(),
            "ssim": ssim_value,
            "psnr": psnr_value,
            "lpips": lpips_value
        })

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Save results
df_results.to_csv(f"metrics/evaluation_metrics_{args.prefix}.csv", index=False)
print(df_results.head())
print(f"Saved at: metrics/evaluation_metrics_{args.prefix}.csv")
