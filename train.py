import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import pathlib
import models.transformer as transformer
import models.StyTRv2 as StyTR
from models.discriminator import Discriminator
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image
import logging 


parser = argparse.ArgumentParser()
parser.add_argument('prefix', help="Start of names for files produced.")
parser.add_argument('--content_dir', default='./datasets/train2014', type=str)
parser.add_argument('--style_dir', default='./datasets/style', type=str)
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
parser.add_argument('--save_dir', default='./experiments', type=str)
parser.add_argument('--log_dir', default='./logs', type=str)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=7.0)
parser.add_argument('--n_threads', type=int, default=6)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--position_embedding', default='sine', choices=('sine', 'learned'))
parser.add_argument('--hidden_dim', default=512, type=int)

args = parser.parse_args()

logging.basicConfig(
    filename=f"{args.prefix}.log",
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s', 
)
logging.info(f"Run the strategy with 25,000 steps")

def train_transform():
    return transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ])


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root, self.path[0])):
            self.paths = [os.path.join(self.root, d, f)
                          for d in os.listdir(self.root)
                          for f in os.listdir(os.path.join(self.root, d))]
        else:
            self.paths = list(pathlib.Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        return self.transform(img)

    def __len__(self):
        return len(self.paths)


def adjust_learning_rate(optimizer, iteration_count):
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(optimizer, iteration_count):
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def d_loss_fn(real, fake):
    bce = nn.BCEWithLogitsLoss()
    real_loss = bce(real, torch.ones_like(real))
    fake_loss = bce(fake, torch.zeros_like(fake))
    return real_loss + fake_loss


def g_loss_fn(fake):
    bce = nn.BCEWithLogitsLoss()
    return bce(fake, torch.ones_like(fake))

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
USE_CUDA = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(f"{args.save_dir}/test", exist_ok=True)
print("CUDA device count:", torch.cuda.device_count())
print("CUDA visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES"))

writer = SummaryWriter(log_dir=args.log_dir)

vgg = StyTR.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])
vgg.eval() 
decoder = StyTR.decoder
embedding = StyTR.PatchEmbed()

Trans = transformer.Transformer()
with torch.no_grad():
    network = StyTR.StyTrans(vgg,decoder,embedding, Trans,args)
network.train()

network.to(device)
# network = nn.DataParallel(network, device_ids=[0])
network.train()

D = Discriminator().to(device)
# D = nn.DataParallel(D)
D.train()

optimizer = torch.optim.Adam([
    {'params': network.transformer.parameters()},
    {'params': network.decode.parameters()},
    {'params': network.embedding.parameters()},
], lr=args.lr)

optimizer_D = torch.optim.Adam(D.parameters(), lr=2e-4)
lambda_gan = 0.0001

content_dataset = FlatFolderDataset(args.content_dir, train_transform())
style_dataset = FlatFolderDataset(args.style_dir, train_transform())

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))

style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))
print("decoder is:", type(StyTR.decoder))
print("decoder repr:", StyTR.decoder)

for i in tqdm(range(args.max_iter)):
    lambda_gan = min(0.01, i / 25000 * 0.01)
    if i < 1e4:
        warmup_learning_rate(optimizer, i)
    else:
        adjust_learning_rate(optimizer, i)

    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)

    out, loss_c, loss_s, l_identity1, l_identity2 = network(content_images, style_images)
    # with torch.no_grad():
    fake_images = out.detach()
    real_images = content_images

    pred_real = D(real_images)
    pred_fake = D(fake_images)
    loss_D = d_loss_fn(pred_real, pred_fake)

    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()

    pred_fake_for_G = D(out)
    loss_G_gan = lambda_gan * g_loss_fn(pred_fake_for_G)

    loss_c = args.content_weight * loss_c
    style_weight = min(10.0, 2.0 + i / 5000 * 8.0)
    loss_s = style_weight * loss_s

    loss = loss_c + loss_s + loss_G_gan + (l_identity1 * 70) + (l_identity2 * 1)
    # logging.info(f'loss = {loss.sum().cpu().detach().numpy()},"-content:", {loss_c.sum().cpu().detach().numpy()},"-style:",{loss_s.sum().cpu().detach().numpy()}, "-l1:", {l_identity1.sum().cpu().detach().numpy()},"-l2:",{l_identity2.sum().cpu().detach().numpy()}')
    logging.info(
        'i = {}, loss = {} - content: {} - style: {} - l1: {} - l2: {} - gan: {}'.format(
            i,
            loss.sum().cpu().detach().numpy(),
            loss_c.sum().cpu().detach().numpy(),
            loss_s.sum().cpu().detach().numpy(),
            l_identity1.sum().cpu().detach().numpy(),
            l_identity2.sum().cpu().detach().numpy(),
            loss_G_gan.sum().cpu().detach().numpy()
        )
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 1000 == 0 or i in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
        output_name = f'{args.save_dir}/test/{i}.jpg'
        with torch.no_grad():
            out_vis = torch.clamp(out, 0, 1).detach().cpu()
            save_image(torch.cat((content_images.cpu(), style_images.cpu(), out_vis), 0), output_name)

    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)
    writer.add_scalar('loss_identity1', l_identity1.item(), i + 1)
    writer.add_scalar('loss_identity2', l_identity2.item(), i + 1)
    writer.add_scalar('loss_gan', loss_G_gan.item(), i + 1)
    writer.add_scalar('loss_d', loss_D.item(), i + 1)
    writer.add_scalar('total_loss', loss.item(), i + 1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        torch.save(network.transformer.state_dict(), f'{args.save_dir}/transformer_iter_{i + 1}.pth')
        torch.save(network.decode.state_dict(), f'{args.save_dir}/decoder_iter_{i + 1}.pth')
        torch.save(network.embedding.state_dict(), f'{args.save_dir}/embedding_iter_{i + 1}.pth')

writer.close()
