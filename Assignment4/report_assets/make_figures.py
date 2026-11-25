"""Build the report figures from the notebook outputs and the saved checkpoints."""
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
from torchvision import transforms
from torchvision.utils import make_grid

ASSETS = Path(__file__).resolve().parent
ROOT = ASSETS.parent
OUT = ROOT / 'outputs'
SUB = ROOT / 'submission'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PAD = 2

NOTEBOOKS = {
    'diffusion': SUB / 'Part 1 - Diffusion Model.ipynb',
    'flow_matching': SUB / 'Part 2 - Flow Matching.ipynb',
    'cyclegan': SUB / 'Part 3 - CycleGan.ipynb',
}
# Must match network_cfg in the Part 1 and Part 2 notebooks.
DIT_CFG = dict(input_size=32, patch_size=4, in_channels=1, hidden_size=256,
               depth=6, num_heads=8, mlp_ratio=4.0, num_classes=10)
DIT_CELLS = ['def get_2d_sincos_pos_embed(', 'class TimestepEmbedder', 'class DiTBlock', 'class DiT(']
# Earlier run of the provided Part 1 loop (constant learning rate); its notebook output was overwritten.
FID_DIFFUSION_CONSTANT_LR = 40.66

BLUE, ORANGE, GREY = '#4c72b0', '#dd8452', '#bbbbbb'
plt.rcParams.update({
    'font.size': 7, 'axes.titlesize': 8, 'axes.titleweight': 'bold',
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})


def code_cells(name):
    nb = json.load(open(NOTEBOOKS[name]))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


def stream_text(name):
    nb = json.load(open(NOTEBOOKS[name]))
    return ''.join(''.join(o['text']) for c in nb['cells'] if c['cell_type'] == 'code'
                   for o in c.get('outputs', []) if o.get('output_type') == 'stream')


def notebook_namespace(name, markers):
    """Run the notebook's own definition cells, so figures use the submitted code."""
    ns = dict(torch=torch, nn=nn, F=F, np=np, math=math, Attention=Attention, Mlp=Mlp, PatchEmbed=PatchEmbed)
    cells = code_cells(name)
    for marker in markers:
        exec(next(src for src in cells if marker in src), ns)
    return ns


def show_grid(ax, images, ncol, title=None, row_labels=(), col_labels=()):
    size = images.shape[-1]
    grid = make_grid(images.detach().cpu().clamp(0, 1), nrow=ncol, padding=PAD, pad_value=1.0)
    ax.imshow(grid.permute(1, 2, 0).numpy(), interpolation='nearest')
    centres = [PAD + i * (size + PAD) + size / 2 for i in range(max(len(row_labels), len(col_labels)))]
    ax.set_yticks(centres[:len(row_labels)], row_labels)
    ax.set_xticks(centres[:len(col_labels)], col_labels)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title)


def save(fig, name):
    fig.savefig(ASSETS / name)
    plt.close(fig)


def make_metrics_figure():
    text = {name: stream_text(name) for name in NOTEBOOKS}
    fid = {name: float(re.search(r'FID Score: ([\d.]+)', text[name]).group(1)) for name in ('diffusion', 'flow_matching')}
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9), gridspec_kw={'width_ratios': [1.3, 1, 1, 1.35]})

    ax = axes[0]
    bars = ax.bar(['Diffusion\nconst. LR', 'Diffusion\ncosine LR', 'Flow\nmatching'],
                  [FID_DIFFUSION_CONSTANT_LR, fid['diffusion'], fid['flow_matching']],
                  color=[GREY, BLUE, ORANGE], width=0.65)
    bars[0].set_hatch('///')
    ax.bar_label(bars, fmt='%.2f', padding=1)
    ax.axhline(30, color='black', linestyle='--', linewidth=0.8)
    ax.text(2.4, 30.6, 'target < 30', ha='right', va='bottom', fontsize=6)
    ax.tick_params(axis='x', labelsize=5.5)
    ax.set_ylim(0, 48)
    ax.set_title('FID at 8 steps (lower is better)')

    for ax, name, title, color in ((axes[1], 'diffusion', 'Diffusion: noise MSE', BLUE),
                                   (axes[2], 'flow_matching', 'Flow matching: velocity MSE', ORANGE)):
        points = re.findall(r'Epoch (\d+)/\d+ \| Noise prediction MSE: ([\d.]+)', text[name])
        ax.plot([int(e) for e, _ in points], [float(v) for _, v in points], marker='o', markersize=3, color=color)
        ax.set_title(title)

    cyc = np.array(re.findall(r'Epoch (\d+)/\d+ \| G: ([\d.]+) \| D_H: ([\d.]+) \| D_Z: ([\d.]+) '
                              r'\| Cycle: ([\d.]+) \| Identity: ([\d.]+)', text['cyclegan']), dtype=float)
    for col, label in enumerate(('Generator', 'D horse', 'D zebra', 'Cycle', 'Identity'), start=1):
        axes[3].plot(cyc[:, 0], cyc[:, col], label=label, linewidth=1)
    axes[3].legend(fontsize=5.5, ncol=2, frameon=False, loc='center right')
    axes[3].set_title('CycleGAN: epoch losses')

    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axes[1:]:
        ax.set_xlabel('Epoch')
        ax.set_xticks([1, 5, 10, 15, 20])
    fig.tight_layout(w_pad=1.0)
    save(fig, 'fig_metrics.png')


def make_samples_figure():
    labels = torch.arange(10).repeat(2)
    runs = (
        ('diffusion', 'part1_diffusion_checkpoint_020.pt', ['class NoiseSchedule', 'def sample_ddim('], 50, 'Diffusion (DDIM)'),
        ('flow_matching', 'part2_flow_matching_checkpoint_020.pt', ['def sample_flow_matching('], 20, 'Flow matching (Euler)'),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 1.7))
    for ax, (name, ckpt, sampler_cells, more_steps, title) in zip(axes, runs):
        ns = notebook_namespace(name, DIT_CELLS + sampler_cells)
        model = ns['DiT'](**DIT_CFG).to(DEVICE)
        model.load_state_dict(torch.load(SUB / ckpt, map_location=DEVICE, weights_only=True)['model'])

        def sample(steps):
            torch.manual_seed(0)  # same starting noise for both step counts
            if name == 'diffusion':
                schedule = ns['NoiseSchedule'](num_steps=1000, device=DEVICE)
                return ns['sample_ddim'](model, schedule, labels, num_steps=steps, device=DEVICE)
            return ns['sample_flow_matching'](model, labels, num_steps=steps, device=DEVICE)

        few, many = sample(8), sample(more_steps)
        rows = torch.cat([few[:10], many[:10], few[10:], many[10:]])
        show_grid(ax, (rows + 1) / 2, 10, title, row_labels=['8 steps', f'{more_steps} steps'] * 2)
    save(fig, 'fig_samples.png')


def cyclegan_generators(checkpoint):
    ns = notebook_namespace('cyclegan', ['class Generator'])
    ckpt = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    generators = []
    for key in ('G_H2Z', 'G_Z2H'):
        G = ns['Generator']().to(DEVICE)
        G.load_state_dict(ckpt[key])
        generators.append(G.eval())
    return generators


def test_images(folder, n):
    """First n held-out test images in filename order, preprocessed as in the notebook."""
    tf = transforms.Compose([
        transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    paths = sorted((ROOT / 'datasets' / 'horse2zebra' / folder).glob('*.jpg'))[:n]
    return torch.stack([tf(Image.open(p).convert('RGB')) for p in paths]).to(DEVICE)


def make_cyclegan_figure(n=8):
    G_H2Z, G_Z2H = cyclegan_generators(SUB / 'part3_cyclegan_checkpoint_020.pt')
    horses, zebras = test_images('testA', n), test_images('testB', n)
    with torch.no_grad():
        rows = torch.cat([horses, G_H2Z(horses), zebras, G_Z2H(zebras)])
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    show_grid(ax, rows * 0.5 + 0.5, n, row_labels=['Real horse', 'Horse → zebra', 'Real zebra', 'Zebra → horse'])
    save(fig, 'fig_cyclegan.png')


def make_cyclegan_epochs_figure(epochs=(1, 5, 10, 15, 20), n=2):
    horses, zebras = test_images('testA', n), test_images('testB', n)
    fake_zebras, fake_horses = [horses], [zebras]
    for epoch in epochs:
        G_H2Z, G_Z2H = cyclegan_generators(OUT / 'cyclegan' / 'checkpoints' / f'checkpoint_{epoch:03d}.pt')
        with torch.no_grad():
            fake_zebras.append(G_H2Z(horses))
            fake_horses.append(G_Z2H(zebras))
    columns = ['Real'] + [f'Ep. {e}' for e in epochs]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 1.5))
    for ax, outputs, title in ((axes[0], fake_zebras, 'Horse → zebra'), (axes[1], fake_horses, 'Zebra → horse')):
        # One row per test image, one column per epoch.
        rows = torch.stack(outputs, dim=1).flatten(0, 1)
        show_grid(ax, rows * 0.5 + 0.5, len(columns), title, col_labels=columns)
    save(fig, 'fig_cyclegan_epochs.png')


if __name__ == '__main__':
    make_metrics_figure()
    make_samples_figure()
    make_cyclegan_figure()
    make_cyclegan_epochs_figure()
