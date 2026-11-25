import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image as PILImage
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, ListFlowable, ListItem

ROOT = Path('/home/laurie/CMPUT328/Assignment4')
A = ROOT / 'report_assets'
OUT = ROOT / 'outputs'
SUB = ROOT / 'submission'


def sample_panel(method, rows=5):
    tiles = [[PILImage.open(OUT / method / 'generated' / f'class_{c}_sample_{r}.png').convert('RGB')
              for c in range(10)] for r in range(rows)]
    s, pad = tiles[0][0].width, 2
    panel = PILImage.new('RGB', (10 * (s + pad) + pad, rows * (s + pad) + pad), 'white')
    for r in range(rows):
        for c in range(10):
            panel.paste(tiles[r][c], (pad + c * (s + pad), pad + r * (s + pad)))
    return panel.resize((panel.width * 3, panel.height * 3), PILImage.NEAREST)


def make_samples_figure():
    left, right = sample_panel('diffusion'), sample_panel('flow_matching')
    gap = 48
    fig = PILImage.new('RGB', (left.width + gap + right.width, left.height), 'white')
    fig.paste(left, (0, 0))
    fig.paste(right, (left.width + gap, 0))
    fig.save(A / 'fig_samples.png')


def make_cyclegan_figure(n=10):
    # The architecture is taken from the notebook cell, so the checkpoint is loaded into the submitted model definition.
    nb = json.load(open(SUB / 'Part 3 - CycleGan.ipynb'))
    sources = (''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    ns = {'torch': torch, 'nn': nn}
    exec(next(src for src in sources if 'class Generator' in src), ns)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(SUB / 'part3_cyclegan_checkpoint_020.pt', map_location=device, weights_only=True)
    tf = transforms.Compose([
        transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    rows = []
    for folder, key in (('testA', 'G_H2Z'), ('testB', 'G_Z2H')):
        G = ns['Generator']().to(device)
        G.load_state_dict(ckpt[key])
        G.eval()
        paths = sorted((ROOT / 'datasets' / 'horse2zebra' / folder).glob('*.jpg'))[:n]
        x = torch.stack([tf(PILImage.open(p).convert('RGB')) for p in paths]).to(device)
        with torch.no_grad():
            rows += [x, G(x)]
    save_image(make_grid(torch.cat(rows) * 0.5 + 0.5, nrow=n, padding=2, pad_value=1), A / 'fig_cyclegan.png')


make_samples_figure()
make_cyclegan_figure()

body = ParagraphStyle('b', fontName='Times-Roman', fontSize=9, leading=10.6, spaceAfter=3)
title = ParagraphStyle('t', fontName='Times-Bold', fontSize=15, leading=18, alignment=TA_CENTER)
sub = ParagraphStyle('s', fontName='Times-Roman', fontSize=11, leading=13, alignment=TA_CENTER)
auth = ParagraphStyle('a', fontName='Times-Roman', fontSize=10, leading=12, alignment=TA_CENTER, spaceAfter=4)
h = ParagraphStyle('h', fontName='Times-Bold', fontSize=11, leading=13, spaceBefore=4, spaceAfter=2)
cap = ParagraphStyle('c', fontName='Times-Italic', fontSize=8, leading=9.5, alignment=TA_CENTER, spaceAfter=3)
cell = ParagraphStyle('cell', fontName='Times-Roman', fontSize=8, leading=9)
cellb = ParagraphStyle('cellb', parent=cell, fontName='Times-Bold')


def P(t, s=body):
    return Paragraph(t, s)


def table(rows, widths):
    data = [[Paragraph(str(c), cellb if r == 0 else cell) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign='CENTER')
    t.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 0.8, colors.black), ('LINEBELOW', (0, 0), (-1, 0), 0.4, colors.black),
        ('LINEBELOW', (0, -1), (-1, -1), 0.8, colors.black), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 1.2), ('BOTTOMPADDING', (0, 0), (-1, -1), 1.2),
        ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3)]))
    return t


def fig(name, width):
    w, hgt = PILImage.open(A / name).size
    return Image(str(A / name), width=width, height=width * hgt / w)


W = 7.0 * inch
doc = SimpleDocTemplate(str(SUB / 'report.pdf'), pagesize=letter,
                        leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.55 * inch, bottomMargin=0.5 * inch,
                        title='CMPUT 328 Assignment 4 Report', author='Laurie Gantuangco')
s = []
s += [P('CMPUT 328 Assignment 4', title), P('Generative Models: Diffusion, Flow Matching and CycleGAN', sub),
      P('Laurie Gantuangco &nbsp;|&nbsp; Student ID: 1771419 &nbsp;|&nbsp; November 2026', auth)]

s.append(P('1. Results Summary', h))
s.append(P('<b>Key takeaway:</b> Both FashionMNIST generators pass the FID &lt; 30 target with 8 sampling steps (diffusion 26.22, flow matching 19.87), '
           'and the CycleGAN visibly translates horses into zebras; zebra-to-horse is weak. Each model was trained once for 20 epochs with no fixed seed, '
           'so every number below is a single-run value.'))
t1 = [['Part', 'Model (parameters)', 'Training objective', 'Evaluation', 'Final train loss', 'Result (target)'],
      ['1. Diffusion', 'DiT, 6 blocks, width 256 (7.38M)', 'Noise-prediction MSE, 1000-step linear &beta; schedule', 'DDIM, zero variance, 8 steps', '0.0248', 'FID 26.22 (&lt; 30)'],
      ['2. Flow matching', 'Same DiT (7.38M)', 'Velocity MSE on straight noise-to-data paths', 'Euler integration, 8 steps', '0.1458', 'FID 19.87 (&lt; 30)'],
      ['3. CycleGAN', '2 ResNet generators (7.83M each), 2 PatchGAN discriminators (2.76M each)', 'LSGAN + 2 &times; cycle L1 + 1 &times; identity L1',
       'Visual, held-out test images', 'G 2.116; cycle 0.435', 'Horse&rarr;zebra visible; zebra&rarr;horse weak (one direction)']]
s.append(table(t1, [1.0 * inch, 1.55 * inch, 1.55 * inch, 1.1 * inch, 0.8 * inch, 1.0 * inch]))
s.append(P('Table 1. Final results. FID uses the provided cell: 9,984 test images (312 batches of 32) against the same number of images generated for the same labels.', cap))
s.append(P('<b>Submitted files.</b> Three notebooks with outputs, and the epoch-20 checkpoint of each run: <font face="Courier" size="8">part1_diffusion_checkpoint_020.pt</font> and '
           '<font face="Courier" size="8">part2_flow_matching_checkpoint_020.pt</font> (DiT weights under key <font face="Courier" size="8">model</font>), and '
           '<font face="Courier" size="8">part3_cyclegan_checkpoint_020.pt</font> (keys <font face="Courier" size="8">G_H2Z, G_Z2H, D_H, D_Z</font>). '
           '<b>Departures from the provided notebooks:</b> they were run on a local RTX 4070 Ti SUPER instead of Colab, so the Drive-mount cell was removed, '
           '<font face="Courier" size="8">project_path</font> is a local folder and <font face="Courier" size="8">!pip</font> became <font face="Courier" size="8">%pip</font>; '
           'Part 1 also adds a learning-rate schedule to the training loop (Section 2).'))

s.append(P('2. Part 1 &mdash; Diffusion (DiT, DDPM objective, DDIM sampler)', h))
s.append(P('<b>Key takeaway:</b> The implementation trained correctly with the provided loop, but passing the 8-step FID target needed a learning-rate decay. '
           '<b>Implementation.</b> The timestep embedder maps t to 256 sinusoidal features (cosines and sines at geometrically spaced frequencies) and then through Linear&ndash;SiLU&ndash;Linear; '
           'the label embedder is an embedding table, and the two are added to form the conditioning vector c. Each DiT block computes '
           'x &larr; x + gate &middot; f(modulate(norm(x), shift, scale)) for the attention branch and again for the MLP branch, with shift, scale and gate predicted from c '
           '(adaLN-Zero: the gates start at zero, so every block starts as the identity). The noise schedule uses &beta; linear from 10<super>-4</super> to 0.02, &alpha; = 1 &minus; &beta; and '
           'a<sub>t</sub> = the cumulative product of &alpha;, and corrupts images in one step: x<sub>t</sub> = &radic;a<sub>t</sub> x<sub>0</sub> + &radic;(1 &minus; a<sub>t</sub>) &epsilon;. '
           'The DDIM sampler visits evenly spaced timesteps from 999 to 0; at each one it predicts the noise, forms the clean-image estimate '
           '(x<sub>t</sub> &minus; &radic;(1 &minus; a<sub>t</sub>) &epsilon;) / &radic;a<sub>t</sub>, and re-noises that estimate to the next timestep with the same predicted noise and no fresh noise '
           '(a = 1 after the last step).'))
s.append(P('<b>Training and result.</b> 20 epochs of 1,875 iterations (batch 32, AdamW, initial learning rate 3&times;10<super>-4</super>) took about 5 minutes. '
           'Epoch-average noise-prediction MSE fell from 0.0678 to 0.0248, and the FID at 8 DDIM steps is 26.22. '
           '<b>Change to the provided loop.</b> With the loop exactly as given (constant learning rate), an earlier run scored FID 40.66 at 8 steps, which fails the target. '
           'The same model scored about 23.5 at 50 steps, and other 8-step timestep spacings or clipping the clean-image estimate only moved it between 35 and 44, '
           'so the sampler was not the cause. Adding a cosine decay of the learning rate to zero over training (the only change; the architecture and FID cell are untouched) gave 26.22. '
           'Those exploratory runs were overwritten and are not in the submitted notebook; only the final run is.'))

s.append(P('3. Part 2 &mdash; Flow Matching (same DiT, Euler sampler)', h))
s.append(P('<b>Key takeaway:</b> With the same network and the same 8 function evaluations, flow matching reached a lower FID (19.87) with the provided loop unchanged. '
           '<b>Implementation.</b> Each training step draws Gaussian noise x<sub>0</sub> and a time t uniform on [0, 1], forms x<sub>t</sub> = (1 &minus; t) x<sub>0</sub> + t x<sub>1</sub> '
           'for a data image x<sub>1</sub>, and regresses the model output onto the target velocity x<sub>1</sub> &minus; x<sub>0</sub> with MSE; t is multiplied by 1000 to match the DiT time-embedding scale. '
           'Sampling starts from noise and takes N forward Euler steps of size 1/N, x &larr; x + (1/N) v(x, t<sub>k</sub>) with t<sub>k</sub> = k/N for k = 0, ..., N &minus; 1. '
           '<b>Result.</b> Epoch-average velocity MSE fell from 0.2875 to 0.1458 in about 5 minutes, and the FID at 8 Euler steps is 19.87. '
           'A plausible reason for the gap to diffusion is that straight paths have a constant target velocity, so coarse Euler steps lose less than 8 DDIM steps on a curved diffusion trajectory, '
           'but I did not test this, and it is not a controlled comparison: these are single runs with different learning-rate schedules (constant here, cosine in Part 1).'))
s.append(fig('fig_samples.png', W))
s.append(P('Figure 1. Samples saved by the notebooks from the final checkpoints. Left: diffusion, 50 DDIM steps. Right: flow matching, 20 Euler steps. '
           'Columns are classes 0&ndash;9 (T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot); rows are different noise samples.', cap))

s.append(P('4. Part 3 &mdash; CycleGAN (horse2zebra)', h))
s.append(P('<b>Key takeaway:</b> After 20 epochs the horse-to-zebra generator clearly paints stripes on held-out horses, with strong artifacts; zebra-to-horse is not convincing. '
           '<b>Implementation.</b> The GAN criterion is MSE on the discriminator&rsquo;s raw patch scores (least-squares GAN, since the discriminator has no sigmoid); the cycle and identity criteria are L1. '
           'Identity loss: each generator is given a real image already in its output domain and penalized for changing it. '
           'For each direction the generator translates the real image, the other domain&rsquo;s discriminator scores the fake against a target of 1, '
           'and the fake is translated back and compared with the original (cycle loss). '
           'Each discriminator is trained to score its real images as 1 and replay-buffered fakes as 0. The generator loss is GAN + 2 &times; cycle + 1 &times; identity, the provided weights.'))
s.append(P('<b>Training and result.</b> 1,067 horse and 1,334 zebra images at 128&times;128, batch 1, Adam at 2&times;10<super>-4</super>, 20 epochs of 1,334 iterations, about 19 minutes. '
           'Epoch-average losses moved from epoch 1 to 20 as follows: generator 2.527&rarr;2.116, cycle 0.584&rarr;0.435, identity 0.548&rarr;0.401, '
           'horse discriminator 0.341&rarr;0.212, zebra discriminator 0.344&rarr;0.134. '
           'In Figure 2 the zebra generator puts black-and-white stripes on the animals in most images, but it also tints backgrounds green-yellow, lets stripes bleed into the background, '
           'and leaves horizontal line artifacts near the top edge. The horse generator does not produce a horse coat: stripes stay visible on several animals, '
           'and the whole image is blurred and shifted toward a muddy green-brown. '
           'Likely contributors are the short schedule, the low resolution and the small cycle weight, but I did not test any of them.'))
s.append(fig('fig_cyclegan.png', 6.0 * inch))
s.append(P('Figure 2. Final checkpoint on the first 10 test images of each domain in filename order (not selected; not used in training). '
           'Rows: real horses, horse&rarr;zebra output, real zebras, zebra&rarr;horse output. '
           'Produced for this report by loading the submitted checkpoint; the notebook itself saves one training example per epoch.', cap))

s.append(P('Key Takeaways', h))
items = ['Diffusion reached FID 26.22 at 8 DDIM steps (target &lt; 30) only after a cosine learning-rate decay was added; an earlier constant-rate run scored 40.66.',
         'Flow matching reached FID 19.87 at 8 Euler steps with the same DiT and the provided loop unchanged.',
         'The CycleGAN translates horses to zebras visibly but with background artifacts; zebra-to-horse is weak after 20 epochs.',
         'All results are single unseeded runs, so differences of a few FID points should not be over-read.']
s.append(ListFlowable([ListItem(P(i), leftIndent=10) for i in items], bulletType='bullet', bulletFontSize=6, leftIndent=10, bulletOffsetY=-1))

s.append(P('AI-Assistance Disclosure', h))
s.append(P('GitHub Copilot (agent mode in VS Code) implemented the TODO sections and their explanatory comments, ran the training and evaluation cells, '
           'ran the exploratory FID experiments described in Section 2, and drafted this report. '
           'Metrics are taken from the notebook outputs except where the text says otherwise; parameter counts were computed from the checkpoints.'))

doc.build(s)
