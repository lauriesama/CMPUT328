# CMPUT 328 Assignment 4

Generative Models: Diffusion, Flow Matching and CycleGAN

Laurie Gantuangco | Student ID: 1771419 | November 2026

## 1. Results Summary

**Key takeaway:** Both FashionMNIST generators pass the FID < 30 target with 8 sampling steps (diffusion 26.22, flow matching 19.87), and the CycleGAN visibly translates horses into zebras; zebra-to-horse is weak. Each model was trained once for 20 epochs with no fixed seed, so every number below is a single-run value.

| Part | Model (parameters) | Training objective | Evaluation | Final train loss | Result (target) |
|---|---|---|---|---|---|
| 1. Diffusion | DiT, 6 blocks, width 256 (7.38M) | Noise-prediction MSE, 1000-step linear β schedule | DDIM, zero variance, 8 steps | 0.0248 | FID 26.22 (< 30) |
| 2. Flow matching | Same DiT (7.38M) | Velocity MSE on straight noise-to-data paths | Euler integration, 8 steps | 0.1458 | FID 19.87 (< 30) |
| 3. CycleGAN | 2 ResNet generators (7.83M each), 2 PatchGAN discriminators (2.76M each) | LSGAN + 2 × cycle L1 + 1 × identity L1 | Visual, held-out test images | G 2.116; cycle 0.435 | Horse→zebra visible; zebra→horse weak (one direction) |

*Table 1. Final results. FID uses the provided cell: 9,984 test images (312 batches of 32) against the same number of images generated for the same labels.*

**Submitted files.** Three notebooks with outputs, and the epoch-20 checkpoint of each run: `part1_diffusion_checkpoint_020.pt` and `part2_flow_matching_checkpoint_020.pt` (DiT weights under key `model`), and `part3_cyclegan_checkpoint_020.pt` (keys `G_H2Z, G_Z2H, D_H, D_Z`). **Departures from the provided notebooks:** they were run on a local RTX 4070 Ti SUPER instead of Colab, so the Drive-mount cell was removed, `project_path` is a local folder and `!pip` became `%pip`; Part 1 also adds a learning-rate schedule to the training loop (Section 2).

## 2. Part 1 — Diffusion (DiT, DDPM objective, DDIM sampler)

**Key takeaway:** The implementation trained correctly with the provided loop, but passing the 8-step FID target needed a learning-rate decay. **Implementation.** The timestep embedder maps t to 256 sinusoidal features (cosines and sines at geometrically spaced frequencies) and then through Linear–SiLU–Linear; the label embedder is an embedding table, and the two are added to form the conditioning vector c. Each DiT block computes x ← x + gate · f(modulate(norm(x), shift, scale)) for the attention branch and again for the MLP branch, with shift, scale and gate predicted from c (adaLN-Zero: the gates start at zero, so every block starts as the identity). The noise schedule uses β linear from 10<sup>−4</sup> to 0.02, α = 1 − β and ᾱ<sub>t</sub> = the cumulative product of α, and corrupts images in one step: x<sub>t</sub> = √ᾱ<sub>t</sub> x<sub>0</sub> + √(1 − ᾱ<sub>t</sub>) ε. The DDIM sampler visits evenly spaced timesteps from 999 to 0; at each one it predicts the noise, forms the clean-image estimate (x<sub>t</sub> − √(1 − ᾱ<sub>t</sub>) ε) / √ᾱ<sub>t</sub>, and re-noises that estimate to the next timestep with the same predicted noise and no fresh noise (ᾱ = 1 after the last step).

**Training and result.** 20 epochs of 1,875 iterations (batch 32, AdamW, initial learning rate 3×10<sup>−4</sup>) took about 5 minutes. Epoch-average noise-prediction MSE fell from 0.0678 to 0.0248, and the FID at 8 DDIM steps is 26.22. **Change to the provided loop.** With the loop exactly as given (constant learning rate), an earlier run scored FID 40.66 at 8 steps, which fails the target. The same model scored about 23.5 at 50 steps, and other 8-step timestep spacings or clipping the clean-image estimate only moved it between 35 and 44, so the sampler was not the cause. Adding a cosine decay of the learning rate to zero over training (the only change; the architecture and FID cell are untouched) gave 26.22. Those exploratory runs were overwritten and are not in the submitted notebook; only the final run is.

## 3. Part 2 — Flow Matching (same DiT, Euler sampler)

**Key takeaway:** With the same network and the same 8 function evaluations, flow matching reached a lower FID (19.87) with the provided loop unchanged. **Implementation.** Each training step draws Gaussian noise x<sub>0</sub> and a time t uniform on [0, 1], forms x<sub>t</sub> = (1 − t) x<sub>0</sub> + t x<sub>1</sub> for a data image x<sub>1</sub>, and regresses the model output onto the target velocity x<sub>1</sub> − x<sub>0</sub> with MSE; t is multiplied by 1000 to match the DiT time-embedding scale. Sampling starts from noise and takes N forward Euler steps of size 1/N, x ← x + (1/N) v(x, t<sub>k</sub>) with t<sub>k</sub> = k/N for k = 0, ..., N − 1. **Result.** Epoch-average velocity MSE fell from 0.2875 to 0.1458 in about 5 minutes, and the FID at 8 Euler steps is 19.87. A plausible reason for the gap to diffusion is that straight paths have a constant target velocity, so coarse Euler steps lose less than 8 DDIM steps on a curved diffusion trajectory, but I did not test this, and it is not a controlled comparison: these are single runs with different learning-rate schedules (constant here, cosine in Part 1).

![Generated FashionMNIST samples from the diffusion and flow matching models](fig_samples.png)

*Figure 1. Samples saved by the notebooks from the final checkpoints. Left: diffusion, 50 DDIM steps. Right: flow matching, 20 Euler steps. Columns are classes 0–9 (T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot); rows are different noise samples.*

## 4. Part 3 — CycleGAN (horse2zebra)

**Key takeaway:** After 20 epochs the horse-to-zebra generator clearly paints stripes on held-out horses, with strong artifacts; zebra-to-horse is not convincing. **Implementation.** The GAN criterion is MSE on the discriminator’s raw patch scores (least-squares GAN, since the discriminator has no sigmoid); the cycle and identity criteria are L1. Identity loss: each generator is given a real image already in its output domain and penalized for changing it. For each direction the generator translates the real image, the other domain’s discriminator scores the fake against a target of 1, and the fake is translated back and compared with the original (cycle loss). Each discriminator is trained to score its real images as 1 and replay-buffered fakes as 0. The generator loss is GAN + 2 × cycle + 1 × identity, the provided weights.

**Training and result.** 1,067 horse and 1,334 zebra images at 128×128, batch 1, Adam at 2×10<sup>−4</sup>, 20 epochs of 1,334 iterations, about 19 minutes. Epoch-average losses moved from epoch 1 to 20 as follows: generator 2.527→2.116, cycle 0.584→0.435, identity 0.548→0.401, horse discriminator 0.341→0.212, zebra discriminator 0.344→0.134. In Figure 2 the zebra generator puts black-and-white stripes on the animals in most images, but it also tints backgrounds green-yellow, lets stripes bleed into the background, and leaves horizontal line artifacts near the top edge. The horse generator does not produce a horse coat: stripes stay visible on several animals, and the whole image is blurred and shifted toward a muddy green-brown. Likely contributors are the short schedule, the low resolution and the small cycle weight, but I did not test any of them.

![CycleGAN translations of held-out test images](fig_cyclegan.png)

*Figure 2. Final checkpoint on the first 10 test images of each domain in filename order (not selected; not used in training). Rows: real horses, horse→zebra output, real zebras, zebra→horse output. Produced for this report by loading the submitted checkpoint; the notebook itself saves one training example per epoch.*

## Key Takeaways

- Diffusion reached FID 26.22 at 8 DDIM steps (target < 30) only after a cosine learning-rate decay was added; an earlier constant-rate run scored 40.66.
- Flow matching reached FID 19.87 at 8 Euler steps with the same DiT and the provided loop unchanged.
- The CycleGAN translates horses to zebras visibly but with background artifacts; zebra-to-horse is weak after 20 epochs.
- All results are single unseeded runs, so differences of a few FID points should not be over-read.

## AI-Assistance Disclosure

GitHub Copilot (agent mode in VS Code) implemented the TODO sections and their explanatory comments, ran the training and evaluation cells, ran the exploratory FID experiments described in Section 2, and drafted this report. Metrics are taken from the notebook outputs except where the text says otherwise; parameter counts were computed from the checkpoints.
