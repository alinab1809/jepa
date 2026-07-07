"""Minimal I-JEPA: EMA target, 4 target blocks, context block, smooth-L1 on LN targets."""
import copy, math, random
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vector_quantize_pytorch import finite_scalar_quantization as fsq
import wandb

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def sincos_2d(h, w, dim):
    assert dim % 4 == 0; sub = dim // 4
    yy, xx = [t.reshape(-1).float() for t in torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")]
    div = torch.exp(torch.arange(0, sub * 2, 2).float() * (-math.log(10000.) / (sub * 2)))
    return torch.cat([torch.sin(yy[:, None] * div), torch.cos(yy[:, None] * div),
                      torch.sin(xx[:, None] * div), torch.cos(xx[:, None] * div)], dim=-1)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp=4.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Linear(int(dim * mlp), dim))

    def forward(self, x):
        h = self.n1(x); x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


def param_groups(modules, wd):
    np_ = [(n, p) for m in modules for n, p in m.named_parameters() if p.requires_grad]
    nd = [p for n, p in np_ if p.ndim < 2 or n.endswith("bias")]
    d = [p for n, p in np_ if p.ndim >= 2 and not n.endswith("bias")]
    return [{"params": d, "weight_decay": wd}, {"params": nd, "weight_decay": 0.0}]


@torch.no_grad()
def ema_update(tgt, online, m):
    # theta_bar <- m * theta_bar + (1 - m) * theta
    for pt, po in zip(tgt.parameters(), online.parameters()): pt.mul_(m).add_(po.detach(), alpha=1 - m)


def lr_warmup_cosine(step, total, base, warmup_frac=0.05):
    warm = max(1, int(total * warmup_frac))
    if step < warm: return base * (step + 1) / warm
    return base * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def pick_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def compute_fsq_metrics(target_indices, total_codebook_size):
    """
    Args:
        target_indices (Tensor): Integer indices returned by FSQ. Shape: (Batch, Seq) or (Batch, H, W)
        total_codebook_size (int): The total number of codes (multiplication of all FSQ levels)
    """
    # 1. Flatten to a 1D tensor of all tokens in the current batch
    flat_indices = target_indices.detach().view(-1)

    # 2. Codebook Usage Percentage
    # Find how many unique indices were actively hit in this batch
    unique_indices = torch.unique(flat_indices)
    codebook_usage = len(unique_indices) / total_codebook_size

    # 3. Perplexity (Entropy-based utilization)
    # Calculate the empirical probability distribution of the codes in this batch
    counts = torch.bincount(flat_indices, minlength=total_codebook_size).float()
    probs = counts / counts.sum()

    # Avoid log(0) by adding a small epsilon where probs > 0
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    perplexity = torch.exp(entropy)
    normalized_perplexity = perplexity / total_codebook_size

    return {
        "codebook_usage_pct": codebook_usage * 100,  # e.g., 45.2%
        "perplexity": perplexity.item(),  # e.g., 128.4 (out of total_codebook_size)
        "normalized_perplexity": normalized_perplexity.item()  # scale of 0 to 1
    }


class Encoder(nn.Module):                                 # f_theta (context encoder)
    def __init__(self, img_size=32, patch_size=4, in_chans=3, dim=128, depth=6, heads=4):
        super().__init__()
        self.grid = img_size // patch_size; self.n_patches = self.grid ** 2
        self.dim = dim; self.patch_size = patch_size; self.img_size = img_size
        self.patch_proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.register_buffer("pos", sincos_2d(self.grid, self.grid, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, imgs, idx=None):
        tokens = self.patch_proj(imgs).flatten(2).transpose(1, 2)
        B, N, D = tokens.shape
        if idx is None:
            idx = torch.arange(N, device=imgs.device).expand(B, -1); x = tokens + self.pos[idx]
        else:
            x = tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, D)) + self.pos[idx]
        for blk in self.blocks: x = blk(x)
        return self.norm(x)


class Predictor(nn.Module):                              # g_phi
    def __init__(self, grid, enc_dim=128, dim=64, depth=4, heads=4):
        super().__init__()
        self.in_proj = nn.Linear(enc_dim, dim); self.out_proj = nn.Linear(dim, enc_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim)); nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer("pos", sincos_2d(grid, grid, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, ctx, ctx_idx, tgt_idx):
        B, T = ctx.size(0), tgt_idx.size(1)
        x = torch.cat([self.in_proj(ctx) + self.pos[ctx_idx],
                       self.mask_token.expand(B, T, -1) + self.pos[tgt_idx]], dim=1)
        for blk in self.blocks: x = blk(x)
        return self.out_proj(self.norm(x[:, -T:]))

class FSQPredictor(nn.Module):                              # g_phi
    def __init__(self, grid, enc_dim=128, dim=64, depth=4, heads=4, fsq_dim=4, fsq_L=8):
        super().__init__()
        self.in_proj = nn.Linear(enc_dim, dim); self.out_proj = nn.Linear(dim, fsq_dim*fsq_L)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim)); nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer("pos", sincos_2d(grid, grid, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, ctx, ctx_idx, tgt_idx):
        B, T = ctx.size(0), tgt_idx.size(1)
        x = torch.cat([self.in_proj(ctx) + self.pos[ctx_idx],
                       self.mask_token.expand(B, T, -1) + self.pos[tgt_idx]], dim=1)
        for blk in self.blocks: x = blk(x)

        return self.out_proj(self.norm(x[:, -T:]))


def _bsize(g, s, ar):
    a = s * g * g
    return (max(1, min(g, round(math.sqrt(a * ar)))), max(1, min(g, round(math.sqrt(a / ar)))))


def _block(g, top, left, h, w):
    return {r * g + c for r in range(top, top + h) for c in range(left, left + w)}


def sample_ijepa_masks(B, grid, n_targets=4, min_ctx=4, rng=None):
    """Block sizes shared per batch; locations per item; random-subsample trim."""
    rng = rng or random
    th, tw = _bsize(grid, rng.uniform(0.15, 0.20), rng.uniform(0.75, 1.5))
    ch, cw = _bsize(grid, rng.uniform(0.85, 1.0), 1.0)
    ctx_list = [None] * B; tgt_lists = [[None] * B for _ in range(n_targets)]
    for b in range(B):
        ts = []
        for m in range(n_targets):
            top, left = rng.randint(0, grid - th), rng.randint(0, grid - tw)
            t = _block(grid, top, left, th, tw); ts.append(t); tgt_lists[m][b] = sorted(t)
        for _ in range(10):
            ct, cl = rng.randint(0, grid - ch), rng.randint(0, grid - cw)
            c = _block(grid, ct, cl, ch, cw) - set().union(*ts)
            if len(c) >= min_ctx: break
        ctx_list[b] = sorted(c) if c else [0]
    L = min(len(c) for c in ctx_list)
    return [sorted(rng.sample(c, L)) for c in ctx_list], tgt_lists


def train(epochs=8, batch_size=256, lr=3e-4, wd=0.05, ema_start=0.996, ema_end=1.0,
          device=None, on_epoch_end=None, fsq_dim=4, fsq_L=8):
    device = device or pick_device(); print(f"device: {device}")
    tfm = transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.3, 1.0)),
                              transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    ctx_enc = Encoder().to(device); tgt_enc = copy.deepcopy(ctx_enc).to(device)
    for p in tgt_enc.parameters(): p.requires_grad_(False)
    pred = FSQPredictor(grid=ctx_enc.grid, fsq_dim=fsq_dim, fsq_L=fsq_L).to(device)
    # opt = torch.optim.AdamW(param_groups([ctx_enc, pred], wd), lr=lr)
    total = epochs * len(loader); rng = random.Random(0); losses = []; step = 0
    D = ctx_enc.dim

    ctx_proj = nn.Linear(D, fsq_dim).to(device)
    tgt_proj = copy.deepcopy(ctx_proj).to(device)
    for p in tgt_proj.parameters(): p.requires_grad_(False)
    quantizer = fsq.FSQ(levels =[fsq_L for _ in range(fsq_dim)]).to(device)
    opt = torch.optim.AdamW(param_groups([ctx_enc, pred, ctx_proj], wd), lr=lr)
    if on_epoch_end is not None:
        on_epoch_end({"epoch": -1, "ctx_enc": ctx_enc, "tgt_enc": tgt_enc,
                      "predictor": pred, "step": 0})
    for epoch in range(epochs):
        for imgs, _ in loader:
            imgs = imgs.to(device)
            cl, tls = sample_ijepa_masks(imgs.size(0), ctx_enc.grid, rng=rng)
            ci = torch.tensor(cl, device=device)
            tis = [torch.tensor(t, device=device) for t in tls]
            for g in opt.param_groups: g["lr"] = lr_warmup_cosine(step, total, lr)
            with torch.no_grad(): full = F.layer_norm(tgt_enc(imgs), (D,))  # LN(s_y); no_grad = stop-gradient
            ce = ctx_enc(imgs, ci)                                          # s_x = f_theta(x_context)
            # loss = sum(
            #     F.smooth_l1_loss(
            #         pred(ce, ci, ti),                                       # hat_s_y(i) = g_phi(s_x, B_i)
            #         full.gather(1, ti.unsqueeze(-1).expand(-1, -1, D)))     # [LN(s_y)]_{B_i}
            #     for ti in tis                                               # for i in 1..M
            # ) / len(tis)                                                    # (1/M) * sum

            # FSQ VARIANT

            loss = torch.tensor(0.0, device=device)

            all_flat_idx = []

            for ti in tis:
                tgt_feats = full.gather(1, ti.unsqueeze(-1).expand(-1, -1, D))  # (B, T, D)
                projected = tgt_proj(tgt_feats)  # (B, T, fsq_dim)
                # _, flat_idx = quantizer(projected)  # flat_idx: (B, T)
                # level_idx = torch.stack([  # (B, T, fsq_dim)
                #     (flat_idx // (fsq_L ** i)) % fsq_L
                #     for i in range(fsq_dim)
                # ], dim=-1)
                quantized, flat_idx = quantizer(projected)  # (B, T, fsq_dim), values in {-3..3}
                level_idx = (quantized + (fsq_L - 1) // 2).long()  # (B, T, fsq_dim), values in {0..fsq_L-1}
                out = pred(ce, ci, ti)  # (B, T, fsq_dim*fsq_L)

                B, T, _ = out.shape
                out = out.view(B, T, fsq_dim, fsq_L)  # (B, T, d, L)
                loss += F.cross_entropy(
                    out.permute(0, 3, 1, 2),  # (B, L, T, d)
                    level_idx.long()  # (B, T, d)
                )
                with torch.no_grad():
                    all_flat_idx.append(flat_idx.detach().cpu())

            metrics = compute_fsq_metrics(flat_idx, fsq_L ** fsq_dim)
            loss = loss / len(tis)

            opt.zero_grad(); loss.backward(); opt.step()
            m = ema_start + (ema_end - ema_start) * (step / max(1, total - 1))
            ema_update(tgt_enc, ctx_enc, m); losses.append(loss.item())
            ema_update(tgt_proj, ctx_proj, m)
            if step % 50 == 0:
                print(f"ep={epoch} step={step:5d} loss={loss.item():.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e} ema={m:.4f}")
                print(f"epoch codebook usage: {metrics['codebook_usage_pct']:.1f}%")
                print(f"epoch normalized perplexity: {metrics['normalized_perplexity']:.3f}")

            wandb.log({"loss": loss.item(),
                       "lr": opt.param_groups[0]['lr'],
                       "ema": m,
                       "codebook usage": metrics['codebook_usage_pct'],
                       "normalized perplexity": metrics['normalized_perplexity'],
                       "perplexity": metrics['perplexity'],
                       "global_step": step})
            
            step += 1
        if on_epoch_end is not None:
            on_epoch_end({"epoch": epoch, "ctx_enc": ctx_enc, "tgt_enc": tgt_enc,
                          "predictor": pred, "step": step})
    return {"ctx_enc": ctx_enc, "tgt_enc": tgt_enc, "predictor": pred,
            "losses": losses, "loader": loader, "device": device}


if __name__ == "__main__": train()
