# ========================= Regressor for x,y,vx,vy,vz at z=L (survivors only) =========================
# Builds a dataset with final positions AND final velocities, trains a 5-output MLP regressor,
# evaluates errors, and provides a one-call inference helper.

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import tqdm
from physics_augment import gammaG_from_bore, ideal_thick_lens_map_gravity, k_from_V_vz


# ---------- (2) Standardizers ----------
class Standardizer:
    def __init__(self, X: np.ndarray):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd == 0] = 1.0

    def torch(self, device):
        return torch.from_numpy(self.mu).float().to(device), torch.from_numpy(
            self.sd
        ).float().to(device)


# ---------- (3) 5-output regressor model ----------
class ResBlock(nn.Module):
    def __init__(self, d, pdrop=0.05):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.act = nn.SiLU()
        self.do = nn.Dropout(pdrop)

    def forward(self, x):
        h = self.do(self.act(self.fc1(x)))
        h = self.fc2(h)
        return self.ln(x + h)


class MLPRes(nn.Module):
    def __init__(self, d_in=6, widths=(256, 256, 256), pdrop=0.05):
        super().__init__()
        layers = [nn.Linear(d_in, widths[0]), nn.SiLU()]
        for i in range(len(widths) - 1):
            layers += [
                ResBlock(widths[i], pdrop),
                nn.Linear(widths[i], widths[i + 1]),
                nn.SiLU(),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Regressor5(nn.Module):
    def __init__(self, d_in=6, widths=(256, 256, 256), pdrop=0.05):
        super().__init__()
        self.backbone = MLPRes(d_in, widths, pdrop)
        self.head = nn.Linear(widths[-1], 5)  # [x/R, y/R, sx, sy, log(vz)]

    def forward(self, x):
        return self.head(self.backbone(x))


class TargetScaler:
    def __init__(self, Y):  # Y: [x/R, y/R, sx, sy, log(vz)]
        Y = np.asarray(Y, dtype=np.float32)
        self.mu = Y.mean(axis=0)
        self.sd = Y.std(axis=0)
        self.sd[self.sd == 0] = 1.0

        # torch buffers (lazy-copied to device on use)
        self._mu_t = torch.from_numpy(self.mu)  # cpu float32
        self._sd_t = torch.from_numpy(self.sd)  # cpu float32

    # numpy API (unchanged)
    def fwd(self, Y):
        return (Y - self.mu) / self.sd

    def inv(self, Z):
        return Z * self.sd + self.mu

    # torch API (differentiable; no detach/cpu/numpy)
    @torch.no_grad()
    def _to_dev(self, device):
        if self._mu_t.device != device:
            self._mu_t = self._mu_t.to(device)
            self._sd_t = self._sd_t.to(device)

    def fwd_torch(self, Yz):  # Y -> z (torch)
        self._to_dev(Yz.device)
        return (Yz - self._mu_t) / self._sd_t

    def inv_torch(self, Z):  # z -> Y (torch)
        self._to_dev(Z.device)
        return Z * self._sd_t + self._mu_t


@dataclass
class TrainCfgReg5:
    epochs: int = 30
    batch: int = 8192
    lr: float = 3e-4
    weight_decay: float = 1e-4
    huber_delta: float = 1.0
    patience: int = 6
    lambda_speed: float = 1e-3  # small speed-consistency weight
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_targets_slopes(Y_surv_tr, R):
    x, y, vx, vy, vz = [Y_surv_tr[:, i] for i in range(5)]
    sx = vx / vz
    sy = vy / vz
    logvz = np.log(vz)
    Y_tgt = np.column_stack([x / R, y / R, sx, sy, logvz]).astype(np.float32)
    return Y_tgt


def build_targets_residual_slopes(
    Y_surv: np.ndarray,
    X_surv_raw: np.ndarray,
    R: float,
    L: float,
    alpha0: float,
    mass: float,
    *,
    g: float = 9.81,
) -> np.ndarray:
    """
    Build residual regression targets relative to the ideal thick-lens map
    ======================================================================

    Converts absolute exit-state targets into residuals relative to the analytical
    ideal thick-lens solution. Optionally includes gravitational sag along y in
    the ideal baseline.

    The regressor predicts:
        [Δx/R, Δy/R, Δsx, Δsy, log(vz)],  where sx = vx/vz.

    Parameters
    ----------
    Y_surv : ndarray, shape (N, 5)
        True final state [x, y, vx, vy, vz].
    X_surv_raw : ndarray, shape (N, 6)
        Raw inputs [x0, y0, vx, vy, vz, V].
    R : float
        Bore radius [m].
    L : float
        Lens length [m].
    alpha0 : float
        Stark polarizability [J/(V/m)²].
    mass : float
        Particle mass [kg].
    g : float, default=9.80665
        Gravitational acceleration [m/s²]. Assumed to act toward −y. If +y is
        downward in your coords, pass g=-9.80665.

    Returns
    -------
    Y_res : ndarray, shape (N, 5), float32
        Residual targets [Δx/R, Δy/R, Δsx, Δsy, log(vz)].

    Notes
    -----
    - Residualization flattens curvature near kL ≈ π/2 and reduces the spike
      observed around 120–165 m/s in absolute training.
    - Gravity in the ideal baseline prevents the regressor from “learning g”
      as a nonlinear correction.
    """
    x0, y0, vx0, vy0, vz0, V0 = [X_surv_raw[:, i] for i in range(6)]
    gammaG = gammaG_from_bore(R)
    k = np.array(
        [k_from_V_vz(alpha0, mass, gammaG, Vi, vzi) for Vi, vzi in zip(V0, vz0)],
        dtype=np.float32,
    )

    x_id, y_id, vx_id, vy_id = ideal_thick_lens_map_gravity(
        x0, y0, vx0, vy0, vz0, k, L, g=g
    )
    sx_id = vx_id / vz0
    sy_id = vy_id / vz0

    x, y, vxf, vyf, vzf = [Y_surv[:, i] for i in range(5)]
    sx = vxf / vzf
    sy = vyf / vzf

    Y_res = np.column_stack(
        [
            (x - x_id) / R,  # Δx/R
            (y - y_id) / R,  # Δy/R
            (sx - sx_id),  # Δsx
            (sy - sy_id),  # Δsy
            np.log(vzf),  # log(vz)
        ]
    ).astype(np.float32)

    return Y_res


def train_regressor5_slopes(
    X_surv_aug: np.ndarray,
    Y_surv: np.ndarray,
    X_surv_raw: np.ndarray,
    muX: np.ndarray,
    sdX: np.ndarray,
    R: float,
    cfg: TrainCfgReg5,
    *,
    L: float,
    alpha0: float,
    mass: float,
) -> tuple[nn.Module, TargetScaler]:
    """
    Train Regressor5 on residual targets relative to the ideal thick-lens map
    =========================================================================

    This variant predicts corrections to the analytical ideal solution:
        [Δx/R, Δy/R, Δsx, Δsy, log(vz)]
    rather than absolute outputs. Residualization flattens target curvature
    near focusing-phase boundaries (e.g., kL ≈ π/2), improving stability and
    high-percentile error in the 120–165 m/s band.

    Parameters
    ----------
    X_surv_aug : ndarray, shape (N, D_aug)
        Augmented input features for TRUE survivors (use the *smoothed* augmenter).
    Y_surv : ndarray, shape (N, 5)
        True exit state [x, y, vx, vy, vz] for survivors.
    X_surv_raw : ndarray, shape (N, 6)
        Raw inputs [x0, y0, vx, vy, vz, V] for survivors.
    muX, sdX : ndarray, shape (D_aug,)
        Mean and std for THIS augmented feature space (from TRAIN survivors).
    R : float
        Bore radius [m].
    cfg : TrainCfgReg5
        Training configuration (batch, lr, weight_decay, etc.).
    L, alpha0, mass : float
        Lens length, Stark polarizability, particle mass (for ideal map).

    Returns
    -------
    model : nn.Module
        Trained Regressor5.
    scalerY : TargetScaler
        Target scaler fitted on the residual targets.

    Notes
    -----
    - The speed-consistency penalty remains unchanged.
    - Keep using the smoothed augmenter (sin/cos(kL), clipped margins) for X.
    """
    # --- residual targets instead of absolute ---
    Y_tgt = build_targets_residual_slopes(
        Y_surv=Y_surv, X_surv_raw=X_surv_raw, R=R, L=L, alpha0=alpha0, mass=mass
    )
    scalerY = TargetScaler(Y_tgt)

    # tensors
    muX_t = torch.from_numpy(muX).float().to(cfg.device)
    sdX_t = torch.from_numpy(sdX).float().to(cfg.device)
    Xz = ((torch.from_numpy(X_surv_aug).float() - muX_t.cpu()) / sdX_t.cpu()).to(
        cfg.device
    )
    Yz = torch.from_numpy(scalerY.fwd(Y_tgt).astype(np.float32)).to(cfg.device)

    ds = torch.utils.data.TensorDataset(Xz, Yz, torch.from_numpy(X_surv_raw).float())
    # split
    n = len(ds)
    n_val = max(1, n // 10)
    n_tr = n - n_val
    tr_set, va_set = torch.utils.data.random_split(
        ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42)
    )
    dl_tr = torch.utils.data.DataLoader(tr_set, batch_size=cfg.batch, shuffle=True)
    dl_va = torch.utils.data.DataLoader(va_set, batch_size=cfg.batch, shuffle=False)

    model = Regressor5(d_in=X_surv_aug.shape[1]).to(cfg.device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    huber = nn.HuberLoss(delta=cfg.huber_delta)

    best = np.inf
    best_state = None
    wait = 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb, xraw in tqdm.tqdm(dl_tr, desc=f"[REG5] Epoch {ep}/{cfg.epochs}"):
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            pred = model(xb)
            # base supervised loss on residual targets
            base = huber(pred, yb)

            # small speed-consistency: ||v_pred|-|v0||
            with torch.no_grad():
                v0 = torch.sqrt(
                    (xraw[:, 2] ** 2 + xraw[:, 3] ** 2 + xraw[:, 4] ** 2)
                ).to(cfg.device)
            # decode residual heads to physical to compute |v_pred|
            Yphys = scalerY.inv(pred.detach().cpu().numpy())
            Yphys = torch.from_numpy(Yphys).float().to(cfg.device)
            logvz = Yphys[:, 4]
            vz = torch.exp(logvz)
            dsx, dsy = Yphys[:, 2], Yphys[:, 3]

            # ideal slopes for this batch (need vx_id/vy_id at exit → sx_id, sy_id)
            x0b, y0b, vx0b, vy0b, vz0b, V0b = [
                xraw[:, i].to(cfg.device) for i in range(6)
            ]
            gammaG = gammaG_from_bore(R)
            # compute k on CPU then move (usually fine); vectorized Python call for clarity
            k_b = torch.tensor(
                [
                    k_from_V_vz(alpha0, mass, gammaG, float(Vi), float(vzi))
                    for Vi, vzi in zip(V0b.cpu(), vz0b.cpu())
                ],
                dtype=torch.float32,
                device=cfg.device,
            )
            # ideal_thick_lens_map should be available in torch or we compute sx_id/sy_id via numpy and move back
            # For consistency with earlier code, keep it on CPU then tensorize:
            x_id_b, y_id_b, vx_id_b, vy_id_b = ideal_thick_lens_map_gravity(
                x0b.cpu().numpy(),
                y0b.cpu().numpy(),
                vx0b.cpu().numpy(),
                vy0b.cpu().numpy(),
                vz0b.cpu().numpy(),
                k_b.cpu().numpy(),
                L,
            )
            sx_id_b = torch.from_numpy(vx_id_b / vz0b.cpu().numpy()).to(cfg.device)
            sy_id_b = torch.from_numpy(vy_id_b / vz0b.cpu().numpy()).to(cfg.device)

            sx = sx_id_b + dsx
            sy = sy_id_b + dsy
            vx = sx * vz
            vy = sy * vz
            vmag = torch.sqrt(vx * vx + vy * vy + vz * vz)
            speed_pen = huber(vmag, v0)

            loss = base + cfg.lambda_speed * speed_pen
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item())

        # val
        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for xb, yb, xraw in dl_va:
                pred = model(xb.to(cfg.device))
                base = huber(pred, yb.to(cfg.device))
                # same speed penalty in val
                Yphys = scalerY.inv(pred.cpu().numpy())
                Yphys = torch.from_numpy(Yphys).float().to(cfg.device)
                logvz = Yphys[:, 4]
                vz = torch.exp(logvz)
                dsx, dsy = Yphys[:, 2], Yphys[:, 3]

                x0b, y0b, vx0b, vy0b, vz0b, V0b = [
                    xraw[:, i].to(cfg.device) for i in range(6)
                ]
                gammaG = gammaG_from_bore(R)
                k_b = torch.tensor(
                    [
                        k_from_V_vz(alpha0, mass, gammaG, float(Vi), float(vzi))
                        for Vi, vzi in zip(V0b.cpu(), vz0b.cpu())
                    ],
                    dtype=torch.float32,
                    device=cfg.device,
                )
                x_id_b, y_id_b, vx_id_b, vy_id_b = ideal_thick_lens_map_gravity(
                    x0b.cpu().numpy(),
                    y0b.cpu().numpy(),
                    vx0b.cpu().numpy(),
                    vy0b.cpu().numpy(),
                    vz0b.cpu().numpy(),
                    k_b.cpu().numpy(),
                    L,
                )
                sx_id_b = torch.from_numpy(vx_id_b / vz0b.cpu().numpy()).to(cfg.device)
                sy_id_b = torch.from_numpy(vy_id_b / vz0b.cpu().numpy()).to(cfg.device)

                sx = sx_id_b + dsx
                sy = sy_id_b + dsy
                vx = sx * vz
                vy = sy * vz
                vmag = torch.sqrt(vx * vx + vy * vy + vz * vz)
                v0 = torch.sqrt(
                    (xraw[:, 2] ** 2 + xraw[:, 3] ** 2 + xraw[:, 4] ** 2)
                ).to(cfg.device)

                vtot += float((base + cfg.lambda_speed * huber(vmag, v0)).item())
        vmean = vtot / max(1, len(dl_va))
        print(
            f"[REG5] Epoch {ep}: train_loss={tot / max(1, len(dl_tr)):.4f}  val_loss={vmean:.4f}"
        )
        if vmean < best - 1e-4:
            best = vmean
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"[REG5] Early stop @ {ep} (best val={best:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scalerY


def _first_in_dim(model: nn.Module) -> int:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
    raise RuntimeError(
        "Could not infer in_features from model (no Linear layer found)."
    )


class SurrogateXYV:
    """
    Flexible surrogate for (survival + final x,y,vx,vy,vz).

    - clf:   trained classifier (expects N x D_clf)
    - reg5:  trained Regressor5 (expects N x D_reg) on 12-D augmented features
    - muX_clf, sdX_clf: normalizer for classifier inputs (len D_clf)
    - muX_reg, sdX_reg: normalizer for regressor inputs  (len D_reg)  <-- your muX_aug/sdX_aug
    - scalerY: TargetScaler used for regressor targets ([x/R,y/R,sx,sy,log(vz)])
    - augment_fn_clf: None if clf was trained on raw 6-D; otherwise your augment function
    - augment_fn_reg: your augment function (must be the SAME you used to train reg5)

    Regressor outputs are converted to physical units:
      x = (x/R)*R, y = (y/R)*R, vz = exp(log(vz)), vx = sx*vz, vy = sy*vz
    """

    def __init__(
        self,
        clf,
        reg5,
        muX_clf,
        sdX_clf,
        muX_reg,
        sdX_reg,
        scalerY,
        R,
        L,
        alpha0,
        mass,
        device,
        thr,
        augment_fn_clf=None,
        augment_fn_reg=None,
    ):
        self.clf = clf
        self.reg5 = reg5
        self.muX_clf = torch.as_tensor(muX_clf, dtype=torch.float32, device=device)
        self.sdX_clf = torch.as_tensor(sdX_clf, dtype=torch.float32, device=device)
        self.muX_reg = torch.as_tensor(muX_reg, dtype=torch.float32, device=device)
        self.sdX_reg = torch.as_tensor(sdX_reg, dtype=torch.float32, device=device)
        self.scalerY = scalerY
        self.R = float(R)
        self.L = float(L)
        self.alpha0 = float(alpha0)
        self.mass = float(mass)
        self.device = device
        self.thr = float(thr)
        self.augment_fn_clf = augment_fn_clf
        self.augment_fn_reg = augment_fn_reg

        self.D_clf = _first_in_dim(self.clf)
        self.D_reg = _first_in_dim(self.reg5)

    def _build_feats(self, Xraw: np.ndarray, D_expected: int, augment_fn):
        # raw is 6-D: [x0,y0,vx,vy,vz,V]
        if D_expected == 6:
            return Xraw.astype(np.float32)
        # otherwise, augment required
        if augment_fn is None:
            raise ValueError(
                f"Model expects {D_expected} features, but no augment_fn was provided."
            )
        X_aug = augment_fn(Xraw, self.R, self.L, self.alpha0, self.mass)
        if X_aug.shape[1] != D_expected:
            raise ValueError(
                f"augment_fn produced {X_aug.shape[1]} features, but model expects {D_expected}. "
                "Make sure you pass the SAME augment_with_physics variant used in training."
            )
        return X_aug.astype(np.float32)

    @torch.no_grad()
    def predict(self, x0, y0, vx0, vy0, vz0, V):
        # vectorize; allow scalar V
        x0 = np.asarray(x0, dtype=np.float32)
        y0 = np.asarray(y0, dtype=np.float32)
        vx0 = np.asarray(vx0, dtype=np.float32)
        vy0 = np.asarray(vy0, dtype=np.float32)
        vz0 = np.asarray(vz0, dtype=np.float32)
        if np.isscalar(V):
            V = np.full_like(x0, float(V), dtype=np.float32)
        else:
            V = np.asarray(V, dtype=np.float32)

        Xraw = np.column_stack([x0, y0, vx0, vy0, vz0, V]).astype(np.float32)  # (N,6)

        # ----- classifier path
        Xc = self._build_feats(Xraw, self.D_clf, self.augment_fn_clf)
        Xt_c = torch.from_numpy(Xc).to(self.device)
        Xt_c = (Xt_c - self.muX_clf) / self.sdX_clf

        # if your clf forward supports apply_temp=True, use it; else remove arg
        try:
            s = self.clf(Xt_c, apply_temp=True).cpu().numpy()
        except TypeError:
            s = self.clf(Xt_c).cpu().numpy()
        p = 1.0 / (1.0 + np.exp(-s))
        survive = p >= self.thr

        # ----- regressor path
        N = Xraw.shape[0]
        x_pred = np.zeros(N, np.float32)
        y_pred = np.zeros(N, np.float32)
        vx_pred = np.zeros(N, np.float32)
        vy_pred = np.zeros(N, np.float32)
        vz_pred = np.zeros(N, np.float32)

        if np.any(survive):
            Xr = self._build_feats(Xraw[survive], self.D_reg, self.augment_fn_reg)
            Xt_r = torch.from_numpy(Xr).to(self.device)
            Xt_r = (Xt_r - self.muX_reg) / self.sdX_reg

            Z = (
                self.reg5(Xt_r).cpu().numpy()
            )  # [x/R, y/R, sx, sy, log(vz)] in scaler space
            Y = self.scalerY.inv(Z)  # back to true target space

            x_pred[survive] = Y[:, 0] * self.R
            y_pred[survive] = Y[:, 1] * self.R
            vz = np.exp(Y[:, 4])
            vx_pred[survive] = Y[:, 2] * vz
            vy_pred[survive] = Y[:, 3] * vz
            vz_pred[survive] = vz

        return p, survive.astype(np.int8), x_pred, y_pred, vx_pred, vy_pred, vz_pred


def evaluate_regressor5(
    model,
    X_all,  # raw [x0,y0,vx,vy,vz,V]
    xf_true,
    yf_true,
    vxf_true,
    vyf_true,
    vzf_true,
    y_survive,  # 0/1 mask (true survivors)
    muX,
    sdX,  # MUST match the feature space fed into 'model' (i.e., 12-D augmented stats)
    scalerY,  # TargetScaler used at training
    R,
    device,
    tag="TEST",
    augment_fn=None,  # pass augment_with_physics if model expects augmented inputs
    alpha0=None,
    mass=None,
    L=None,
):
    """
    Evaluates Regressor5 on TRUE survivors only.
    If model expects 12-D, provide augment_fn and 12-D muX/sdX (from TRAIN survivors).
    If model expects 6-D, set augment_fn=None and pass 6-D muX/sdX.
    """
    D_expected = _first_in_dim(model)

    # Build feature matrix to match the model's expected input size
    if D_expected == 6:
        X_feat_all = X_all.astype(np.float32)
    else:
        if augment_fn is None:
            raise ValueError(
                f"Model expects {D_expected} features but augment_fn=None. "
                "Pass the same augment function used in training."
            )
        X_feat_all = augment_fn(X_all, R=R, L=L, alpha0=alpha0, mass=mass).astype(
            np.float32
        )

    # TRUE survivors only
    mask = y_survive.astype(bool)
    if not np.any(mask):
        print(f"[{tag}] No true survivors in set.")
        return {}

    # Slice survivors for features and truths
    Xs = X_feat_all[mask]
    xs_true = xf_true[mask].astype(np.float32)
    ys_true = yf_true[mask].astype(np.float32)
    vxs_true = vxf_true[mask].astype(np.float32)
    vys_true = vyf_true[mask].astype(np.float32)
    vzs_true = vzf_true[mask].astype(np.float32)

    # Normalize with the SAME stats used at training for THIS feature space
    muX_t = torch.as_tensor(muX, dtype=torch.float32, device=device)
    sdX_t = torch.as_tensor(sdX, dtype=torch.float32, device=device)
    Xt = torch.from_numpy(Xs).to(device)
    Xt = (Xt - muX_t) / sdX_t

    model.eval()
    with torch.no_grad():
        Yz_pred = model(Xt).cpu().numpy()  # in scaler space
    Y_pred_scaled = scalerY.inv(Yz_pred)  # [Δx/R, Δy/R, Δsx, Δsy, log(vz)]

    # Residual heads
    dx_over_R = Y_pred_scaled[:, 0]
    dy_over_R = Y_pred_scaled[:, 1]
    dsx = Y_pred_scaled[:, 2]
    dsy = Y_pred_scaled[:, 3]
    logvz = Y_pred_scaled[:, 4]
    vz_pred = np.exp(logvz).astype(np.float32)

    # Ideal solution for THIS survivor batch from raw inputs (masked!)
    X_raw_surv = X_all[mask].astype(np.float32)
    x0s, y0s, vx0s, vy0s, vz0s, V0s = [X_raw_surv[:, i] for i in range(6)]

    gammaG = gammaG_from_bore(R)
    k_surv = np.array(
        [k_from_V_vz(alpha0, mass, gammaG, Vi, vzi) for Vi, vzi in zip(V0s, vz0s)],
        dtype=np.float32,
    )
    x_id, y_id, vx_id, vy_id = ideal_thick_lens_map_gravity(
        x0s, y0s, vx0s, vy0s, vz0s, k_surv, L
    )
    sx_id = vx_id / vz0s
    sy_id = vy_id / vz0s

    # Add residuals to ideal predictions → absolute outputs
    x_pred = x_id + dx_over_R * R
    y_pred = y_id + dy_over_R * R
    sx_pred = sx_id + dsx
    sy_pred = sy_id + dsy
    vx_pred = sx_pred * vz_pred
    vy_pred = sy_pred * vz_pred

    # Errors
    dx = x_pred - xs_true
    dy = y_pred - ys_true
    dvx = vx_pred - vxs_true
    dvy = vy_pred - vys_true
    dvz = vz_pred - vzs_true
    rerr = np.sqrt(dx * dx + dy * dy)

    mm = 1e3
    mae_x = float(np.mean(np.abs(dx)) * mm)
    mae_y = float(np.mean(np.abs(dy)) * mm)
    rmse_x = float(np.sqrt(np.mean(dx * dx)) * mm)
    rmse_y = float(np.sqrt(np.mean(dy * dy)) * mm)
    r2x = float(
        1.0
        - np.sum((xs_true - x_pred) ** 2)
        / (np.sum((xs_true - np.mean(xs_true)) ** 2) + 1e-20)
    )
    r2y = float(
        1.0
        - np.sum((ys_true - y_pred) ** 2)
        / (np.sum((ys_true - np.mean(ys_true)) ** 2) + 1e-20)
    )
    p50, p90, p99 = [float(v) * mm for v in np.percentile(rerr, [50, 90, 99])]

    mae_vx = float(np.mean(np.abs(dvx)))
    mae_vy = float(np.mean(np.abs(dvy)))
    mae_vz = float(np.mean(np.abs(dvz)))
    rmse_vx = float(np.sqrt(np.mean(dvx * dvx)))
    rmse_vy = float(np.sqrt(np.mean(dvy * dvy)))
    rmse_vz = float(np.sqrt(np.mean(dvz * dvz)))
    r2_vx = float(
        1.0
        - np.sum((vxs_true - vx_pred) ** 2)
        / (np.sum((vxs_true - np.mean(vxs_true)) ** 2) + 1e-20)
    )
    r2_vy = float(
        1.0
        - np.sum((vys_true - vy_pred) ** 2)
        / (np.sum((vys_true - np.mean(vys_true)) ** 2) + 1e-20)
    )
    r2_vz = float(
        1.0
        - np.sum((vzs_true - vz_pred) ** 2)
        / (np.sum((vzs_true - np.mean(vzs_true)) ** 2) + 1e-20)
    )

    print(f"=== Final state ({tag}, TRUE survivors) ===")
    print(
        f"POS  MAE: x={mae_x:.3f} mm, y={mae_y:.3f} mm   RMSE: x={rmse_x:.3f} mm, y={rmse_y:.3f} mm"
    )
    print(
        f"POS  R² : x={r2x:.4f}, y={r2y:.4f}   Radial P50={p50:.3f} mm, P90={p90:.3f} mm, P99={p99:.3f} mm"
    )
    print(f"VEL  MAE: vx={mae_vx:.3f} m/s, vy={mae_vy:.3f} m/s, vz={mae_vz:.3f} m/s")
    print(f"VEL RMSE: vx={rmse_vx:.3f} m/s, vy={rmse_vy:.3f} m/s, vz={rmse_vz:.3f} m/s")
    print(f"VEL   R²: vx={r2_vx:.4f}, vy={r2_vy:.4f}, vz={r2_vz:.4f}")

    return dict(
        pos=dict(
            mae_x_mm=mae_x,
            mae_y_mm=mae_y,
            rmse_x_mm=rmse_x,
            rmse_y_mm=rmse_y,
            r2x=r2x,
            r2y=r2y,
            radial_p50_mm=p50,
            radial_p90_mm=p90,
            radial_p99_mm=p99,
        ),
        vel=dict(
            mae_vx=mae_vx,
            mae_vy=mae_vy,
            mae_vz=mae_vz,
            rmse_vx=rmse_vx,
            rmse_vy=rmse_vy,
            rmse_vz=rmse_vz,
            r2_vx=r2_vx,
            r2_vy=r2_vy,
            r2_vz=r2_vz,
        ),
        n_surv=int(mask.sum()),
        x_pred=x_pred,
        y_pred=y_pred,
        vx_pred=vx_pred,
        vy_pred=vy_pred,
        vz_pred=vz_pred,
    )
