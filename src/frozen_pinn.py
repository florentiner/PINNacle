"""Frozen-PINN: gradient-free PINN training via frozen random features.

Reference: "Fast training of accurate physics-informed neural networks without
gradient descent" (arXiv:2405.20836). Instead of training a full network with
gradient descent, the spatial part of the solution is a random-feature layer
Phi(x) = tanh(W x + b) with W, b sampled once and frozen forever. Only the
time-dependent output coefficients C(t) are solved for, by substituting the
ansatz into the PDE to get an ODE for C(t) and integrating it with a classical
ODE solver instead of backpropagation.

This module implements the method for the 1D viscous Burgers' equation
(u_t + u u_x - nu u_xx = 0) with zero Dirichlet boundary conditions, enforced
exactly (rather than via a loss term) using the boundary-compliant ansatz
u(x, t) = D(x) * C(t) . Phi(x), with D(x) = (x - l)(r - x) vanishing at both
endpoints.

This uses the paper's simpler ELM sampling (data-independent Gaussian/uniform
weights) rather than its data-adaptive SWIM sampling; ELM is markedly less
accurate on sharp features like the shock viscous Burgers' develops near t=1
for small nu (e.g. nu=0.01/pi), so expect noticeably higher error there than
the paper's headline numbers, which use SWIM.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class RandomFeatures:
    """Frozen random-feature layer Phi(x) = tanh(W x + b) (ELM-style sampling).

    W is sampled from a standard Gaussian, b from Uniform(-eta, eta); both are
    fixed at construction and never updated.
    """

    def __init__(self, num_features, input_dim=1, eta=2.0, w_scale=1.0, seed=None):
        rng = np.random.default_rng(seed)
        self.num_features = num_features
        self.input_dim = input_dim
        self.w_scale = w_scale
        self.eta = eta
        # w_scale=1.0 reproduces the original standard-Gaussian sampling exactly.
        self.W = rng.normal(scale=w_scale, size=(num_features, input_dim))
        self.b = rng.uniform(-eta, eta, size=(num_features,))

    def eval(self, x):
        """x: (N, input_dim) -> Phi(x): (N, num_features + 1), incl. a constant feature."""
        z = x @ self.W.T + self.b
        tanh_z = np.tanh(z)
        return np.concatenate([tanh_z, np.ones((x.shape[0], 1))], axis=1)

    def eval_with_x_derivatives(self, x):
        """1D input only. Returns (Phi, dPhi/dx, d2Phi/dx2), each (N, num_features + 1)."""
        assert x.shape[1] == 1, "closed-form derivatives are only implemented for 1D input"
        w = self.W[:, 0]
        z = x @ self.W.T + self.b
        tanh_z = np.tanh(z)
        sech2 = 1.0 - tanh_z ** 2

        zeros_col = np.zeros((x.shape[0], 1))
        phi = np.concatenate([tanh_z, np.ones((x.shape[0], 1))], axis=1)
        dphi = np.concatenate([sech2 * w, zeros_col], axis=1)
        d2phi = np.concatenate([-2.0 * tanh_z * sech2 * (w ** 2), zeros_col], axis=1)
        return phi, dphi, d2phi

    def eval_with_laplacian(self, x):
        """Any input dim. Returns (Phi, LaplacianPhi), each (N, num_features + 1).

        Laplacian of tanh(W.x + b) = -2 tanh (1 - tanh^2) ||W||^2 (constant feature -> 0).
        Used for the 2D Gray-Scott diffusion term u_xx + u_yy.
        """
        x = np.asarray(x, dtype=float).reshape(-1, self.input_dim)
        z = x @ self.W.T + self.b
        tanh_z = np.tanh(z)
        sech2 = 1.0 - tanh_z ** 2
        w_sq_norm = np.sum(self.W ** 2, axis=1)  # (num_features,)

        zeros_col = np.zeros((x.shape[0], 1))
        phi = np.concatenate([tanh_z, np.ones((x.shape[0], 1))], axis=1)
        lap = np.concatenate([-2.0 * tanh_z * sech2 * w_sq_norm, zeros_col], axis=1)
        return phi, lap


def solve_burgers1d_frozen(
    geom=(-1.0, 1.0),
    time=(0.0, 1.0),
    nu=0.01 / np.pi,
    ic_func=lambda x: -np.sin(np.pi * x),
    num_features=2000,
    num_collocation=4000,
    eta=2.0,
    seed=0,
    num_time_eval=201,
    rtol=1e-6,
    atol=1e-9,
    pinv_rcond=1e-8,
):
    """Frozen-PINN solve of viscous Burgers' equation with zero Dirichlet BC.

    Returns:
        sol: the scipy.integrate.solve_ivp OdeResult for the time-coefficient ODE.
        features: the frozen RandomFeatures instance used.
        predict: callable predict(x, t) -> u, evaluated pointwise for equal-length
            1D arrays x, t (broadcasting scalars as needed).
    """
    l, r = geom
    t0, t1 = time
    x = np.linspace(l, r, num_collocation).reshape(-1, 1)

    features = RandomFeatures(num_features, input_dim=1, eta=eta, seed=seed)
    phi, dphi, d2phi = features.eval_with_x_derivatives(x)

    # Boundary-compliant ansatz: u(x, t) = D(x) * C(t).Phi(x), D vanishing at l, r
    # so that Dirichlet BC u(l, t) = u(r, t) = 0 holds by construction.
    xf = x[:, 0]
    D0 = (xf - l) * (r - xf)
    D1 = (r + l) - 2.0 * xf
    D2 = -2.0 * np.ones_like(xf)

    # Effective features for u, u_x, u_xx (product rule through D(x)), as (num_features+1, Nc).
    psi0 = (D0[:, None] * phi).T
    psi1 = (D1[:, None] * phi + D0[:, None] * dphi).T
    psi2 = (D2[:, None] * phi + 2.0 * D1[:, None] * dphi + D0[:, None] * d2phi).T

    # Truncated pseudoinverse: raw tanh random features are severely ill-conditioned
    # (near-duplicate saturated columns), so small singular values are dropped for
    # numerical stability (equivalent to the paper's optional SVD-truncation layer).
    psi0_pinv = np.linalg.pinv(psi0, rcond=pinv_rcond)  # (Nc, num_features + 1)

    # Initial condition: least-squares fit of C(0) to u(x, 0) = ic_func(x).
    ic_values = ic_func(xf)
    C0 = ic_values @ psi0_pinv

    def rhs(_t, C):
        u = C @ psi0
        u_x = C @ psi1
        u_xx = C @ psi2
        residual_rhs = nu * u_xx - u * u_x
        return residual_rhs @ psi0_pinv

    t_eval = np.linspace(t0, t1, num_time_eval)
    sol = solve_ivp(rhs, (t0, t1), C0, t_eval=t_eval, method="RK45", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"Frozen-PINN time integration failed: {sol.message}")

    C_interp = interp1d(sol.t, sol.y, axis=1, fill_value="extrapolate")

    def predict(x_query, t_query):
        x_query = np.asarray(x_query, dtype=float).reshape(-1)
        t_query = np.broadcast_to(np.asarray(t_query, dtype=float), x_query.shape)
        phi_q = features.eval(x_query.reshape(-1, 1))
        D_q = (x_query - l) * (r - x_query)
        psi_q = D_q[:, None] * phi_q  # (Nq, num_features + 1)
        C_q = C_interp(t_query).T  # (Nq, num_features + 1)
        return np.sum(C_q * psi_q, axis=1)

    return sol, features, predict


# --------------------------------------------------------------------------- #
# Additions for the chaotic PDEs (KS, Gray-Scott) + a conditioning helper.
#
# The KS/Gray-Scott solvers return a 4th value `diagnostics` (the frozen
# feature/projection matrix singular spectrum + integrator status) that the
# landscape-compare pipeline uses as the "convexity / conditioning" contrast to
# the gradient-descent loss landscapes. See experiments/landscape_compare/.
# --------------------------------------------------------------------------- #
class FourierFeatures:
    """Frozen periodic Fourier basis on [x0, x0 + L]: [1, cos(k.), sin(k.)], k=1..K.

    Inherently periodic with exact derivatives, so it is the natural frozen basis for
    periodic PDEs such as Kuramoto-Sivashinsky. With enough modes and collocation
    points the projection matrix is essentially orthogonal (condition number ~ 1),
    i.e. the coefficient problem is convex and well-conditioned -- the opposite of the
    tanh random features, which saturate and become ill-conditioned.
    """

    def __init__(self, num_modes, x0=0.0, length=2.0 * np.pi):
        self.num_modes = num_modes
        self.x0 = x0
        self.length = length
        self.wavenumbers = 2.0 * np.pi / length * np.arange(1, num_modes + 1)  # k_m

    @property
    def size(self):
        return 2 * self.num_modes + 1

    def eval(self, x):
        """x: (N,) or (N,1) -> Phi: (N, 2K+1) = [1, cos(k x), sin(k x)]."""
        x = np.asarray(x, dtype=float).reshape(-1)
        kx = np.outer(x, self.wavenumbers)  # (N, K)
        return np.concatenate([np.ones((x.shape[0], 1)), np.cos(kx), np.sin(kx)], axis=1)

    def eval_with_derivatives(self, x):
        """Returns (Phi, Phi_x, Phi_xx, Phi_xxxx), each (N, 2K+1)."""
        x = np.asarray(x, dtype=float).reshape(-1)
        k = self.wavenumbers
        kx = np.outer(x, k)  # (N, K)
        c, s = np.cos(kx), np.sin(kx)
        ones = np.ones((x.shape[0], 1))
        zeros = np.zeros((x.shape[0], 1))

        phi = np.concatenate([ones, c, s], axis=1)
        phi_x = np.concatenate([zeros, -k * s, k * c], axis=1)
        phi_xx = np.concatenate([zeros, -(k ** 2) * c, -(k ** 2) * s], axis=1)
        phi_xxxx = np.concatenate([zeros, (k ** 4) * c, (k ** 4) * s], axis=1)
        return phi, phi_x, phi_xx, phi_xxxx


def _singular_spectrum(matrix):
    """Singular values + condition number of a (frozen) projection matrix."""
    sv = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    return sv, cond


def solve_kuramoto_sivashinsky_frozen(
    geom=(0.0, 2.0 * np.pi),
    time=(0.0, 1.0),
    alpha=100.0 / 16.0,
    beta=100.0 / (16.0 * 16.0),
    gamma=100.0 / (16.0 ** 4),
    ic_func=lambda x: np.cos(x) * (1.0 + np.sin(x)),
    num_modes=64,
    num_collocation=512,
    num_time_eval=251,
    method="BDF",
    rtol=1e-8,
    atol=1e-10,
):
    """Frozen-PINN solve of Kuramoto-Sivashinsky   u_t + a u u_x + b u_xx + g u_xxxx = 0.

    Uses a frozen periodic Fourier basis on [x0, x0+L]. The spatial derivatives are
    exact (diagonal in Fourier); the nonlinear term u u_x is handled pseudo-spectrally
    by evaluating on the collocation grid and projecting back with the frozen basis'
    pseudo-inverse. The linear part is stiff (g k^4), so an implicit integrator (BDF /
    Radau) is used. This is effectively a spectral Galerkin solve and is expected to be
    accurate on KS, unlike a gradient-descent PINN.

    Returns (sol, features, predict, diagnostics). Does not hard-fail: a partial
    trajectory is still a usable data point (see diagnostics['integrator_success']).
    """
    x0, x1 = geom
    length = x1 - x0
    t0, t1 = time

    features = FourierFeatures(num_modes, x0=x0, length=length)
    x_col = x0 + (np.arange(num_collocation) + 0.5) * length / num_collocation  # uniform, periodic
    phi, phi_x, phi_xx, phi_xxxx = features.eval_with_derivatives(x_col)
    phi_pinv = np.linalg.pinv(phi)  # (2K+1, Nc)

    a0 = ic_func(x_col) @ phi_pinv.T  # least-squares coeffs of the IC

    # Linear part of the coefficient ODE is constant: L_lin @ a = phi_pinv @ (-(b u_xx + g u_xxxx)).
    L_lin = -phi_pinv @ (beta * phi_xx + gamma * phi_xxxx)  # (M, M)

    def rhs(_t, a):
        u = phi @ a
        u_x = phi_x @ a
        u_xx = phi_xx @ a
        u_xxxx = phi_xxxx @ a
        field = -(alpha * u * u_x + beta * u_xx + gamma * u_xxxx)  # u_t
        return phi_pinv @ field

    def jac(_t, a):
        # d/da of phi_pinv @ (-alpha u u_x) plus the constant linear part.
        u = phi @ a
        u_x = phi_x @ a
        return L_lin - alpha * (phi_pinv @ (u_x[:, None] * phi + u[:, None] * phi_x))

    t_eval = np.linspace(t0, t1, num_time_eval)
    # An analytic Jacobian keeps the stiff implicit integrator from re-estimating a dense
    # Jacobian by finite differences every step (orders of magnitude faster, esp. for few modes).
    sol = solve_ivp(rhs, (t0, t1), a0, t_eval=t_eval, method=method, rtol=rtol, atol=atol, jac=jac)
    C_interp = interp1d(sol.t, sol.y, axis=1, fill_value="extrapolate", bounds_error=False)

    def predict(x_query, t_query):
        x_query = np.asarray(x_query, dtype=float).reshape(-1)
        t_query = np.broadcast_to(np.asarray(t_query, dtype=float), x_query.shape)
        phi_q = features.eval(x_query)  # (Nq, 2K+1)
        C_q = C_interp(t_query).T       # (Nq, 2K+1)
        return np.sum(C_q * phi_q, axis=1)

    sv, cond = _singular_spectrum(phi)
    diagnostics = {"singular_values": sv, "condition_number": cond,
                   "num_features": features.size, "basis": "fourier",
                   "integrator_success": bool(sol.success), "integrator_message": sol.message}
    return sol, features, predict, diagnostics


def solve_grayscott_frozen(
    bbox=(-1.0, 1.0, -1.0, 1.0, 0.0, 200.0),
    b=0.04,
    d=0.1,
    epsilon=(1e-5, 5e-6),
    ic_u=lambda x, y: 1.0 - np.exp(-80.0 * ((x + 0.05) ** 2 + (y + 0.02) ** 2)),
    ic_v=lambda x, y: np.exp(-80.0 * ((x - 0.05) ** 2 + (y - 0.02) ** 2)),
    num_features=300,
    num_collocation_per_dim=32,
    eta=2.0,
    w_scale=5.0,
    seed=0,
    num_time_eval=21,
    method="BDF",
    rtol=1e-5,
    atol=1e-8,
    max_step=None,
):
    """Best-effort Frozen-PINN solve of the Gray-Scott system

        u_t = e1 (u_xx+u_yy) + b(1-u) - u v^2
        v_t = e2 (v_xx+v_yy) - d v      + u v^2

    on a rectangle with localized-bump initial conditions. Two shared 2D tanh random
    features drive coefficient vectors (C_u, C_v); the reaction/diffusion right-hand
    side is evaluated on a collocation grid and projected back with the frozen basis'
    pseudo-inverse, giving a coupled stiff ODE integrated over [t0, t1].

    IMPORTANT CAVEATS (documented on purpose): the projection of the nonlinear reaction
    field onto a finite tanh span is only approximate, no spatial boundary condition is
    enforced exactly, and the horizon (T=200) is long. Convergence to the true chaotic
    pattern is NOT guaranteed; the solver never hard-fails and returns whatever
    trajectory the integrator reached (`diagnostics['integrator_success']`) so the
    comparison pipeline always has a data point.

    Returns (sol, features, predict, diagnostics).
    """
    x0, x1, y0, y1, t0, t1 = bbox
    xs = np.linspace(x0, x1, num_collocation_per_dim)
    ys = np.linspace(y0, y1, num_collocation_per_dim)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)  # (Nc, 2)

    features = RandomFeatures(num_features, input_dim=2, eta=eta, w_scale=w_scale, seed=seed)
    phi, lap = features.eval_with_laplacian(pts)  # (Nc, F+1)
    phi_pinv = np.linalg.pinv(phi)                # (F+1, Nc)

    e1, e2 = epsilon
    Cu0 = ic_u(pts[:, 0], pts[:, 1]) @ phi_pinv.T
    Cv0 = ic_v(pts[:, 0], pts[:, 1]) @ phi_pinv.T
    m = phi.shape[1]

    def rhs(_t, y):
        Cu, Cv = y[:m], y[m:]
        u = phi @ Cu
        v = phi @ Cv
        lap_u = lap @ Cu
        lap_v = lap @ Cv
        reaction = u * v * v
        field_u = e1 * lap_u + b * (1.0 - u) - reaction
        field_v = e2 * lap_v - d * v + reaction
        return np.concatenate([phi_pinv @ field_u, phi_pinv @ field_v])

    t_eval = np.linspace(t0, t1, num_time_eval)
    solve_kwargs = dict(t_eval=t_eval, method=method, rtol=rtol, atol=atol)
    if max_step is not None:
        solve_kwargs["max_step"] = max_step
    try:
        sol = solve_ivp(rhs, (t0, t1), np.concatenate([Cu0, Cv0]), **solve_kwargs)
        success, message = bool(sol.success), sol.message
        t_grid, y_grid = sol.t, sol.y
    except Exception as exc:  # integration blew up -> keep the initial condition only
        success, message = False, f"integration raised: {exc}"
        t_grid, y_grid = np.array([t0]), np.concatenate([Cu0, Cv0])[:, None]
        sol = None

    Cu_interp = interp1d(t_grid, y_grid[:m], axis=1, fill_value="extrapolate", bounds_error=False)
    Cv_interp = interp1d(t_grid, y_grid[m:], axis=1, fill_value="extrapolate", bounds_error=False)

    def predict(x_query, y_query, t_query):
        x_query = np.asarray(x_query, dtype=float).reshape(-1)
        y_query = np.asarray(y_query, dtype=float).reshape(-1)
        t_query = np.broadcast_to(np.asarray(t_query, dtype=float), x_query.shape)
        phi_q = features.eval(np.stack([x_query, y_query], axis=1))  # (Nq, F+1)
        Cu_q = Cu_interp(t_query).T
        Cv_q = Cv_interp(t_query).T
        u = np.sum(Cu_q * phi_q, axis=1)
        v = np.sum(Cv_q * phi_q, axis=1)
        return np.stack([u, v], axis=1)  # (Nq, 2)

    sv, cond = _singular_spectrum(phi)
    diagnostics = {"singular_values": sv, "condition_number": cond,
                   "num_features": m, "basis": "tanh_random_2d",
                   "integrator_success": success, "integrator_message": message}
    return sol, features, predict, diagnostics
