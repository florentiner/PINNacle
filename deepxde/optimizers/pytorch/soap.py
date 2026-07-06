import torch


class SOAP(torch.optim.Optimizer):
    """SOAP optimizer: "Improving and Stabilizing Shampoo using Adam" (Vyas et al.,
    2024, https://arxiv.org/abs/2409.11321).

    Runs Adam-style adaptive updates in the eigenbasis of a Shampoo-style (G G^T,
    G^T G) preconditioner, which is refreshed every `precondition_frequency` steps.
    Parameters with more than 2 dimensions are flattened to (dim0, -1); parameters
    with 1 dimension fall back to plain Adam (no matrix structure to precondition)
    unless `precondition_1d` is set.
    """

    def __init__(
        self,
        params,
        lr=3e-3,
        betas=(0.95, 0.95),
        shampoo_beta=0.95,
        eps=1e-8,
        weight_decay=0.01,
        precondition_frequency=10,
        max_precond_dim=10000,
        precondition_1d=False,
        correct_bias=True,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            precondition_1d=precondition_1d,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _to_matrix(grad):
        if grad.dim() == 1:
            return grad.unsqueeze(1), True
        if grad.dim() == 2:
            return grad, False
        return grad.reshape(grad.shape[0], -1), False

    @staticmethod
    def _init_state(state, matrix, precondition_1d, max_precond_dim):
        m, n = matrix.shape
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(matrix)
        state["exp_avg_sq"] = torch.zeros_like(matrix)
        state["GG_L"] = (
            torch.zeros(m, m, device=matrix.device, dtype=matrix.dtype)
            if (m > 1 or precondition_1d) and m <= max_precond_dim
            else None
        )
        state["GG_R"] = (
            torch.zeros(n, n, device=matrix.device, dtype=matrix.dtype)
            if n > 1 and n <= max_precond_dim
            else None
        )
        state["Q_L"] = None
        state["Q_R"] = None

    @staticmethod
    def _update_preconditioner(state, matrix, shampoo_beta):
        if state["GG_L"] is not None:
            state["GG_L"].mul_(shampoo_beta).add_(matrix @ matrix.T, alpha=1 - shampoo_beta)
        if state["GG_R"] is not None:
            state["GG_R"].mul_(shampoo_beta).add_(matrix.T @ matrix, alpha=1 - shampoo_beta)

    @staticmethod
    def _refresh_basis(state):
        if state["GG_L"] is not None:
            eye = torch.eye(state["GG_L"].shape[0], device=state["GG_L"].device, dtype=state["GG_L"].dtype)
            _, state["Q_L"] = torch.linalg.eigh(state["GG_L"] + 1e-30 * eye)
        if state["GG_R"] is not None:
            eye = torch.eye(state["GG_R"].shape[0], device=state["GG_R"].device, dtype=state["GG_R"].dtype)
            _, state["Q_R"] = torch.linalg.eigh(state["GG_R"] + 1e-30 * eye)

    @staticmethod
    def _project(matrix, Q_L, Q_R):
        out = matrix
        if Q_L is not None:
            out = Q_L.T @ out
        if Q_R is not None:
            out = out @ Q_R
        return out

    @staticmethod
    def _project_back(matrix, Q_L, Q_R):
        out = matrix
        if Q_L is not None:
            out = Q_L @ out
        if Q_R is not None:
            out = out @ Q_R.T
        return out

    @staticmethod
    def _adam_step_size(state, beta1, beta2, correct_bias, lr):
        if not correct_bias:
            return lr
        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]
        return lr * (bias_correction2 ** 0.5) / bias_correction1

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            shampoo_beta = group["shampoo_beta"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            precondition_frequency = group["precondition_frequency"]
            max_precond_dim = group["max_precond_dim"]
            precondition_1d = group["precondition_1d"]
            correct_bias = group["correct_bias"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                orig_shape = p.shape
                matrix, was_1d = self._to_matrix(grad)
                state = self.state[p]

                if was_1d and not precondition_1d:
                    if "exp_avg" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(grad)
                        state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["step"] += 1
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = exp_avg_sq.sqrt().add_(eps)
                    step_size = self._adam_step_size(state, beta1, beta2, correct_bias, lr)
                    if weight_decay > 0:
                        p.add_(p, alpha=-lr * weight_decay)
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    continue

                if "exp_avg" not in state:
                    self._init_state(state, matrix, precondition_1d, max_precond_dim)

                self._update_preconditioner(state, matrix, shampoo_beta)
                state["step"] += 1
                if state["step"] == 1 or state["step"] % precondition_frequency == 0:
                    self._refresh_basis(state)

                Q_L, Q_R = state["Q_L"], state["Q_R"]
                grad_proj = self._project(matrix, Q_L, Q_R)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad_proj, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_proj, grad_proj, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)
                update_proj = exp_avg / denom
                update = self._project_back(update_proj, Q_L, Q_R).reshape(orig_shape)

                step_size = self._adam_step_size(state, beta1, beta2, correct_bias, lr)
                if weight_decay > 0:
                    p.add_(p, alpha=-lr * weight_decay)
                p.add_(update, alpha=-step_size)

        return loss
