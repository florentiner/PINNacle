"""
The 22 PINNacle PDEs for chain evaluation — names only, no heavy imports.

Names match the optuna experiment names (and the historical CSV names in the
HF dataset). Values: (module, class name, constructor kwargs).
"""

PDE_SPECS = {
    # Burgers 1d-C / 2d-C
    "burgers_1d": ("src.pde.burgers", "Burgers1D", {}),
    "burgers_2d": ("src.pde.burgers", "Burgers2D", {}),
    # Poisson 2d-C / 2d-CG / 3d-CG / 2d-MS
    "poisson2d_classic": ("src.pde.poisson", "Poisson2D_Classic", {}),
    "poissonboltzmann2d": ("src.pde.poisson", "PoissonBoltzmann2D", {}),
    "poisson3d_complexgeometry": ("src.pde.poisson", "Poisson3D_ComplexGeometry", {}),
    "poisson2d_manyarea": ("src.pde.poisson", "Poisson2D_ManyArea", {}),
    # Heat 2d-VC / 2d-MS / 2d-CG / 2d-LT
    "heat2d_varyingcoef": ("src.pde.heat", "Heat2D_VaryingCoef", {}),
    "heat2d_multiscale": ("src.pde.heat", "Heat2D_Multiscale", {}),
    "heat2d_complexgeometry": ("src.pde.heat", "Heat2D_ComplexGeometry", {}),
    "heat2d_longtime": ("src.pde.heat", "Heat2D_LongTime", {}),
    # NS 2d-C / 2d-CG / 2d-LT
    "ns2d_classic": ("src.pde.ns", "NS2D_Classic", {}),
    "ns2d_backstep": ("src.pde.ns", "NS2D_BackStep", {}),
    # каверна с подвижной крышкой: класс в репозитории есть (ref/lid_driven_a4.dat),
    # в реестре его не было, хотя именно на ней собран буфер ns2d_liddriven
    "ns2d_liddriven": ("src.pde.ns", "NS2D_LidDriven", {}),
    "ns2d_longtime": ("src.pde.ns", "NS2D_LongTime", {}),
    # Wave 1d-C / 2d-CG / 2d-MS
    "wave1d": ("src.pde.wave", "Wave1D", {}),
    "wave2d_heterogeneous": ("src.pde.wave", "Wave2D_Heterogeneous", {}),
    "wave2d_longtime": ("src.pde.wave", "Wave2D_LongTime", {}),
    # Chaotic GS / KS
    "grayscott": ("src.pde.chaotic", "GrayScottEquation", {}),
    "kuramoto_sivashinsky": ("src.pde.chaotic", "KuramotoSivashinskyEquation", {}),
    # High-dim PNd / HNd
    "poissonnd": ("src.pde.poisson", "PoissonND", {"dim": 5}),
    "heatnd": ("src.pde.heat", "HeatND", {"dim": 5}),
    # Inverse PInv / HInv (use pde.recommend_net)
    "poissoninv": ("src.pde.inverse", "PoissonInv", {}),
    "heatinv": ("src.pde.inverse", "HeatInv", {}),
}

INVERSE_PDE_NAMES = {"poissoninv", "heatinv"}

ALL_PDE_NAMES = list(PDE_SPECS)
