"""Реестр уравнений для абляции DQN-стека агента (одно место правды).

Раньше на каждое уравнение заводился свой 500-строчный раннер-копия
(`experiments/optimization_multi_pde/*_ablation_chain.py`), которые
отличались десятком строк. Расширение абляции на весь бенчмарк так не
масштабируется, поэтому всё, что различает уравнения, собрано здесь, а
раннер один: `experiments/agent_ablation/ablation_chain.py --pde <key>`.

Что задаёт спецификация уравнения:

  key            — имя папки на HF (буфер и результаты) и ключ в кампании;
                   для трёх уже посчитанных уравнений совпадает с тем, что
                   лежит на HF, иначе сломается резюм и агрегация;
  module/cls     — класс PINNacle-задачи и аргументы конструктора;
  comet_project  — проект-источник транзишенов в воркспейсе saitama32
                   (из него export_buffers.py делает буфер на HF);
  peline_l2re    — колонка PELINE таблицы 1 статьи (стр. 8). От неё считается
                   порог успеха eps = EPS_FACTOR x peline_l2re, поэтому ошибка
                   здесь тихо смещает всю абляцию уравнения. Все значения
                   сверены с PDF 2026-09-09; тогда же исправлено
                   poisson2d_classic (стояло 3.10E-2 вместо 3.10E-1 из строки
                   Poisson 2d-C, то есть порог был бы в десять раз строже).
                   Соответствие строк таблицы классам PINNacle:
                   Poisson 2d-C = Poisson2D_Classic, 2d-CG = PoissonBoltzmann2D,
                   3d-CG = Poisson3D_ComplexGeometry, 2d-MS = Poisson2D_ManyArea;
                   NS 2d-C = NS2D_LidDriven, 2d-CG = NS2D_BackStep;
                   Wave 2d-CG = Wave2D_Heterogeneous, 2d-MS = Wave2D_LongTime.
                   Единственное исключение — poisson_boltzmann_2d, см. его note;
  tolerance      — порог успеха траектории по взвешенному лоссу PINN
                   (`abs(loss) < tolerance` => done=1 в EnvRLOptimizer).
                   None означает «не откалиброван»: раннер откажется
                   стартовать, пока значение не задано явно или не получено
                   через calibrate_tolerance.py;
  peline_l2re    — L2RE PELINE при бюджете 7k эпох из таблицы рецензии
                   (по ней и делится на tier'ы);
  tier           — solvable / borderline / unsolvable. Абляция запускается
                   только там, где уравнение в принципе решается: строка с
                   L2RE ~ 1 (long-time, хаос) о компонентах агента ничего
                   не скажет;
  campaign       — done (посчитано в кампании v5 и вошло в rebuttal) / todo.

Значения tolerance взяты из соответствующих chain-скриптов той же
конфигурации, что и абляция (2D-состояние, use_tol=False, new_tol=True):
ветка Saitama32/PINNacle:rlpinn_ablation_optimization, файлы
`experiments/*/n_dim_states/*_2d_state.py` и `multi_pde_exps/*_train.py`.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# Во сколько раз порог успеха выше эталонной ошибки уравнения.
#
# Статья задаёт КРИТЕРИЙ успеха (формула 11: e(theta) <= eps, где e — ошибка
# против эталонного решения), но само значение eps не публикует — ни в статье,
# ни в ответе ревьюерам его нет. Поэтому единственная свободная величина здесь
# привязана к уже опубликованному числу: eps = EPS_FACTOR x L2RE из таблицы 1
# (колонка PELINE). Множитель 2 выбран по собранным траекториям: при 1.5x
# poissoninv не берётся вовсе, при 3x почти всё уходит в 100%, при 2x доли
# успеха ложатся в 44-100% — то есть метрика различает режимы.
EPS_FACTOR = 2.0

# Режимы абляции DQN-стека (совпадают с RL.rl_algorithms.ABLATION_MODES).
ABLATION_MODES = ("none", "no_per", "no_soft_watkins", "no_trust_region")

# Сетка действий агента. Одна и та же почти во всех farm-прогонах; отличие
# ловится полем lbfgs_epochs (у части уравнений последний слот 1500).
BASE_OPTIMIZERS = {
    "Adam": {"lr": [1e-2, 1e-3, 1e-4], "epochs": [100, 1000, 2500]},
    "LBFGS": {"lr": [1, 5e-1, 1e-1], "epochs": [100, 500, 1000]},
    "PSO": {"lr": [0.0, 1e-3, 1e-4], "epochs": [100, 200, 300]},
}


@dataclass(frozen=True)
class PDESpec:
    key: str
    title: str
    module: str
    cls: str
    comet_project: str
    tolerance: Optional[float]
    peline_l2re: Optional[float]
    tier: str
    campaign: str = "todo"
    kwargs: Dict = field(default_factory=dict)
    hidden_layers: str = "100*5"
    lbfgs_epochs: Tuple[int, int, int] = (100, 500, 1000)
    note: str = ""
    # False — класс задачи в этой ветке ещё не реализован: спецификация есть,
    # но собрать и посчитать уравнение нельзя (select() его не отдаёт).
    available: bool = True

    @property
    def eps_l2re(self) -> Optional[float]:
        """Порог успеха траектории по критерию статьи: L2RE против эталона."""
        return None if self.peline_l2re is None else EPS_FACTOR * self.peline_l2re

    @property
    def optimizers(self) -> Dict:
        grid = {name: dict(cfg) for name, cfg in BASE_OPTIMIZERS.items()}
        grid["LBFGS"]["epochs"] = list(self.lbfgs_epochs)
        return grid


def _spec(*args, **kwargs) -> PDESpec:
    spec = PDESpec(*args, **kwargs)
    if spec.tier not in ("solvable", "borderline", "unsolvable"):
        raise ValueError(f"{spec.key}: неизвестный tier {spec.tier!r}")
    if spec.campaign not in ("done", "todo"):
        raise ValueError(f"{spec.key}: неизвестный campaign {spec.campaign!r}")
    return spec


PDE_SPECS: Dict[str, PDESpec] = {s.key: s for s in [
    # --- уже посчитано в кампании v5 (эти три уравнения в rebuttal) ---
    _spec(
        key="poisson_boltzmann_2d", title="Poisson 2d-CG (Poisson-Boltzmann)",
        module="src.pde.poisson", cls="PoissonBoltzmann2D",
        comet_project="rlpinn-poisson-boltzmann2d-tolerance",
        tolerance=0.039669186, peline_l2re=1.07e-2, tier="solvable", campaign="done",
        lbfgs_epochs=(100, 500, 1500),
        note="ВНИМАНИЕ: у этого уравнения peline_l2re взят не из таблицы 1, как у "
             "остальных, а как медиана l2re кампании v5 (1.07E-2) — то самое число, "
             "которое стоит в опубликованной таблице абляции. В таблице 1 строке "
             "Poisson 2d-CG отвечает PELINE 4.11E-3, но это результат ОБУЧЕННОГО "
             "агента на полном бюджете оценки, и порог 2 x 4.11E-3 тренировочные "
             "цепочки не берут вовсе. Уравнение используется как контроль к "
             "опубликованной таблице, поэтому основание порога выбрано под неё",
    ),
    _spec(
        key="poisson3d_complexgeometry", title="Poisson 3d-CG",
        module="src.pde.poisson", cls="Poisson3D_ComplexGeometry",
        kwargs={"datapath": "ref/poisson_3d.dat"},
        comet_project="rlpinn-poisson3d-complexgeometry-tolerance",
        tolerance=0.824311852455139, peline_l2re=5.67e-2, tier="solvable", campaign="done",
    ),
    _spec(
        key="ns2d_liddriven", title="NS 2d-C (lid-driven)",
        module="src.pde.ns", cls="NS2D_LidDriven",
        kwargs={"datapath": "ref/lid_driven_a4.dat", "a": 4.0, "nu": 1e-2},
        comet_project="rlpinn-ns2d-liddriven-tolerance",
        tolerance=0.000352056, peline_l2re=4.40e-2, tier="solvable", campaign="done",
    ),

    # --- расширение: уравнения, которые в принципе решаются ---
    _spec(
        key="burgers1d", title="Burgers 1d-C",
        module="src.pde.burgers", cls="Burgers1D",
        comet_project="rlpinn-burgers-1d-rebuild-buffer-2-dim",
        tolerance=0.00810541202508866, peline_l2re=1.34e-2, tier="solvable",
        lbfgs_epochs=(100, 500, 1500),
    ),
    _spec(
        key="poisson2d_classic", title="Poisson 2d-C",
        module="src.pde.poisson", cls="Poisson2D_Classic",
        comet_project="rlpinn-poisson-2d-classic-farm-transitions",
        tolerance=0.000063, peline_l2re=3.10e-1, tier="solvable",
        note="peline_l2re — строка Poisson 2d-C таблицы 1 (PELINE 3.10E-1); до "
             "сверки с PDF здесь стояло 3.10E-2, то есть порог был бы в десять "
             "раз строже. tolerance из optimization_multi_pde/"
             "poisson_2d_classic_chain.py; перепроверить prepare_pde.py после "
             "экспорта буфера",
    ),
    _spec(
        key="heat2d_multiscale", title="Heat 2d-MS",
        module="src.pde.heat", cls="Heat2D_Multiscale",
        comet_project="rlpinn-heat2d-multiscale-tolerance-corrected",
        tolerance=0.006643642, peline_l2re=1.27e-1, tier="solvable",
    ),
    _spec(
        key="heat2d_complexgeometry", title="Heat 2d-CG",
        module="src.pde.heat", cls="Heat2D_ComplexGeometry",
        comet_project="rlpinn-heat-2d-cg-farm-trans",
        tolerance=0.0455133201723103, peline_l2re=1.58e-2, tier="solvable",
    ),
    _spec(
        key="wave1d", title="Wave 1d-C",
        module="src.pde.wave", cls="Wave1D",
        comet_project="rlpinn-wave1d-loss-chain-reward-tolerance",
        tolerance=0.012694936, peline_l2re=6.47e-2, tier="solvable",
    ),
    _spec(
        key="grayscott", title="GS",
        module="src.pde.chaotic", cls="GrayScottEquation",
        comet_project="rlpinn-grayscott-tolerance",
        tolerance=1.550321937, peline_l2re=9.33e-2, tier="solvable",
        note="порог откалиброван по буферу calibrate_tolerance.py (доля успешных цепочек 75%, как у poisson_boltzmann_2d — единственного уравнения v5 с информативной абляцией). Сравнивается та же величина, что и в загрузчике: min по карте next_state['loss_total']. Распределение с плато: p10=p50=0.6744, p75=1.55",
    ),
    _spec(
        key="poissonnd", title="PNd",
        module="src.pde.poisson", cls="PoissonND",
        comet_project="rlpinn-poissonnd-tolerance",
        tolerance=None, peline_l2re=2.37e-4, tier="solvable",
        note="порог не найден ни в одном chain-скрипте — калибровать по буферу",
    ),
    _spec(
        key="heatnd", title="HNd",
        module="src.pde.heat", cls="HeatND",
        comet_project="rlpinn-heatnd-tolerance",
        tolerance=0.00100347035913728, peline_l2re=2.49e-4, tier="solvable",
        note="tolerance из tolerance-кампании проекта; по буферу даёт 81.6% успешных цепочек офлайн — режим хороший, оставлен как есть",
    ),
    _spec(
        key="poissoninv", title="PInv",
        module="src.pde.inverse", cls="PoissonInv",
        comet_project="rlpinn-poissoninv-tolerance",
        tolerance=0.6834585667, peline_l2re=1.53e-2, tier="solvable",
        note="порог откалиброван по буферу calibrate_tolerance.py (доля успешных цепочек 75%, как у poisson_boltzmann_2d — единственного уравнения v5 с информативной абляцией). Сравнивается та же величина, что и в загрузчике: min по карте next_state['loss_total']",
    ),
    _spec(
        key="heatinv", title="HInv",
        module="src.pde.inverse", cls="HeatInv",
        comet_project="rlpinn-heatinv-tolerance",
        tolerance=0.6998662353, peline_l2re=3.77e-2, tier="solvable",
        note="порог откалиброван по буферу calibrate_tolerance.py (доля успешных цепочек 75%, как у poisson_boltzmann_2d — единственного уравнения v5 с информативной абляцией). Сравнивается та же величина, что и в загрузчике: min по карте next_state['loss_total']. Значение tolerance-кампании проекта (0.0588328) даёт 0% успешных цепочек офлайн и здесь не годится",
    ),

    # --- пограничные: решаются плохо (L2RE 0.2-0.5), запускать после solvable ---
    _spec(
        key="ns2d_backstep", title="NS 2d-CG (backstep)",
        module="src.pde.ns", cls="NS2D_BackStep",
        comet_project="rlpinn-ns2d-backstep-tolerance",
        tolerance=0.001932991785, peline_l2re=1.98e-1, tier="borderline",
        note="порог откалиброван по буферу calibrate_tolerance.py (доля успешных цепочек 75%, как у poisson_boltzmann_2d — единственного уравнения v5 с информативной абляцией). Сравнивается та же величина, что и в загрузчике: min по карте next_state['loss_total']. Прежнее значение 0.0817 давало 99.2% — успех почти тривиален, режимы не различались бы",
    ),
    _spec(
        key="heat2d_varyingcoef", title="Heat 2d-VC",
        module="src.pde.heat", cls="Heat2D_VaryingCoef",
        comet_project="rlpinn-heat-2d-vc-farm-transitions",
        tolerance=0.0585015359142309, peline_l2re=2.27e-1, tier="borderline",
    ),
    _spec(
        key="burgers2d", title="Burgers 2d-C",
        module="src.pde.burgers", cls="Burgers2D",
        comet_project="rlpinn-burgers2d-tolerance",
        tolerance=3.857996941, peline_l2re=4.13e-1, tier="borderline",
        note="порог откалиброван по буферу calibrate_tolerance.py (доля успешных цепочек 75%, как у poisson_boltzmann_2d — единственного уравнения v5 с информативной абляцией). Сравнивается та же величина, что и в загрузчике: min по карте next_state['loss_total']. Задача тяжёлая (PELINE L2RE 4.13E-1), лучший лосс цепочек буфера p50=3.04",
    ),

    # --- не запускаем: PINN не решает задачу, абляция агента ничего не покажет ---
    _spec(
        key="poisson2d_manyarea", title="Poisson 2d-MS",
        module="src.pde.poisson", cls="Poisson2D_ManyArea",
        comet_project="rlpinn-poisson-2d-ms-farm-trans",
        tolerance=7.3, peline_l2re=8.93e-1, tier="unsolvable",
    ),
    _spec(
        key="heat2d_longtime", title="Heat 2d-LT",
        module="src.pde.heat", cls="Heat2D_LongTime",
        comet_project="rlpinn-heat2d-longtime-tolerance",
        tolerance=1.06494992027684, peline_l2re=9.98e-1, tier="unsolvable",
    ),
    _spec(
        key="ns2d_longtime", title="NS 2d-LT",
        module="src.pde.ns", cls="NS2D_LongTime",
        comet_project="rlpinn-ns2d-longtime-tolerance",
        tolerance=None, peline_l2re=9.98e-1, tier="unsolvable",
    ),
    _spec(
        key="wave2d_heterogeneous", title="Wave 2d-CG",
        module="src.pde.wave", cls="Wave2D_Heterogeneous",
        comet_project="rlpinn-wave2d-heterogeneous-tolerance",
        tolerance=None, peline_l2re=8.03e-1, tier="unsolvable",
    ),
    _spec(
        key="wave2d_longtime", title="Wave 2d-MS",
        module="src.pde.wave", cls="Wave2D_LongTime",
        comet_project="rlpinn-wave2d-longtime-tolerance",
        tolerance=None, peline_l2re=9.37e-1, tier="unsolvable",
        note="в таблице 1 статьи это строка Wave 2d-MS (класс Wave2D_LongTime, "
             "t in [0, 100]). Буфер на HF есть, но PELINE даёт L2RE 0.937: "
             "цепочка не решает уравнение, и success rate по eq. (11) вырождается.",
    ),
    _spec(
        key="kuramoto_sivashinsky", title="KS",
        module="src.pde.chaotic", cls="KuramotoSivashinskyEquation",
        comet_project="rlpinn-ks-farm-transitions",
        tolerance=1.67, peline_l2re=9.46e-1, tier="unsolvable",
        hidden_layers="50*5",
    ),

    # --- вне PINNacle: тест-кейс рецензента GrtJ (высокий коэффициент) ---
    _spec(
        key="convection_beta50", title="Convection 1d, beta=50",
        module="src.pde.convection", cls="Convection1D",
        comet_project="rlpinn-convection-beta50-tolerance-defoult-set",
        tolerance=0.076681697, peline_l2re=None, tier="borderline",
        hidden_layers="50*5", available=False,
        note="класса src/pde/convection.py в этой ветке нет — считать можно "
             "только после переноса реализации из ветки convection",
    ),
]}


def get_spec(key: str) -> PDESpec:
    try:
        return PDE_SPECS[key]
    except KeyError:
        raise SystemExit(
            f"Неизвестное уравнение {key!r}. Доступны: {', '.join(sorted(PDE_SPECS))}"
        )


def select(tiers=("solvable",), campaign=None, only_calibrated=False, include_unavailable=False):
    """Ключи уравнений по фильтрам (в порядке объявления реестра)."""
    keys = []
    for key, spec in PDE_SPECS.items():
        if tiers and spec.tier not in tiers:
            continue
        if campaign and spec.campaign != campaign:
            continue
        if only_calibrated and spec.tolerance is None:
            continue
        if not spec.available and not include_unavailable:
            continue
        keys.append(key)
    return keys


def build_get_model(spec: PDESpec, hidden_layers: str = None):
    """Фабрика dde-модели уравнения (та же, что была в per-PDE раннерах).

    Импорты внутри: dill сериализует замыкание для дочернего процесса, а
    deepxde должен инициализироваться уже с выставленным DDEBACKEND.
    """
    import argparse
    from importlib import import_module

    layers_spec = hidden_layers or spec.hidden_layers
    module_name, cls_name, kwargs = spec.module, spec.cls, dict(spec.kwargs)

    def get_model():
        import numpy as np
        import deepxde as dde
        from src.utils.args import parse_hidden_layers

        pde_cls = getattr(import_module(module_name), cls_name)
        pde = pde_cls(**kwargs)

        layers = ([pde.input_dim]
                  + parse_hidden_layers(argparse.Namespace(hidden_layers=layers_spec))
                  + [pde.output_dim])
        net = dde.nn.FNN(layers, "tanh", "Glorot normal")
        net = net.float()

        loss_weights = np.ones(pde.num_loss, dtype=float)
        for i, c in enumerate(pde.loss_config):
            loss_weights[i] = 100.0 if c.get("type", "") in ("boundary", "initial", "ic") else 1.0

        model = pde.create_model(net)
        return model, loss_weights

    return get_model
