#!/usr/bin/env python
"""План задач итогового метода для a-dgx (run_jobs.py) по протоколу статьи PINN-PELINE.

Итоговый метод: кодировщик ResNet с MAE на процедурных картах, две сети, модель среды и учитель на журналах трёх
уравнений по 20 цепочек, направленный сбор цели, лучшее из обучений. Протокол статьи (как у абляции агента статьи,
experiments/agent_ablation в ветке rlpinn_agent_ablation): оптимизатор создаётся заново на каждой стадии (среда
без --keep-opt — и при сборе, и при оценке), бюджет 7000 эпох, сиды оценки 101, 202, 303 по одной траектории,
три обученных агента (сиды обучения 1234, 4321, 2718), метрика — полная l2re в конце бюджета; у burgers1d длинный
L-BFGS 1500 эпох (RL_LBFGS_LONG), у KS сеть 50*5, у обратных задач обычная FNN (--plain-fnn).

Задачи одного уравнения <key>:
  col-<key>-<k>     направленный сбор с картами, 2 сборщика по --collect-hours; суррогат выбора цепочек — журналы
                    источников по 20 цепочек (--src-chains), без цели;
  teach-<key>       учитель MBPO на журналах источников и собранном буфере;
  train-<key>-<m>   обучение агента (сид m), модель копируется в /work/<key>/models/;
  sel-<key>-<m>     оценка для выбора обучения: с откатом и мягкой обрезкой, сиды 42, 43 (вне тестовых);
  strict-<key>-<m>  оценка строго как у статьи (без отката и обрезки), сиды 101, 202, 303;
  rules-<key>-<m>   оценка с правилами метода (откат + мягкая обрезка), сиды 101, 202, 303.
Строки оценок пишутся в /work/results (RL_LOCAL_RESULTS) как <pde>_<tag>_seed<s>.json.

    python experiments/rl_arch/v100/plan.py --pdes b1d,hinv,p3dcg,ns2dlt,ks --out jobs.json
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
META = json.load(open(os.path.join(HERE, "..", "track3", "pde_meta.json")))

SRC_A = "burgers1d,poisson2d_classic,ns2d_backstep"
SRC_B = "wave1d,poisson2d_classic,ns2d_backstep"      # цель burgers1d: сам burgers1d в источниках быть не может
PDES = {
    "b1d": dict(pde="burgers_1d", subdir="burgers1d", src="B", env={"RL_LBFGS_LONG": "1500"}, flags=""),
    "hinv": dict(pde="heatinv", subdir="heatinv", src="A", flags="--plain-fnn"),
    "p3dcg": dict(pde="poisson3d_complexgeometry", subdir="poisson3d_complexgeometry", src="A", flags=""),
    # переход у NS 2D longtime и KS в разы дороже: сборщикам больше часов
    "ns2dlt": dict(pde="ns2d_longtime", subdir="ns2d_longtime", src="A", flags="", collect_hours=12.0),
    "ks": dict(pde="kuramoto_sivashinsky", subdir="kuramoto_sivashinsky", src="A", flags="--hidden-layers 50*5",
               collect_hours=12.0),
}
TRAIN_SEEDS = (1234, 4321, 2718)
EVAL_SEEDS = "101,202,303"
SEL_SEEDS = "42,43"
RULES = "--guard-rollback 1.0 --guard-fallback LBFGS:1:500 --stall-guard --stall-window 250 --stall-min 0.01 --stall-opts Adam,PSO"
ENC = "/app/experiments/rl_arch/v100/assets/cvprobe_w55proc_enc_mae_resnet.pt"


def jobs_for(key, a):
    p = PDES[key]
    pde, sub, fl = p["pde"], p["subdir"], p["flags"]
    src = SRC_A if p["src"] == "A" else SRC_B
    table = f"/app/experiments/rl_arch/v100/assets/src_{p['src']}.npz"
    ie = round(float(META[pde]["init_err"]), 3)
    base = f"/work/{key}"
    cwd = f"{base}/run"
    env = dict(p.get("env", {}))
    py = "python -u /app/experiments/rl_arch"
    out = []
    bufs = []
    hrs = p.get("collect_hours", a.collect_hours)
    for k in range(a.collectors):
        tag = f"tbuf_{key}_w{k}"
        bufs.append(f"{base}/collect/buffers_new/{tag}")
        out.append(dict(id=f"col-{key}-{k}", pde=key, cwd=cwd, env=env, needs=[], cmds=[
            f"{py}/collect_active.py --pde {pde} --budget 7000 --rounds 99 --batch 3 --share-prefix --prefix-bonus 0.1 "
            f"--truncate 3 --acq lcb --lcb-k 1 --table {table} --src-chains 20 --src-seed 0 --save-buffer "
            f"--worker {k}/{a.collectors} --hours {hrs} --kernel-hours {hrs} --resume "
            f"--save-dir {base}/collect --tag {tag} {fl}".strip()]))
    col_ids = [j["id"] for j in out]
    buf = ",".join(bufs)
    teach = f"{base}/teach/rl_arch/agents_online/teach_{key}.pt"
    out.append(dict(id=f"teach-{key}", pde=key, cwd=cwd, env=env, needs=col_ids, cmds=[
        f"{py}/agent_combo.py --pde {pde} --mbpo-h 5 --mopo-lam 0.5 --seed 0 --offline-eval none "
        f"--target-buffer {buf} --subdirs {src} --src-chains 20 --src-seed 0 --table {table} --tag teach_{key} "
        f"--out-dir {base}/teach --no-upload"]))
    for m in TRAIN_SEEDS:
        mtag = f"v100{key}"
        model = f"{base}/models/agent_{key}_seed{m}.pt"
        out.append(dict(id=f"train-{key}-{m}", pde=key, cwd=cwd, env=env, needs=[f"teach-{key}"], cmds=[
            f"{py}/offline_rl.py --variant convnext_dqn --subdir {sub} --chain-fix --reward-form delta --init-err {ie} "
            f"--value-bound --gamma 0.99 --episode-budget 7000 --ctx-kmax 70 --save-model --scalar-ctx --encoder resnet "
            f"--encoder-init {ENC} --mb-aux 0.5 --mb-h 3 --mb-lam 0.5 --mb-subdirs {src} --mb-src-chains 20 "
            f"--mb-src-seed 0 --mb-table {table} --target-buffer {buf} --teacher-from {teach} --teacher-w 0.5 "
            f"--two-nets --seeds {m} --model-tag {mtag}",
            f"mkdir -p {base}/models && cp $(ls -t /tmp/agent_convnext_dqn*_{mtag}_seed{m}.pt | head -1) {model}"]))
        for kind, seeds, rules in (("sel", SEL_SEEDS, RULES), ("strict", EVAL_SEEDS, ""), ("rules", EVAL_SEEDS, RULES)):
            out.append(dict(id=f"{kind}-{key}-{m}", pde=key, cwd=cwd, env=env, needs=[f"train-{key}-{m}"], cmds=[
                f"{py}/online_eval_env.py --policy agent --model-file {model} {rules} --pde {pde} --budget 7000 "
                f"--resume --ckpt-every 10 --seeds {seeds} --tag v100f_{key}_m{m}_{kind} --save-dir {base}/eval {fl}"
                .replace("  ", " ").strip()]))
    return out


def smokify(j):
    """Короткая версия задачи: та же цепочка вызовов на крошечных бюджетах."""
    out = []
    for c in j["cmds"]:
        if "collect_active.py" in c:
            c = (c.replace("--budget 7000", "--budget 300").replace("--rounds 99", "--rounds 1")
                 .replace("--batch 3", "--batch 2") + " --smoke")
        elif "agent_combo.py" in c:
            c += " --smoke"
        elif "offline_rl.py" in c:
            # без --smoke: в этом режиме тренер модель не сохраняет, а она нужна оценкам
            c = c.replace("--episode-budget 7000", "--episode-budget 300") + " --epochs 2 --fqe-epochs 2"
        elif "online_eval_env.py" in c:
            c = (c.replace("--budget 7000", "--budget 300").replace(f"--seeds {EVAL_SEEDS}", "--seeds 101")
                 .replace(f"--seeds {SEL_SEEDS}", "--seeds 42"))
        import re
        c = re.sub(r"--hours [0-9.]+ --kernel-hours [0-9.]+", "--hours 0.5 --kernel-hours 0.5", c)
        out.append(c)
    return dict(j, cmds=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdes", default="b1d,hinv,p3dcg,ns2dlt,ks")
    ap.add_argument("--collectors", type=int, default=2)
    ap.add_argument("--collect-hours", type=float, default=8.0)
    ap.add_argument("--smoke", action="store_true",
                    help="проверка всей цепочки: бюджет 300, 1 сборщик на 1 раунд, 1 короткое обучение, оценки по 1 сиду")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.smoke:
        a.collectors = 1
    keys = [k for k in a.pdes.split(",") if k]
    bad = [k for k in keys if k not in PDES]
    if bad:
        raise SystemExit(f"неизвестные уравнения {bad}; есть {sorted(PDES)}")
    jobs = []
    for k in keys:
        jobs += jobs_for(k, a)
    if a.smoke:
        jobs = [smokify(j) for j in jobs if not any(j["id"].endswith(f"-{m}") for m in TRAIN_SEEDS[1:])]
    # порядок: по уравнениям в порядке --pdes (последние — последними), внутри — сбор, учитель, обучения, оценки;
    # run_jobs берёт первую готовую задачу списка, поэтому ранние уравнения доводятся до конца раньше поздних
    jobs.sort(key=lambda j: keys.index(j["pde"]))
    with open(a.out, "w") as f:
        json.dump(jobs, f, indent=1, ensure_ascii=False)
    print(f"задач {len(jobs)} -> {a.out}")
    for j in jobs:
        print(f"  {j['id']:22s} ждёт {','.join(j['needs']) or '-'}")


if __name__ == "__main__":
    main()
