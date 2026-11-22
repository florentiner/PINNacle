#!/usr/bin/env bash
# Итоговый метод на a-dgx (V100) — запуск с мака, по правилам A_DGX_GUIDE.txt (ветка абляции):
# только наша карта, контейнер без сети, ограничение CPU/памяти/потоков, имена rlpinn-*, тома rlpinn_*.
#
#   experiments/rl_arch/v100/dgx.sh build                         собрать образ на a-dgx (код + ресурсы из v100/assets)
#   experiments/rl_arch/v100/dgx.sh start <имя> <plan.py флаги>   план -> том rlpinn_<имя> -> контейнер rlpinn-<имя>
#   experiments/rl_arch/v100/dgx.sh status <имя>                  контейнер, сводка задач, хвост журнала
#   experiments/rl_arch/v100/dgx.sh log <имя> <задача> [строк]    хвост лога задачи
#   experiments/rl_arch/v100/dgx.sh restart <имя>                 поднять упавший контейнер (готовое не теряется)
#   experiments/rl_arch/v100/dgx.sh pull <имя>                    результаты, состояние и логи -> v100/results_local/<имя>
#   experiments/rl_arch/v100/dgx.sh push <имя>                    строки оценок -> HF rl_arch/online_env/ (нужен интернет)
#   experiments/rl_arch/v100/dgx.sh stop <имя>
# Переменные: GPU (uuid нашей карты), CPUS (8), MEMORY (48g), SLOTS (4 задачи одновременно), THREADS (2 на задачу).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
D=(docker --context a-dgx)
GPU="${GPU:-GPU-226cf391-f2ba-abce-add6-6a7f2a6bdd81}"
OURS="GPU-c4d77630-d224-c192-f2f7-33ffec53d852 GPU-be62e07b-a4b5-bafc-48e5-88ca609fb6d4 GPU-226cf391-f2ba-abce-add6-6a7f2a6bdd81"
CPUS="${CPUS:-8}"; MEMORY="${MEMORY:-48g}"; SLOTS="${SLOTS:-4}"; THREADS="${THREADS:-2}"
TAGFILE="$ROOT/experiments/rl_arch/v100/results_local/image_tag"
PY="${PYTHON:-/opt/homebrew/opt/python@3.11/libexec/bin/python3}"

die() { echo "ОШИБКА: $*" >&2; exit 1; }
need_vpn() { nc -z -G 5 10.32.2.3 2376 >/dev/null 2>&1 || die "a-dgx недоступен — включи VPN"; }
image() { [ -f "$TAGFILE" ] || die "образа нет — сначала build"; cat "$TAGFILE"; }
helper() { "${D[@]}" run --rm -i --network none --runtime runc -e NVIDIA_VISIBLE_DEVICES=void "$@"; }

cmd="${1:-}"; shift || true
case "$cmd" in
build)
  need_vpn
  case " $OURS " in *" $GPU "*) ;; *) die "карта $GPU не наша";; esac
  for f in src_A.npz src_B.npz cvprobe_w55proc_enc_mae_resnet.pt; do
    [ -f "$ROOT/experiments/rl_arch/v100/assets/$f" ] || die "нет ресурса v100/assets/$f"
  done
  sha="$(git -C "$ROOT" rev-parse --short HEAD)"; [ -z "$(git -C "$ROOT" status --porcelain -- experiments src RL deepxde)" ] || sha="$sha-dirty"
  tag="rlpinn-final-v100:$sha-$(date +%m%d%H%M)"
  echo "сборка $tag (контекст: код + ref/ + v100/assets)"
  (cd "$ROOT" && "${D[@]}" build -f experiments/rl_arch/v100/Dockerfile --build-arg GIT_COMMIT="$sha" -t "$tag" .)
  mkdir -p "$(dirname "$TAGFILE")"; echo "$tag" > "$TAGFILE"; echo "образ $tag"
  ;;
start)
  need_vpn
  name="${1:?имя запуска, например final}"; shift
  img="$(image)"; vol="rlpinn_${name//-/_}"; ctr="rlpinn-${name}"
  "${D[@]}" ps -a --format '{{.Names}}' | grep -qx "$ctr" && die "контейнер $ctr уже есть (status/restart/stop)"
  used="$("${D[@]}" ps --format '{{.Names}}' | while read -r c; do
            "${D[@]}" inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$c" 2>/dev/null | grep -q "NVIDIA_VISIBLE_DEVICES=$GPU" && echo "$c"; done || true)"
  [ -z "$used" ] || echo "ВНИМАНИЕ: на карте $GPU уже работают: $used"
  tmp="$(mktemp -d)"; "$PY" "$ROOT/experiments/rl_arch/v100/plan.py" --out "$tmp/jobs.json" "$@"
  (cd "$tmp" && tar cf - jobs.json) | helper -v "$vol:/work" --entrypoint tar "$img" xf - -C /work
  "${D[@]}" run -d --name "$ctr" --runtime nvidia -e NVIDIA_VISIBLE_DEVICES="$GPU" --network none \
    --cpus "$CPUS" --memory "$MEMORY" --shm-size 8g --restart on-failure:5 --stop-timeout 120 \
    -e RL_LOCAL_RESULTS=/work/results -e RUN_SLOTS="$SLOTS" -e RUN_THREADS="$THREADS" \
    -e OMP_NUM_THREADS="$THREADS" -e MKL_NUM_THREADS="$THREADS" -e HF_HUB_OFFLINE=1 \
    -v "$vol:/work" "$img" >/dev/null
  echo "запущен $ctr (том $vol, карта $GPU, cpus $CPUS, память $MEMORY, задач одновременно $SLOTS)"
  ;;
status)
  need_vpn
  name="${1:?имя}"; ctr="rlpinn-${name}"
  "${D[@]}" ps -a --filter "name=^${ctr}$" --format '{{.Names}}: {{.Status}}'
  "${D[@]}" inspect -f 'перезапусков: {{.RestartCount}}, код выхода: {{.State.ExitCode}}' "$ctr" 2>/dev/null || true
  helper -v "rlpinn_${name//-/_}:/work:ro" --entrypoint python "$(image)" - <<'EOF'
import json, os, collections
st = json.load(open("/work/state.json")) if os.path.exists("/work/state.json") else {}
c = collections.Counter(s["state"] for s in st.values())
print("задачи:", dict(c))
for k, s in sorted(st.items(), key=lambda kv: kv[1].get("start", "")):
    if s["state"] in ("running", "failed"):
        print(f"  {s['state']:8s} {k:24s} попыток {s['attempts']} старт {s.get('start', '')} rc {s.get('rc', '')}")
if os.path.exists("/work/heartbeat.txt"):
    print("heartbeat:", open("/work/heartbeat.txt").read().strip())
res = sorted(os.listdir("/work/results")) if os.path.isdir("/work/results") else []
print("строк оценок:", len(res))
EOF
  "${D[@]}" logs --tail 5 "$ctr" 2>&1 | sed 's/^/  | /'
  ;;
log)
  need_vpn
  name="${1:?имя}"; job="${2:?задача}"; n="${3:-40}"
  helper -v "rlpinn_${name//-/_}:/work:ro" --entrypoint tail "$(image)" -n "$n" "/work/logs/$job.log"
  ;;
restart)
  # поднять остановившийся контейнер; RESET_FAILED=1 — упавшие задачи снова в очередь (после правки причины)
  need_vpn
  name="${1:?имя}"; ctr="rlpinn-${name}"
  if [ "${RESET_FAILED:-0}" = "1" ]; then
    helper -v "rlpinn_${name//-/_}:/work" --entrypoint python "$(image)" - <<'EOF'
import json
p = "/work/state.json"; st = json.load(open(p))
n = 0
for s in st.values():
    if s["state"] == "failed":
        s.update(state="pending", attempts=0); n += 1
json.dump(st, open(p, "w"), indent=1); print("в очередь снова:", n)
EOF
  fi
  "${D[@]}" start "$ctr" >/dev/null; echo "поднят $ctr"
  ;;
pull)
  need_vpn
  name="${1:?имя}"; out="$ROOT/experiments/rl_arch/v100/results_local/$name"; mkdir -p "$out"
  helper -v "rlpinn_${name//-/_}:/work:ro" --entrypoint sh "$(image)" -c \
    'cd /work && tar cf - results state.json heartbeat.txt logs $(ls -d */collect/*_collect.json */teach/rl_arch/online_env 2>/dev/null)' \
    | tar xf - -C "$out"
  echo "забрано в $out: строк $(ls "$out/results" 2>/dev/null | wc -l | tr -d ' ')"
  ;;
push)
  name="${1:?имя}"; out="$ROOT/experiments/rl_arch/v100/results_local/$name/results"
  [ -d "$out" ] || die "сначала pull"
  "$PY" - "$out" "$ROOT/experiments/chain_eval/accounts.json" <<'EOF'
import json, os, sys
from huggingface_hub import HfApi, CommitOperationAdd
src, acc = sys.argv[1], sys.argv[2]
tok = json.load(open(acc)).get("hf_token_write") or os.environ.get("HF_TOKEN_WRITE")
if not tok:
    sys.exit("нет токена записи HF (accounts.json hf_token_write или HF_TOKEN_WRITE)")
api = HfApi()
repo = "danil-e/pinnacle-optuna-db"
have = {f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith("rl_arch/online_env/")}
ops = []
for f in sorted(os.listdir(src)):
    if not f.endswith(".json") or f.startswith("."):
        continue
    d = json.load(open(os.path.join(src, f)))
    if d.get("partial") or d.get("unfinished"):
        continue
    dst = f"rl_arch/online_env/{f}"
    if dst not in have:
        ops.append(CommitOperationAdd(path_in_repo=dst, path_or_fileobj=os.path.join(src, f)))
if ops:
    api.create_commit(repo_id=repo, repo_type="dataset", token=tok, operations=ops,
                      commit_message=f"a-dgx final method: {len(ops)} eval rows")
print(f"выгружено {len(ops)}, уже были {len(have & {'rl_arch/online_env/' + f for f in os.listdir(src)})}")
EOF
  ;;
stop)
  need_vpn
  name="${1:?имя}"; "${D[@]}" stop -t 120 "rlpinn-${name}"
  ;;
*)
  sed -n '2,14p' "$0"; exit 1;;
esac
