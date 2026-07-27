# SOAP и Muon для PINN: ресёрч перед интеграцией (2026-07-27)

Ответ на фидбек рецензента («сравниться на SOAP и Muon») и на вопросы коллег.
Все утверждения ниже проверены по первоисточникам (статьи + официальный код).

## 1. Что это за оптимизаторы

### SOAP (Vyas et al., "SOAP: Improving and Stabilizing Shampoo using Adam")
- Идея: Shampoo строит для каждой матрицы весов W (m×n) два предобуславливателя
  GG^T (m×m) и G^TG (n×n); SOAP берёт их **собственный базис** и крутит внутри
  этого базиса обычный **Adam**, периодически обновляя базис (QR/eig раз в
  `precondition_frequency` шагов).
- **Это НЕ «надстройка, которой нужен отдельный Adam»** — опасение коллег не
  подтверждается. Официальная реализация — самостоятельный
  `torch.optim.Optimizer` (класс `SOAP`), Adam «встроен» внутрь алгоритма.
  Никакого базового оптимизатора пользователь не передаёт и предварительный
  прогон Adam не требуется: в PINN-статье (см. §3) SOAP обучает с нуля.
- Официальный код: https://github.com/nikhilvyas/SOAP (`soap.py`, один файл).
  Сигнатура: `SOAP(params, lr=3e-3, betas=(0.95, 0.95), shampoo_beta=-1,
  eps=1e-8, weight_decay=0.01, precondition_frequency=10, max_precond_dim=10000,
  merge_dims=False, precondition_1d=False, ...)`.
- Формы параметров: 2D матрицы предобуславливаются с двух сторон; **1D bias по
  умолчанию идут чистым Adam** (`precondition_1d=False`) — т.е. никакого
  ручного разбиения параметров не нужно, кладём все параметры сети.
- Память/стоимость на нашей сети (FNN 100×5): GG (100×100)×2 + базис Q на слой —
  копейки; шаг ≈ 1.5–2× дороже Adam (подтверждено PINN-статьёй: «~2x longer
  training time», Table 7).

### Muon (Keller Jordan et al.)
- Идея: momentum-градиент матрицы весов **ортогонализуется** (приближение
  UV^T через 5 итераций Ньютона–Шульца) и применяется как апдейт; масштаб
  `sqrt(max(1, m/n))`. Работает **именно на 2D матрицах** скрытых слоёв.
- **Опасение коллег «Muon не работает с полносвязными сетями / нужны
  более высокоразмерные тензоры» — неверно, ровно наоборот.** Muon создан для
  2D hidden-матриц; conv-фильтры (4D) он сам решейпит в 2D. Ошибка, о которой
  они слышали, возникает на **1D/скалярных** параметрах (bias, gains,
  embeddings) — их по дизайну оптимизирует вспомогательный AdamW. Наша FNN
  100×5 — это 6 матриц 2D + 6 bias 1D: идеальный случай.
- Официальный код: https://github.com/KellerJordan/Muon (`muon.py`, один файл).
  Для нашего случая (один GPU, без torch.distributed) — класс
  **`SingleDeviceMuonWithAuxAdam`**: параметр-группы с флагом `use_muon`.
  Дефолты Muon: lr=0.02, momentum=0.95, nesterov, ns_steps=5, wd=0;
  aux-Adam: lr=3e-4, betas=(0.9, 0.95), eps=1e-10.
- Разбиение параметров для PINN FNN (рекомендация README: embeddings/heads →
  AdamW): **hidden 100×100 (4 шт.) → Muon; первый (in→100), последний (100→out)
  и все bias → aux AdamW.** Вариант «все 2D в Muon» оставить как абляцию.
- «Нужен Adam сначала» — тоже не требуется: Muon обучает с нуля (aux AdamW —
  это параллельная оптимизация *других* параметров, не предварительная фаза).

## 2. Прецеденты в PINN-литературе (что, вероятно, имеет в виду рецензент)

### Главный: "Gradient Alignment in Physics-informed Neural Networks:
### A Second-Order Optimization Perspective" — NeurIPS 2025 (arXiv 2502.00604)
- Sifan Wang, Bhartari, Li, Perdikaris. Показывают, что конфликт градиентов
  между лоссами PINN снимается (квази)второпорядковыми методами, и **SOAP даёт
  SOTA на 10 бенчмарках** (Wave, Burgers, Allen–Cahn, KdV,
  Kuramoto–Sivashinsky, Grey–Scott, Ginzburg–Landau, lid-driven cavity Re=5000,
  Kolmogorov flow Re=10 000, Rayleigh–Taylor), улучшения 2–14×.
- **Baselines в статье включают и Muon, и Kron** — то есть «SOAP и Muon как
  бейзлайны для PINN» — уже установленная практика; наш рецензент почти
  наверняка отсылает сюда.
- Их конфиг SOAP (jaxpi, ветка pirate, examples/*/configs/soap.py):
  lr=1e-3, betas=(0.9, 0.999), warmup 5000, экспоненциальный decay 0.9/5000,
  300k шагов, batch 8192, PirateNet (256×3). Абляции статьи:
  precondition_frequency оптимально ~2 (дальше выигрыш исчезает), β₁=0.99
  лучше всего.
- Реализация у них JAX; для нас — официальный torch `soap.py` (тот же алгоритм).
- **Convection в этой статье НЕТ.**

### Muon на PINN: "Muon with Spectral Guidance" (SpecMuon, arXiv 2602.16167, Purdue, 02.2026)
- Применяют ванильный Muon и свой SpecMuon к PINN (1D Burgers, 2D heat,
  fractional PDE + DeepONet). **Ванильный Muon уже обгоняет Adam** (~1.7× по
  лоссу на Burgers; Muon lr=0.02, Adam lr=5e-3).
- Отмечают ограничение Muon: апдейты с единичными сингулярными числами могут
  быть чересчур агрессивными для physics-informed лоссов (мотивация их метода) —
  наш прогон это либо подтвердит, либо нет; для нас это просто бейзлайн.

### Convection β=30/50
- Это бенчмарк из Krishnapriyan et al., "Characterizing possible failure modes
  in PINNs" (NeurIPS 2021): u_t + β u_x = 0, β∈{30,50}, периодика, u(0,x)=sin x.
- **Опубликованной работы, где SOAP/Muon гоняют именно convection β=30/50, не
  нашлось** (проверено поиском 27.07.2026). Так что если рецензент упоминал
  convection — это, скорее всего, требование добавить SOAP/Muon в *наши*
  convection-эксперименты, а не отсылка к готовой статье. В PINNacle
  convection нет; реализуется тривиально (1D PDE + периодические BC), если
  нужно будет отвечать именно на этом бенчмарке.

## 3. Интеграция в наш chain_eval (план, код не написан)

1. **Вендорим два файла** (оба MIT): `soap.py` (nikhilvyas/SOAP) и `muon.py`
   (KellerJordan/Muon) в `experiments/chain_eval/vendor/`. Без pip-зависимостей
   (Kaggle-кернелы клонируют ветку — стабильность версий).
2. `build_stage_optimizer` получает два новых типа стадий:
   - `{"optimizer": "SOAP", "lr": ..., "epochs": ...}` →
     `SOAP(model.net.parameters(), lr=..., precondition_frequency=10)`;
     опционально ключи betas/wd/freq из стадии.
   - `{"optimizer": "Muon", "lr": ..., "epochs": ...}` →
     `SingleDeviceMuonWithAuxAdam([{hidden 2D, use_muon=True, lr},
     {остальное, use_muon=False, lr=3e-4}])`.
3. deepxde совместимость: наш вендоренный deepxde принимает инстансы
   `torch.optim.Optimizer` напрямую (optimizers.py:18) и зовёт
   `self.opt.step(closure)` (model.py:346) — Adam-инстансы так уже работали.
   Проверить, что `step(closure=None)` есть в сигнатурах обоих (у обоих
   стандартная конвенция; если нет — обёртка в 3 строки).
4. **T4-ловушка:** Ньютон–Шульц в muon.py работает в bfloat16, а Kaggle T4
   (sm_75, Turing) не имеет аппаратного bf16 — возможен креш или замедление.
   Решение: патч в вендоренном muon.py — fp32 для NS на pre-Ampere GPU
   (матрицы 100×100, стоимость нулевая).
5. Локальный смоук (CPU, 2-3 эпохи на стадию) → Kaggle-смоук → полный запуск.

## 4. Предлагаемая матрица экспериментов для ответа рецензенту

Бюджет-матчинг с нашим adam+lbfgs бейзлайном (31k шагов, сиды 42–51):

| Вариант | Цепочка | Куда | Стоимость (оценка) |
|---|---|---|---|
| SOAP-only | `[{SOAP, lr 3e-3, 31000}]` | csv_chain, chain_key=`soap` | ~1.5–2× Adam ≈ 60–90 GPU-ч на 22 PDE |
| Muon-only | `[{Muon, lr 0.02, 31000}]` | csv_chain, chain_key=`muon` | ~1.1× Adam ≈ 40–50 GPU-ч |
| Adam→SOAP | `[{Adam 1e-3, 1000}, {SOAP, 30000}]` | опция, наша инфраструктура умеет из коробки | как SOAP-only |
| Adam→Muon | аналогично | опция | как Muon-only |

- Минимальный ответ рецензенту: SOAP-only + Muon-only на тех же 22 PDE × 10
  сидов (или сперва на 5 PDE из статьи — 15–25 GPU-ч, за один вечер).
- lr-чувствительность: SOAP {1e-3, 3e-3}, Muon {0.02, 0.05} на 1–2 PDE перед
  полным прогоном (по 1 сиду) — статьи используют разные lr, дёшево снять риск.
- Отчёт по метрикам — та же схема CSV, что и остальные наборы.

## 5. Ответы на вопросы коллег (кратко)

1. *«SOAP — надстройка над Adam, непонятен базовый оптимизатор»* — нет: SOAP
   самостоятельный оптимизатор (Adam внутри его собственного базиса — часть
   алгоритма, а не внешняя зависимость). Никакой предварительный Adam не нужен.
2. *«Есть ли статья, где SOAP и Muon применяют к convection β=30/50»* — не
   нашлась. Ближайшее: NeurIPS-2025 SOAP-статья (10 других PDE, baselines
   включают Muon/Kron) и SpecMuon (Burgers/heat/fractional).
3. *«Muon не работает с полносвязными сетями, нужны тензоры выше 2D»* —
   наоборот: Muon работает ИМЕННО на 2D матрицах; ошибка возникает на 1D
   (bias и т.п.), которые по дизайну уходят в aux-AdamW
   (`SingleDeviceMuonWithAuxAdam` делает это штатно).
4. *«Им нужен Adam или что-то такое сначала»* — не нужен: обе статьи обучают
   с нуля. (Но цепочки Adam→SOAP/Muon мы можем прогнать бесплатно — это ровно
   наш фреймворк.)

## Источники
- SOAP-PINN: https://arxiv.org/abs/2502.00604 (NeurIPS 2025),
  код: https://github.com/PredictiveIntelligenceLab/jaxpi/tree/pirate
- SOAP optimizer: https://github.com/nikhilvyas/SOAP
- Muon: https://github.com/KellerJordan/Muon (+ https://kellerjordan.github.io/posts/muon/)
- SpecMuon: https://arxiv.org/abs/2602.16167
- Convection benchmark: Krishnapriyan et al., NeurIPS 2021 (arXiv 2109.01050)
