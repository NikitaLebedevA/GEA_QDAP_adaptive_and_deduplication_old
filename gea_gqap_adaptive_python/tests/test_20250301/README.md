# Тест 2025-03-01 (test_20250301)

Тест по параметрам статьи GEA for GQAP: **1000 итераций**, **популяция 350**, **лимит времени 1000 с**.  
Датасеты: **все доступные** (c — Cordeau et al., T — Fathollahi-Fard и др.).

На каждую модель (GA, GEA_1, GEA_2, GEA_3, GEA) запускаются **4 типа** алгоритма:

| Тип | Описание |
|-----|----------|
| **adaptive** | Адаптивный GEA с пулом лучших (без дедупликации). |
| **non_adaptive** | Неадаптивный GEA с тем же пулом лучших (без дедупликации). |
| **adaptive_wo_duplicates** | Адаптивный GEA с дедупликацией пула лучших; при нехватке особей — дополнение случайными. |
| **non_adaptive_wo_duplicates** | Неадаптивный GEA с дедупликацией пула лучших; при нехватке — дополнение случайными. |

- По **30 запусков** на каждый тип, на каждый датасет (c и T).
- Параметры задаются в `test_config.json` (Table 1 статьи: crossover 0.7, mutation 0.3, p_scenario1/2/3, time_limit 1000 и т.д.).

## Запуск

```bash
cd gea_gqap_adaptive_python/tests/test_20250301
python3 run_comparison_test.py
```

Лог: `test_output.log`, результаты по датасетам: `results/<dataset>_results.json`, сводка: `results/summary_all_datasets.json`.

## Отчёты и графики

- **Excel** (после прогона):
  ```bash
  python3 create_excel_report.py
  ```
  Файл: `comparison_results.xlsx` (листы: Сводка, best_cost, elapsed_time, paper_table).

- **Графики cost**:
  ```bash
  python3 plot_cost_evolution.py
  ```
  Папка: `cost_plots/`.

- **Графики лямбд** (только для двух адаптивных типов):
  ```bash
  python3 plot_lambda_evolution.py
  ```
  Папка: `lambda_plots/`.

## Параметры (из статьи, Table 1)

| Параметр | Значение |
|----------|----------|
| Maximum number of generations | 1000 |
| Population size | 350 |
| Crossover rate | 0.7 |
| Mutation rate | 0.3 |
| Time limit (max runtime) | 1000 s |
| p% Scenario 1/2/3, fixing gene threshold, scenario rates | см. test_config.json |

Остановка алгоритма — по достижении **1000 итераций** или **1000 секунд** (что наступит раньше).
