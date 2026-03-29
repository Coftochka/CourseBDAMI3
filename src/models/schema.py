"""
Схема входа для моделей под минутные свечи (parquet вида SBER_BASIC_FULL).

Исходная таблица содержит колонки:
  open, close, high, low, value (оборот), volume, begin (timestamp).

В матрицу признаков X для fit/predict попадают только числовые колонки в фиксированном
порядке; время `begin` используется для сортировки и сплитов, в X не входит.
"""

from typing import Final, Tuple

# Порядок столбцов в X — (n_timesteps, len(FEATURE_COLS)), float32 после препроцесса.
FEATURE_COLS: Final[Tuple[str, ...]] = (
    "open",
    "high",
    "low",
    "close",
    "value",
    "volume",
)

TIME_COLUMN: Final[str] = "begin"

INPUT_SIZE: Final[int] = len(FEATURE_COLS)
