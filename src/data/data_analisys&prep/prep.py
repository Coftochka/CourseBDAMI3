import pandas as pd

T_FROM = pd.Timestamp("2015-01-01")
T_TO   = pd.Timestamp("2026-03-06")


def clip_timerange(
    dfs: dict[str, pd.DataFrame] | pd.DataFrame,
    t_from: pd.Timestamp = T_FROM,
    t_to: pd.Timestamp = T_TO,
    time_col: str = "timestamp",
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    """
    Обрезает один или словарь DataFrame'ов по промежутку [t_from, t_to].

    Параметры
    ----------
    dfs      : один DataFrame или dict {ticker: DataFrame}
    t_from   : начало промежутка (включительно), по умолчанию T_FROM
    t_to     : конец промежутка (включительно), по умолчанию T_TO
    time_col : колонка с временем (если она есть); если в df DatetimeIndex — игнорируется
    """
    def _clip(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.index, pd.DatetimeIndex):
            return df.loc[(df.index >= t_from) & (df.index <= t_to)].copy()
        return df[(df[time_col] >= t_from) & (df[time_col] <= t_to)].reset_index(drop=True)

    if isinstance(dfs, dict):
        return {ticker: _clip(df) for ticker, df in dfs.items()}
    return _clip(dfs)


def fill_forward(
    dfs: dict[str, pd.DataFrame] | pd.DataFrame,
    time_col: str = "timestamp",
    limit: int | None = None,
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    """
    Переиндексирует каждый DataFrame по объединению всех торговых дат
    (union всех timestamp'ов), затем заполняет пропуски предыдущим значением.

    Параметры
    ----------
    dfs      : один DataFrame или dict {ticker: DataFrame}
    time_col : колонка с временем (если DatetimeIndex — игнорируется)
    limit    : максимальное количество подряд идущих пропусков для заполнения
    """
    # для одиночного df просто ffill без переиндексации
    if isinstance(dfs, pd.DataFrame):
        return dfs.ffill(limit=limit)

    # строим union всех торговых меток времени
    all_ts = pd.DatetimeIndex(
        pd.concat(
            [
                pd.Series(df.index if isinstance(df.index, pd.DatetimeIndex) else df[time_col])
                for df in dfs.values()
            ]
        ).drop_duplicates().sort_values()
    )

    result = {}
    for ticker, df in dfs.items():
        # выставляем индекс = timestamp
        if isinstance(df.index, pd.DatetimeIndex):
            indexed = df
        else:
            indexed = df.set_index(time_col)

        # переиндексируем по union-календарю — появятся NaN в пропущенные дни
        reindexed = indexed.reindex(all_ts)

        # заполняем пропуски предыдущим значением
        filled = reindexed.ffill(limit=limit)

        result[ticker] = filled

    return result
