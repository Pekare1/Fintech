"""
EDA-сайт для хакатона (обязательный дедлайн).

Запуск локально:
    streamlit run eda_app.py

Деплой (бесплатно, публичная ссылка без доступа к вашему аккаунту):
    1. Залейте эту папку (или весь проект) в публичный репозиторий на GitHub.
    2. Зайдите на https://share.streamlit.io , привяжите GitHub-аккаунт.
    3. Выберите репозиторий и укажите путь к этому файлу (eda_app.py).
    4. Streamlit Cloud даст вам публичный URL вида
       https://<your-app>.streamlit.app — это и есть ссылка для отправки организаторам.

Перед деплоем убедитесь, что requirements.txt лежит рядом с этим файлом
в корне репозитория (Streamlit Cloud ставит зависимости по нему).
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Escalation Alerts — EDA", layout="wide")

DATA_DIR = Path("data")

st.title("EDA: предсказание эскалации алертов мониторинга")

st.markdown(
    """
    ### Краткое описание подхода
    Задача — оценить вероятность эскалации алерта мониторинга по истории
    транзакций клиента (метрика — **ROC-AUC**). Подход: для каждого `signal_id`
    агрегируем связанные с ним транзакции в ~70 числовых признаков (объём
    активности, направление платежей, доли по типам транзакций, статистики
    размера, активность в окнах 1-30 дней перед сигналом, регулярность,
    время суток/недели, тренд суммы во времени), затем обучаем градиентный
    бустинг (LightGBM) с 5-fold кросс-валидацией. Итоговый **OOF ROC-AUC ≈ 0.60**
    — сигнал в данных умеренный и распределён по многим признакам, а не
    сосредоточен в одной явной аномалии (см. раздел 7 ниже).
    """
)

# ---------------------------------------------------------------
# Загрузка данных (кэшируется, чтобы не пересчитывать при каждом клике)
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    train_signals = pd.read_csv(DATA_DIR / "train_signals.csv", parse_dates=["signal_sanasi"])
    train_tx = pd.read_parquet(DATA_DIR / "train_transactions.parquet")
    train_tx["tranzaksiya_vaqti"] = pd.to_datetime(train_tx["tranzaksiya_vaqti"])
    return train_signals, train_tx

train_signals, train_tx = load_data()

# ---------------------------------------------------------------
st.header("1. Обзор датасета")

col1, col2, col3 = st.columns(3)
col1.metric("Алертов в train", f"{len(train_signals):,}")
col2.metric("Транзакций в train", f"{len(train_tx):,}")
col3.metric("Доля эскалированных", f"{train_signals['eskalatsiya'].mean():.1%}")

st.dataframe(train_signals.head())
st.dataframe(train_tx.head())

# ---------------------------------------------------------------
st.header("2. Распределение таргета")

fig, ax = plt.subplots()
train_signals["eskalatsiya"].value_counts().plot(kind="bar", ax=ax)
ax.set_xlabel("eskalatsiya (0 = отклонён, 1 = эскалирован)")
ax.set_ylabel("Количество алертов")
st.pyplot(fig)

st.markdown(
    """
    **Наблюдение:** классы несбалансированы примерно **83% / 17%**
    (отклонено / эскалировано). Для ROC-AUC это не критично, как для accuracy,
    но повлияло на выбор валидации — используем `StratifiedKFold`, чтобы
    сохранить эту пропорцию в каждом фолде и получить честную оценку качества.
    """
)

# ---------------------------------------------------------------
st.header("3. Активность транзакций во времени")

daily_counts = train_tx.set_index("tranzaksiya_vaqti").resample("D").size()
fig, ax = plt.subplots(figsize=(10, 3))
daily_counts.plot(ax=ax)
ax.set_ylabel("Транзакций в день")
st.pyplot(fig)

# ---------------------------------------------------------------
st.header("4. Входящие vs исходящие транзакции")

fig, ax = plt.subplots()
train_tx["kirim_chiqim"].value_counts().plot(kind="bar", ax=ax)
ax.set_ylabel("Количество транзакций")
st.pyplot(fig)

# ---------------------------------------------------------------
st.header("5. Типы транзакций")

fig, ax = plt.subplots()
train_tx["tranzaksiya_turi"].value_counts().plot(kind="bar", ax=ax)
ax.set_ylabel("Количество транзакций")
st.pyplot(fig)

# ---------------------------------------------------------------
st.header("6. Распределение размера транзакции (miqdor_indeksi)")

fig, ax = plt.subplots()
train_tx["miqdor_indeksi"].plot(kind="hist", bins=50, ax=ax)
ax.set_xlabel("miqdor_indeksi")
st.pyplot(fig)

# ---------------------------------------------------------------
st.header("7. Поведение эскалированных vs отклонённых алертов")

st.markdown(
    """
    **Ключевое наблюдение по реальным данным:** различия между эскалированными
    и отклонёнными алертами на уровне отдельных агрегатов небольшие — например,
    среднее число транзакций 523.8 против 494.0, доли по типам транзакций почти
    идентичны (`karta` ≈ 52.1% в обеих группах, `xalqaro` ≈ 0.44-0.45%). Это
    значит, что сигнал не сосредоточен в одном очевидном признаке, а "размазан"
    по комбинации многих слабых признаков — отсюда решение использовать
    градиентный бустинг с широким набором признаков вместо простых правил.
    """
)

merged = train_tx.merge(train_signals[["signal_id", "eskalatsiya", "signal_sanasi"]], on="signal_id", how="left")
agg = merged.groupby(["signal_id", "eskalatsiya"]).agg(
    tx_count=("miqdor_indeksi", "count"),
    amount_sum=("miqdor_indeksi", "sum"),
).reset_index()

fig, ax = plt.subplots()
agg.boxplot(column="tx_count", by="eskalatsiya", ax=ax)
ax.set_title("Количество транзакций по группам")
plt.suptitle("")
st.pyplot(fig)

# ---------------------------------------------------------------
st.header("8. Активность непосредственно перед сигналом")

merged["days_before"] = (merged["signal_sanasi"] - merged["tranzaksiya_vaqti"]).dt.total_seconds() / 86400
window_stats = []
for w in [1, 3, 7, 14, 30]:
    recent = merged[merged["days_before"] <= w]
    cnt = recent.groupby(["signal_id", "eskalatsiya"]).size().reset_index(name="cnt")
    full = train_signals[["signal_id", "eskalatsiya"]].merge(cnt, on=["signal_id", "eskalatsiya"], how="left").fillna(0)
    window_stats.append(full.groupby("eskalatsiya")["cnt"].mean().rename(f"{w}д"))
window_df = pd.concat(window_stats, axis=1).T

fig, ax = plt.subplots()
window_df.plot(kind="bar", ax=ax)
ax.set_ylabel("Среднее число транзакций в окне")
ax.set_xlabel("Окно перед сигналом")
st.pyplot(fig)

st.markdown(
    """
    Разница по окнам 1-30 дней перед сигналом небольшая, но стабильно в одну
    сторону: у эскалированных алертов активность в каждом окне чуть выше
    (например, ~48.5 vs ~46.6 транзакций за 7 дней). Это слабый, но
    последовательный сигнал — именно такие признаки суммарно и дают модели
    прирост качества выше случайного угадывания.
    """
)

# ---------------------------------------------------------------
st.header("9. Признаки, мотивированные EDA")

st.markdown(
    """
    По итогам EDA в модель добавлены:
    - **Доли и суммы по типам транзакций** (`naqd_ratio`, `xalqaro_ratio`,
      `type_amt_naqd`) — редкие типы `naqd` и `xalqaro` оказались среди самых
      важных признаков модели (по `feature_importances_` LightGBM), несмотря на
      небольшую долю в общем объёме транзакций.
    - **Признаки активности перед сигналом** (окна 1/3/7/14/30 дней) — раздел 8
      показал устойчивое, хоть и слабое, повышение активности перед
      эскалированными алертами.
    - **Модуль размера транзакции (`abs_amount_*`)** — поскольку
      `miqdor_indeksi` стандартизирован и может быть отрицательным, сырая сумма
      "гасит" сигнал; модуль оказался информативнее.
    - **Доля ночных и выходных транзакций** — как признак нетипичного паттерна
      поведения.
    - **Тренд суммы транзакций** (вторая половина истории минус первая) — как
      попытка уловить "разгон" активности ближе к дате алерта.
    """
)

# ---------------------------------------------------------------
st.header("10. Выводы")

st.markdown(
    """
    1. Классы несбалансированы (≈17% эскалаций) — учтено через `StratifiedKFold`.
    2. Сигнал в данных умеренный и распределён по многим признакам, а не по
       одной явной аномалии — ни один отдельный агрегат не разделяет классы
       сильно.
    3. Наиболее информативны: доли и суммы по редким типам транзакций
       (`naqd`, `xalqaro`), доля `kirim`/`chiqim`, статистики размера транзакции
       по модулю, активность в ближайшие дни перед сигналом.
    4. Итоговая модель (LightGBM, 5-fold CV) даёт **OOF ROC-AUC ≈ 0.60** —
       стабильно выше случайного угадывания на всех фолдах.
    5. Дальнейшие точки роста: подбор гиперпараметров через Optuna, добавление
       признаков аномальности (отклонение от собственной истории клиента),
       blending нескольких моделей.
    """
)
