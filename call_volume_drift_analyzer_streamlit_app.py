# streamlit_app.py — Call Volume Drift Analyzer & Staffing Helper
# ---------------------------------------------------------------
# Questa app aiuta ad analizzare variazioni di volumi inbound e della loro
# distribuzione settimanale/intra-day, e a riflettere tali pattern nel
# dimensionamento (Erlang C) per livello di servizio.
#
# Come usare:
# 1) `pip install streamlit pandas numpy scipy ruptures pytz` (ruptures è opzionale)
# 2) `streamlit run streamlit_app.py`
# 3) Carica un CSV con una riga per chiamata (colonna "timestamp") oppure
#    un CSV aggregato con slot (colonne tipo "slot_start" e "calls").
#    Timestamp in ora locale o UTC; l'app consente di specificare il fuso.

import io
import math
import json
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Opzionali ma consigliati
try:
    import ruptures as rpt  # change-point detection
except Exception:
    rpt = None

from scipy.stats import chi2_contingency, entropy

# ----------------------------------------
# Utility & Config
# ----------------------------------------
st.set_page_config(page_title="Call Volume Drift Analyzer", layout="wide")
st.title("📞 Call Volume Drift Analyzer & Staffing Helper")

st.sidebar.header("⚙️ Parametri di input")

# Timezone handling
TZ_DEFAULT = "Europe/Rome"

# App state dataclass
@dataclass
class AppConfig:
    tz: str = TZ_DEFAULT
    slot_minutes: int = 30
    business_hours_only: bool = False
    open_hour: int = 8
    close_hour: int = 20

    # Periodi per confronto
    baseline_start: Optional[pd.Timestamp] = None
    baseline_end: Optional[pd.Timestamp] = None
    compare_start: Optional[pd.Timestamp] = None
    compare_end: Optional[pd.Timestamp] = None

    # Staffing params
    aht_sec: int = 240  # Average Handle Time (secondi)
    target_sl: float = 0.8  # es. 80% entro T sec
    target_t_sec: int = 20
    shrinkage: float = 0.3  # assenze, pause, formazione, ecc.
    max_occupancy: float = 0.85  # opzionale: cap occupazione

cfg = AppConfig()

# Sidebar controls
cfg.tz = st.sidebar.text_input("Fuso orario", value=TZ_DEFAULT, help="es. Europe/Rome, UTC, Europe/Paris…")
cfg.slot_minutes = st.sidebar.number_input("Granularità slot (min)", min_value=5, max_value=120, step=5, value=30)

cfg.business_hours_only = st.sidebar.checkbox("Limita ad orario lavorativo", value=False)
if cfg.business_hours_only:
    c1, c2 = st.sidebar.columns(2)
    cfg.open_hour = int(c1.number_input("Apertura (h)", min_value=0, max_value=23, value=8))
    cfg.close_hour = int(c2.number_input("Chiusura (h)", min_value=1, max_value=24, value=20))

# Staffing params
st.sidebar.subheader("📐 Parametri dimensionamento (Erlang C)")
cfg.aht_sec = int(st.sidebar.number_input("AHT (sec)", min_value=30, max_value=3600, value=240, step=10))
cfg.target_sl = float(st.sidebar.slider("Target Service Level (%)", min_value=50, max_value=99, value=80)) / 100.0
cfg.target_t_sec = int(st.sidebar.number_input("T (sec) per SL", min_value=5, max_value=600, value=20, step=5))
cfg.shrinkage = float(st.sidebar.slider("Shrinkage (%)", min_value=0, max_value=60, value=30)) / 100.0
cfg.max_occupancy = float(st.sidebar.slider("Max Occupancy (%)", min_value=60, max_value=100, value=85)) / 100.0

# ----------------------------------------
# Upload dati
# ----------------------------------------
upload = st.file_uploader("Carica CSV (timestamp per chiamata o slot aggregati)", type=["csv"]) 
example_hint = st.expander("📎 Formati supportati (esempi)")
with example_hint:
    st.markdown(
        """
**Opzione A – Per chiamata**

```
call_id,timestamp
123,2024-11-01 09:12:01
124,2024-11-01 09:12:05
```

**Opzione B – Aggregato per slot**

```
slot_start,calls
2024-11-01 09:00:00,42
2024-11-01 09:30:00,35
```
        """
    )

# ----------------------------------------
# Parsing & Normalizzazione
# ----------------------------------------
def _localize(ts: pd.Series, tz: str) -> pd.DatetimeIndex:
    idx = pd.to_datetime(ts, utc=False, errors="coerce")
    # se naive, localizza; se aware, converti
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    else:
        idx = idx.dt.tz_convert(tz)
    return idx


def load_and_normalize(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]

    if "timestamp" in df.columns:
        # Per chiamata: ogni riga = 1 chiamata
        ts = _localize(df["timestamp"], cfg.tz)
        out = pd.DataFrame({"ts": ts, "calls": 1}).dropna(subset=["ts"]) 
    elif {"slot_start", "calls"}.issubset(df.columns):
        ts = _localize(df["slot_start"], cfg.tz)
        out = pd.DataFrame({"ts": ts, "calls": pd.to_numeric(df["calls"], errors="coerce")})
        out = out.dropna(subset=["ts"]).fillna({"calls": 0})
    else:
        raise ValueError("Colonne non riconosciute. Usa 'timestamp' oppure 'slot_start,calls'.")

    # Filtra orari lavorativi se richiesto
    if cfg.business_hours_only:
        out = out[(out["ts"].dt.hour >= cfg.open_hour) & (out["ts"].dt.hour < cfg.close_hour)]

    # Resample a granularità uniforme
    out = (
        out.set_index("ts")
        .sort_index()
        .resample(f"{cfg.slot_minutes}min")["calls"].sum()
        .to_frame()
        .reset_index()
    )
    out["date"] = out["ts"].dt.date
    out["week"] = out["ts"].dt.to_period("W-MON").dt.start_time  # settimana con lunedì start
    out["weekday_num"] = out["ts"].dt.weekday  # 0=Lun
    weekday_map = {0: "Lunedì", 1: "Martedì", 2: "Mercoledì", 3: "Giovedì", 4: "Venerdì", 5: "Sabato", 6: "Domenica"}
    out["weekday"] = out["weekday_num"].map(weekday_map)
    out["hour"] = out["ts"].dt.hour
    return out


if upload is None:
    st.info("⬆️ Carica un CSV per iniziare l'analisi.")
    st.stop()

try:
    data = load_and_normalize(upload)
except Exception as e:
    st.error(f"Errore nel parsing: {e}")
    st.stop()

st.success(f"Dati caricati: {len(data)} righe, intervallo dal {data['ts'].min()} al {data['ts'].max()}")

# Period pickers
c1, c2 = st.columns(2)
with c1:
    st.subheader("Periodo Baseline")
    bmin, bmax = data["ts"].min(), data["ts"].max()
    bstart = st.date_input("Inizio baseline", value=bmin.date())
    bend = st.date_input("Fine baseline", value=min(bmax.date(), bstart))
with c2:
    st.subheader("Periodo Confronto")
    cstart = st.date_input("Inizio confronto", value=bend)
    cend = st.date_input("Fine confronto", value=data["ts"].max().date())

cfg.baseline_start = pd.Timestamp(bstart).tz_localize(cfg.tz)
cfg.baseline_end = pd.Timestamp(bend).tz_localize(cfg.tz) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
cfg.compare_start = pd.Timestamp(cstart).tz_localize(cfg.tz)
cfg.compare_end = pd.Timestamp(cend).tz_localize(cfg.tz) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

# Clip data to windows
base_df = data[(data["ts"] >= cfg.baseline_start) & (data["ts"] <= cfg.baseline_end)].copy()
comp_df = data[(data["ts"] >= cfg.compare_start) & (data["ts"] <= cfg.compare_end)].copy()

# ----------------------------------------
# Funzioni analitiche
# ----------------------------------------

def weekly_totals(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("week", as_index=False)["calls"].sum().rename(columns={"calls": "weekly_calls"})


def weekday_weights_by_week(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["week", "weekday_num"], as_index=False)["calls"].sum()
    w = g.merge(g.groupby("week")["calls"].sum().rename("week_sum"), on="week")
    w["weight"] = w["calls"] / w["week_sum"].replace(0, np.nan)
    return w


def daily_shape(df: pd.DataFrame) -> pd.DataFrame:
    # Distribuzione intra-day media per weekday (normalizzata per giorno)
    g = df.groupby(["weekday_num", "hour"], as_index=False)["calls"].sum()
    g = g.merge(g.groupby("weekday_num")["calls"].sum().rename("day_sum"), on="weekday_num")
    g["p"] = g["calls"] / g["day_sum"].replace(0, np.nan)
    return g


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    # require prob distributions, same length, handle zeros
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
    q = q / q.sum() if q.sum() > 0 else np.ones_like(q) / len(q)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m, base=2) + entropy(q, m, base=2))


def chi_square_dow(base: pd.DataFrame, comp: pd.DataFrame) -> Tuple[float, float]:
    # Testa se la distribuzione dei volumi per giorno-settimana differisce tra i due periodi
    def _dow_counts(df):
        d = df.groupby("weekday_num")["calls"].sum()
        # assicura 7 giorni
        return d.reindex(range(7), fill_value=0).values

    observed = np.vstack([_dow_counts(base), _dow_counts(comp)])
    chi2, p, dof, expected = chi2_contingency(observed)
    return chi2, p


def change_points_weekly(df: pd.DataFrame, pen: float = 3.0) -> list:
    if rpt is None or df.empty:
        return []
    y = df["weekly_calls"].values
    algo = rpt.Pelt(model="rbf").fit(y)
    # pen penalità: aumentare per meno breakpoint
    bkps = algo.predict(pen=pen)
    # convert indices to timestamps (end-exclusive indices)
    weeks = df["week"].tolist()
    cuts = [weeks[i - 1] for i in bkps if i - 1 < len(weeks) and i - 1 >= 0]
    return cuts


# ----------------------------------------
# Calcoli principali
# ----------------------------------------
base_weekly = weekly_totals(base_df)
comp_weekly = weekly_totals(comp_df)

# Change-points sul totale settimanale (baseline+confronto uniti in ordine cronologico)
all_weekly = weekly_totals(data[(data["ts"] >= cfg.baseline_start) & (data["ts"] <= cfg.compare_end)])
cp_weeks = change_points_weekly(all_weekly, pen=5.0)

# Day-of-week weights & varianza
base_w = weekday_weights_by_week(base_df)
comp_w = weekday_weights_by_week(comp_df)

# Statistiche su pesi (media/var/cv per weekday)
summary = (
    base_w.groupby("weekday_num")["weight"]
    .agg(["mean", "var", "std"])
    .rename(columns={"mean": "baseline_mean", "var": "baseline_var", "std": "baseline_std"})
)
summary_compare = (
    comp_w.groupby("weekday_num")["weight"]
    .agg(["mean", "var", "std"])
    .rename(columns={"mean": "compare_mean", "var": "compare_var", "std": "compare_std"})
)
summary = summary.join(summary_compare, how="outer")
summary["cv_baseline"] = summary["baseline_std"] / summary["baseline_mean"]
summary["cv_compare"] = summary["compare_std"] / summary["compare_mean"]

# Chi-square tra distribuzioni per DOW
chi2, pval = chi_square_dow(base_df, comp_df)

# Shape intra-day per weekday
base_shape = daily_shape(base_df)
comp_shape = daily_shape(comp_df)

# JS divergence per weekday tra periodi
js_by_weekday = []
for wd in range(7):
    pb = base_shape[base_shape["weekday_num"] == wd].sort_values("hour")["p"].values
    pc = comp_shape[comp_shape["weekday_num"] == wd].sort_values("hour")["p"].values
    if len(pb) == 0 or len(pc) == 0:
        js = np.nan
    else:
        # align lengths (hours 0..23)
        if len(pb) != 24:
            tmp = np.zeros(24)
            hb = base_shape[base_shape["weekday_num"] == wd]
            for h, p_ in zip(hb["hour"], hb["p"]):
                tmp[int(h)] = p_
            pb = tmp
        if len(pc) != 24:
            tmp = np.zeros(24)
            hc = comp_shape[comp_shape["weekday_num"] == wd]
            for h, p_ in zip(hc["hour"], hc["p"]):
                tmp[int(h)] = p_
            pc = tmp
        js = jensen_shannon(pb, pc)
    js_by_weekday.append({"weekday_num": wd, "JS_divergence": js})
js_df = pd.DataFrame(js_by_weekday)

# ----------------------------------------
# Visualizzazioni
# ----------------------------------------
st.subheader("📈 Totali settimanali e cambi di livello")
col1, col2 = st.columns([2, 1])
with col1:
    st.line_chart(all_weekly.set_index("week")["weekly_calls"])
with col2:
    if cp_weeks:
        st.caption("Breakpoints stimati (PELT/RBF):")
        for w in cp_weeks:
            st.write(f"• {pd.to_datetime(w).date()}")
    else:
        st.caption("Nessun breakpoint rilevato (o libreria 'ruptures' non disponibile)")

st.subheader("🧭 Pesi per giorno della settimana")
col3, col4 = st.columns(2)
with col3:
    st.markdown("**Baseline – media pesi**")
    tmp = base_w.groupby("weekday_num")["weight"].mean().reindex(range(7))
    tmp.index = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
    st.bar_chart(tmp)
with col4:
    st.markdown("**Confronto – media pesi**")
    tmp = comp_w.groupby("weekday_num")["weight"].mean().reindex(range(7))
    tmp.index = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
    st.bar_chart(tmp)

st.markdown("**Stabilità dei pesi (varianza/CV)**")
st.dataframe(summary.round(4))

st.subheader("🧪 Test statistico sulla distribuzione DOW")
st.write(
    f"Chi-square = {chi2:.2f}, p-value = {pval:.4f} — "
    + ("**Differenza significativa**" if pval < 0.05 else "nessuna differenza significativa (α=0.05)")
)

st.subheader("🕒 Divergenza intra-day per weekday (Jensen–Shannon)")
js_view = js_df.set_index("weekday_num").reindex(range(7))
js_view.index = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
st.bar_chart(js_view)

# ----------------------------------------
# Dimensionamento per slot (Erlang C)
# ----------------------------------------
st.header("👷 Dimensionamento per slot (Erlang C)")

interval_seconds = cfg.slot_minutes * 60


def erlang_c_probability_waiting(A: float, N: int) -> float:
    # A = traffic intensity (Erlang), N = agenti
    if N <= A:  # sistema instabile -> prob attesa = 1
        return 1.0
    # Calcolo P0 con ricorrenza stabile
    sum_terms = 0.0
    fact = 1.0
    for k in range(N):
        if k > 0:
            fact *= A / k
        else:
            fact = 1.0
        sum_terms += fact
    # termine in coda
    tail = (A ** N) / math.factorial(N) * (N / (N - A))
    P0 = 1.0 / (sum_terms + tail)
    Pw = tail * P0
    return min(max(Pw, 0.0), 1.0)


def service_level(A: float, N: int, t_sec: int, aht_sec: int) -> float:
    if N <= A:
        return 0.0
    Pw = erlang_c_probability_waiting(A, N)
    return 1.0 - Pw * math.exp(-(N - A) * (t_sec / aht_sec))


def required_agents_for_slot(call_count: float, aht_sec: int, t_sec: int, target_sl: float,
                              interval_sec: int, max_occ: Optional[float] = None) -> int:
    # traffico in Erlang: A = λ * AHT, con λ = calls / interval_sec
    A = (call_count * aht_sec) / interval_sec
    # lower bound: ceil(A) + 1 per stabilità
    N = max(1, math.ceil(A) + 1)
    # ricerca incrementale
    best = None
    for agents in range(N, N + 1000):  # guardrail
        sl = service_level(A, agents, t_sec, aht_sec)
        occ = A / agents if agents > 0 else 1.0
        if sl >= target_sl and (max_occ is None or occ <= max_occ):
            best = agents
            break
    return best or (N + 1000)

# Calcolo agenti richiesti per ogni slot nei periodi selezionati

def staffing_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["required_agents_raw"] = out["calls"].apply(
        lambda c: required_agents_for_slot(
            call_count=c,
            aht_sec=cfg.aht_sec,
            t_sec=cfg.target_t_sec,
            target_sl=cfg.target_sl,
            interval_sec=interval_seconds,
            max_occ=cfg.max_occupancy,
        )
    )
    # Applica shrinkage -> agenti pianificati
    out["agents_planned"] = np.ceil(out["required_agents_raw"] / (1.0 - cfg.shrinkage))
    return out

with st.spinner("Calcolo dimensionamento per slot (con shrinkage)…"):
    staff_df = staffing_table(comp_df)

c5, c6 = st.columns([2, 1])
with c5:
    st.markdown("**Agenti pianificati medi per ora (periodo di confronto)**")
    hourly = staff_df.groupby([staff_df["ts"].dt.date, staff_df["ts"].dt.hour])["agents_planned"].max()
    hourly = hourly.groupby(level=1).mean()
    hourly.index = [f"{int(h):02d}:00" for h in hourly.index]
    st.bar_chart(hourly)
with c6:
    st.metric("Agenti medi pianificati/slot", f"{staff_df['agents_planned'].mean():.1f}")
    st.metric("Agenti massimi su uno slot", int(staff_df["agents_planned"].max()))

# Downloadable outputs
st.subheader("⬇️ Esporta risultati")
col7, col8, col9 = st.columns(3)
with col7:
    csv_weekly = all_weekly.to_csv(index=False).encode("utf-8")
    st.download_button("Weekly totals (CSV)", data=csv_weekly, file_name="weekly_totals.csv", mime="text/csv")
with col8:
    basew = base_w.copy(); basew["period"] = "baseline"
    compw = comp_w.copy(); compw["period"] = "compare"
    csv_weights = pd.concat([basew, compw], ignore_index=True).to_csv(index=False).encode("utf-8")
    st.download_button("Weekday weights (CSV)", data=csv_weights, file_name="weekday_weights.csv", mime="text/csv")
with col9:
    st.download_button("Staffing per slot (CSV)", data=staff_df.to_csv(index=False).encode("utf-8"),
                      file_name="staffing_slots.csv", mime="text/csv")

# ----------------------------------------
# Note & Suggerimenti
# ----------------------------------------
st.markdown(
    """
### 🔍 Suggerimenti analitici
- **Trend & break strutturali**: osserva i *weekly totals* e i *breakpoints*. Se vedi un cambio netto post-lancio campagna/nuova IVR/nuovo canale, considera un nuovo baseline.
- **Stabilità dei pesi DOW**: media/varianza/CV dei pesi ti dicono se, a parità di 100=volumi settimanali, cambia il mix (es. lunedì più pesante). Il test χ² verifica se la distribuzione è cambiata significativamente.
- **Forma intra-day**: la divergenza Jensen–Shannon per weekday mette in evidenza spostamenti di picco (es. più tardi la sera). Valori > 0.1 spesso indicano cambi non banali.
- **Parametri Erlang**: verifica **AHT** e **shrinkage** aggiornati. Se AHT è aumentato, servono più agenti anche a volumi invariati.
- **Occupancy cap**: imposta un tetto all'occupancy per evitare over-stress: l'app rispetta `max_occupancy`.
- **DST**: i dati sono localizzati in `{}`; verifica i giorni di cambio ora.
    """.format(cfg.tz)
)
