import os
import re
import time
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
import numpy as np

# load .env if present
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

import streamlit as st
import plotly.express as px

try:
    from streamlit_plotly_events import plotly_events  # optional
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False


DEFAULT_BASE_URL = "https://api.sigecloud.com.br"
DEFAULT_ENDPOINT = "/request/Pedidos/GetTodosPedidos"
DEFAULT_RPS = 2.0

CACHE_DIR = ".cache_sige"
ORDERS_PARQUET = os.path.join(CACHE_DIR, "orders.parquet")
ITEMS_PARQUET = os.path.join(CACHE_DIR, "items.parquet")
META_JSON = os.path.join(CACHE_DIR, "meta.json")


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def digits_only(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(r"\D+", "", str(s))


def doc_type_from_cliente_cnpj(value: Any) -> str:
    d = digits_only(value)
    if len(d) == 11:
        return "CPF"
    if len(d) == 14:
        return "CNPJ"
    if len(d) == 0:
        return "DESCONHECIDO"
    return "DESCONHECIDO"


def parse_dt(x: Any) -> pd.Timestamp:
    if x is None:
        return pd.NaT
    s = str(x)
    if s.startswith("0001-01-01"):
        return pd.NaT
    return pd.to_datetime(s, utc=True, errors="coerce")


def normalize_text_title(s: Any) -> str:
    """Basic normalization for bairro/cidade/vendedor strings."""
    if s is None:
        return "N/I"
    t = str(s).strip()
    if not t:
        return "N/I"
    t = re.sub(r"\s+", " ", t)
    return t.title()


def money_br(x: float) -> str:
    try:
        return f"R$ {float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "R$ 0,00"


def request_with_backoff(
    session: requests.Session,
    url: str,
    headers: Dict[str, str],
    params: Dict[str, Any],
    timeout: int = 60,
    max_retries: int = 6,
    base_sleep: float = 0.75,
) -> requests.Response:
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = session.get(url, headers=headers, params=params, timeout=timeout)

            if resp.status_code in (429, 500, 502, 503, 504):
                sleep_s = base_sleep * (2 ** attempt) + float(np.random.random()) * 0.2
                time.sleep(sleep_s)
                continue

            resp.raise_for_status()
            return resp

        except Exception as e:
            last_exc = e
            sleep_s = base_sleep * (2 ** attempt) + float(np.random.random()) * 0.2
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed after retries: {last_exc}")


def fetch_all_orders_paginated(
    base_url: str,
    endpoint: str,
    headers: Dict[str, str],
    rps: float = DEFAULT_RPS,
    start_page: int = 1,
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    url = base_url.rstrip("/") + endpoint
    sleep_between = 1.0 / max(rps, 0.1)

    all_rows: List[Dict[str, Any]] = []
    session = requests.Session()
    page = start_page

    while True:
        if max_pages is not None and (page - start_page + 1) > max_pages:
            break

        t0 = time.time()
        resp = request_with_backoff(session, url, headers, params={"page": page})
        data = resp.json()

        if not isinstance(data, list):
            raise ValueError(f"Resposta inesperada (não é lista). Página {page}: {type(data)}")

        if len(data) == 0:
            break

        all_rows.extend(data)

        elapsed = time.time() - t0
        if elapsed < sleep_between:
            time.sleep(sleep_between - elapsed)

        page += 1

    return all_rows


def normalize_orders(raw_orders: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not raw_orders:
        return pd.DataFrame(), pd.DataFrame()

    df_orders = pd.json_normalize(raw_orders, sep=".")

    # Ensure columns exist
    for col in [
        "ID",
        "ClienteID",
        "Cliente",
        "ClienteCNPJ",
        "StatusSistema",
        "ValorFinal",
        "Municipio",
        "Bairro",
        "DataFaturamento",
        "Vendedor",
    ]:
        if col not in df_orders.columns:
            df_orders[col] = None

    df_orders["DataFaturamento_dt"] = df_orders["DataFaturamento"].apply(parse_dt)

    # AnoMes: seguro com tz
    df_orders["AnoMes"] = df_orders["DataFaturamento_dt"].dt.to_period("M").dt.to_timestamp()
    df_orders["AnoMes"] = pd.to_datetime(df_orders["AnoMes"], utc=True, errors="coerce")

    df_orders["DocTipo"] = df_orders["ClienteCNPJ"].apply(doc_type_from_cliente_cnpj)

    # Normalize address
    df_orders["Municipio_norm"] = df_orders["Municipio"].apply(normalize_text_title)
    df_orders["Bairro_norm"] = df_orders["Bairro"].apply(normalize_text_title)

    # Normalize seller
    df_orders["Vendedor_norm"] = df_orders["Vendedor"].apply(normalize_text_title)

    # Items
    items_rows = []
    for o in raw_orders:
        order_id = o.get("ID")
        items = o.get("Items") or []
        for it in items:
            row = dict(it)
            row["_OrderID"] = order_id
            items_rows.append(row)

    df_items = pd.DataFrame(items_rows)
    return df_orders, df_items


def compute_first_purchase_flags(df_orders: pd.DataFrame) -> pd.DataFrame:
    df = df_orders.copy()
    df = df[~df["ClienteID"].isna()].copy()
    df["DataFaturamento_dt"] = pd.to_datetime(df["DataFaturamento_dt"], utc=True, errors="coerce")

    first_dt = (
        df.groupby("ClienteID", as_index=False)["DataFaturamento_dt"]
        .min()
        .rename(columns={"DataFaturamento_dt": "PrimeiraCompra_dt"})
    )

    df_orders = df_orders.merge(first_dt, on="ClienteID", how="left")

    df_orders["SegmentoCompra"] = np.where(
        (df_orders["PrimeiraCompra_dt"].notna())
        & (df_orders["DataFaturamento_dt"] == df_orders["PrimeiraCompra_dt"]),
        "Primeira compra",
        "Retenção",
    )
    df_orders.loc[
        df_orders["ClienteID"].isna() | df_orders["DataFaturamento_dt"].isna(),
        "SegmentoCompra",
    ] = "DESCONHECIDO"

    return df_orders


def load_cache() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any]]:
    ensure_cache_dir()
    meta: Dict[str, Any] = {}

    if os.path.exists(META_JSON):
        try:
            with open(META_JSON, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    df_orders = pd.read_parquet(ORDERS_PARQUET) if os.path.exists(ORDERS_PARQUET) else None
    df_items = pd.read_parquet(ITEMS_PARQUET) if os.path.exists(ITEMS_PARQUET) else None

    return df_orders, df_items, meta


def save_cache(df_orders: pd.DataFrame, df_items: pd.DataFrame, meta: Dict[str, Any]):
    ensure_cache_dir()
    df_orders.to_parquet(ORDERS_PARQUET, index=False)
    (df_items if df_items is not None else pd.DataFrame()).to_parquet(ITEMS_PARQUET, index=False)
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def run_update(
    base_url: str,
    endpoint: str,
    auth_token: str,
    user_header: str,
    app_header: str,
    rps: float,
    max_pages: Optional[int] = None,
):
    if not auth_token or not user_header:
        raise SystemExit("Faltou SIGE_AUTH_TOKEN ou SIGE_USER no ambiente/.env")

    headers = {
        "Accept": "application/json",
        "Authorization-Token": auth_token,
        "User": user_header,
        "App": app_header,
    }

    print("Coletando dados da API (paginando até vir vazio)...")
    raw = fetch_all_orders_paginated(
        base_url=base_url,
        endpoint=endpoint,
        headers=headers,
        rps=rps,
        start_page=1,
        max_pages=max_pages,
    )
    print(f"Total bruto recebido: {len(raw)}")

    df_orders, df_items = normalize_orders(raw)

    # keep only Pedido Faturado
    df_orders = df_orders[df_orders["StatusSistema"] == "Pedido Faturado"].copy()

    # compute flags
    df_orders = compute_first_purchase_flags(df_orders)

    meta = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_count": len(raw),
        "orders_count_after_filter": int(len(df_orders)),
    }

    save_cache(df_orders, df_items, meta)
    print(f"Cache atualizado: {ORDERS_PARQUET} (pedidos faturados: {len(df_orders)})")


def parse_args():
    p = argparse.ArgumentParser(add_help=True)
    sub = p.add_subparsers(dest="cmd")

    up = sub.add_parser("update", help="Baixa da API e grava cache local (parquet).")
    up.add_argument("--base-url", default=DEFAULT_BASE_URL)
    up.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    up.add_argument("--rps", type=float, default=DEFAULT_RPS)
    up.add_argument("--max-pages", type=int, default=None)

    return p.parse_args()


def running_in_streamlit() -> bool:
    return os.environ.get("STREAMLIT_SERVER_RUNNING") == "1"


def main_cli():
    args = parse_args()
    if args.cmd == "update":
        run_update(
            base_url=args.base_url,
            endpoint=args.endpoint,
            auth_token=os.getenv("SIGE_AUTH_TOKEN", ""),
            user_header=os.getenv("SIGE_USER", ""),
            app_header=os.getenv("SIGE_APP", "API"),
            rps=args.rps,
            max_pages=args.max_pages,
        )
        raise SystemExit(0)


if __name__ == "__main__":
    # via "python app.py ..." => CLI
    # via "streamlit run app.py" => não encerra; deixa seguir para UI
    if not running_in_streamlit():
        main_cli()


# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Pedidos Faturados - Dashboard", layout="wide")
st.title("Dashboard - Pedidos Faturados (SIGE Cloud)")

df_orders, df_items, meta = load_cache()

if df_orders is None or len(df_orders) == 0:
    st.error("Cache não encontrado ou vazio. Rode primeiro: `python app.py update`")
    st.stop()

# Ensure types
df_orders["DataFaturamento_dt"] = pd.to_datetime(df_orders["DataFaturamento_dt"], utc=True, errors="coerce")
df_orders["AnoMes"] = pd.to_datetime(df_orders["AnoMes"], utc=True, errors="coerce")
df_orders["ValorFinal"] = pd.to_numeric(df_orders["ValorFinal"], errors="coerce").fillna(0.0)

st.caption(f"Última atualização (UTC): {meta.get('fetched_at_utc', 'N/I')}")

# Sidebar filters
with st.sidebar:
    st.header("Filtros")

    min_dt = df_orders["DataFaturamento_dt"].min()
    max_dt = df_orders["DataFaturamento_dt"].max()
    if pd.isna(min_dt) or pd.isna(max_dt):
        st.error("Datas de faturamento não disponíveis/parseadas.")
        st.stop()

    date_range = st.date_input(
        "Período (DataFaturamento)",
        value=(min_dt.date(), max_dt.date()),
        min_value=min_dt.date(),
        max_value=max_dt.date(),
    )
    d0, d1 = date_range if isinstance(date_range, tuple) else (min_dt.date(), max_dt.date())
    start_dt = pd.Timestamp(d0).tz_localize("UTC")
    end_dt = pd.Timestamp(d1).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    vendedores = sorted(df_orders["Vendedor_norm"].fillna("N/I").unique().tolist())
    selected_vendedores = st.multiselect("Vendedores", options=vendedores, default=vendedores)

    st.subheader("Segmentações")
    selected_doc_types = st.multiselect(
        "Tipo de documento", options=["CPF", "CNPJ", "DESCONHECIDO"], default=["CPF", "CNPJ"]
    )
    selected_seg_compra = st.multiselect(
        "Tipo de compra",
        options=["Primeira compra", "Retenção", "DESCONHECIDO"],
        default=["Primeira compra", "Retenção"],
    )

    st.subheader("Top clientes")
    top_metric = st.radio("Ordenar por", options=["Valor (R$)", "Nº pedidos"], horizontal=True)
    top_n = st.slider("Top N", min_value=5, max_value=50, value=15, step=5)

# Apply filters
df_f = df_orders.copy()
df_f = df_f[df_f["DataFaturamento_dt"].between(start_dt, end_dt, inclusive="both")]
df_f = df_f[df_f["Vendedor_norm"].isin(selected_vendedores)]
df_f = df_f[df_f["DocTipo"].isin(selected_doc_types)]
df_f = df_f[df_f["SegmentoCompra"].isin(selected_seg_compra)]

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pedidos (filtro)", f"{len(df_f):,}".replace(",", "."))
c2.metric("Faturamento (filtro)", money_br(df_f["ValorFinal"].sum()))
c3.metric("Clientes únicos", f"{df_f['ClienteID'].nunique():,}".replace(",", "."))
c4.metric("Ticket médio", money_br(df_f["ValorFinal"].mean() if len(df_f) else 0.0))

st.divider()

# Monthly revenue stacked CPF/CNPJ
st.subheader("Faturamento por mês (CPF x CNPJ empilhado)")

monthly = df_f.groupby(["AnoMes", "DocTipo"], as_index=False).agg(Faturamento=("ValorFinal", "sum"))

if len(monthly) == 0:
    st.info("Sem dados no período/filtros.")
else:
    fig_m = px.bar(
        monthly,
        x="AnoMes",
        y="Faturamento",
        color="DocTipo",
        barmode="stack",
        category_orders={"DocTipo": ["CPF", "CNPJ", "DESCONHECIDO"]},
        hover_data={"AnoMes": "|%Y-%m", "Faturamento": ":,.2f", "DocTipo": True},
    )
    fig_m.update_layout(xaxis_title="Mês", yaxis_title="Faturamento (R$)", yaxis_tickprefix="R$ ")
    fig_m.update_yaxes(tickformat=",.2f")
    st.plotly_chart(fig_m, use_container_width=True)

st.divider()

# Top clients
st.subheader("Top clientes no período")

if len(df_f) == 0:
    st.info("Sem dados no período/filtros.")
else:
    by_client = (
        df_f.groupby(["Cliente"], as_index=False)
        .agg(Faturamento=("ValorFinal", "sum"), Pedidos=("ID", "count"))
    )
    by_client["TicketMedio"] = by_client["Faturamento"] / by_client["Pedidos"].replace(0, np.nan)
    by_client["TicketMedio"] = by_client["TicketMedio"].fillna(0.0)

    sort_col = "Faturamento" if top_metric == "Valor (R$)" else "Pedidos"
    by_client = by_client.sort_values(sort_col, ascending=False).head(top_n).reset_index(drop=True)
    by_client.insert(0, "Ranking", by_client.index + 1)

    out = by_client.copy()
    out["Faturamento"] = out["Faturamento"].map(money_br)
    out["TicketMedio"] = out["TicketMedio"].map(money_br)

    st.dataframe(out[["Ranking", "Cliente", "Faturamento", "TicketMedio", "Pedidos"]],
                 use_container_width=True, hide_index=True)

st.divider()

# Bairros analysis
st.subheader("Bairros (normalizados): compradores, faturamento e ticket mediano")

buyers_bairro = (
    df_f.groupby(["Municipio_norm", "Bairro_norm"], as_index=False)
    .agg(
        Compradores=("ClienteID", pd.Series.nunique),
        Faturamento=("ValorFinal", "sum"),
        TicketMediano=("ValorFinal", "median"),
    )
)

colA, colB = st.columns(2)

with colA:
    st.write("**Faturamento por bairro (Top 20)**")
    top_rev = buyers_bairro.sort_values("Faturamento", ascending=False).head(20)
    fig_b1 = px.bar(
        top_rev, x="Faturamento", y="Bairro_norm", color="Municipio_norm",
        orientation="h", hover_data={"Faturamento": ":,.2f"},
    )
    fig_b1.update_layout(xaxis_title="Faturamento (R$)", yaxis_title="Bairro")
    fig_b1.update_xaxes(tickprefix="R$ ", tickformat=",.2f")
    st.plotly_chart(fig_b1, use_container_width=True)

with colB:
    st.write("**Compradores por bairro (Top 20)**")
    top_buy = buyers_bairro.sort_values("Compradores", ascending=False).head(20)
    fig_b2 = px.bar(top_buy, x="Compradores", y="Bairro_norm", color="Municipio_norm", orientation="h")
    fig_b2.update_layout(xaxis_title="Compradores únicos", yaxis_title="Bairro")
    st.plotly_chart(fig_b2, use_container_width=True)

st.write("**Ticket mediano (mediana do valor do pedido) por bairro (Top 20)**")
top_med = buyers_bairro.sort_values("TicketMediano", ascending=False).head(20)
fig_b3 = px.bar(
    top_med, x="TicketMediano", y="Bairro_norm", color="Municipio_norm",
    orientation="h", hover_data={"TicketMediano": ":,.2f"},
)
fig_b3.update_layout(xaxis_title="Ticket mediano (R$)", yaxis_title="Bairro")
fig_b3.update_xaxes(tickprefix="R$ ", tickformat=",.2f")
st.plotly_chart(fig_b3, use_container_width=True)

st.divider()

# Cities + drilldown
st.subheader("Cidades e drill-down para bairros")

by_city = (
    df_f.groupby("Municipio_norm", as_index=False)
    .agg(
        Faturamento=("ValorFinal", "sum"),
        Pedidos=("ID", "count"),
        Compradores=("ClienteID", pd.Series.nunique),
    )
    .sort_values("Faturamento", ascending=False)
)

left_city, right_city = st.columns([1.1, 1.0])
selected_city = None

with left_city:
    fig_city = px.bar(by_city.head(30), x="Municipio_norm", y="Faturamento")
    fig_city.update_layout(xaxis_title="Cidade", yaxis_title="Faturamento (R$)")
    fig_city.update_xaxes(tickangle=45)
    fig_city.update_yaxes(tickprefix="R$ ", tickformat=",.2f")

    if HAS_PLOTLY_EVENTS:
        st.caption("Clique em uma barra para selecionar a cidade.")
        clicked = plotly_events(
            fig_city,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=450,
        )
        if clicked:
            selected_city = clicked[0].get("x")
    else:
        st.plotly_chart(fig_city, use_container_width=True)
        selected_city = st.selectbox(
            "Selecionar cidade (fallback)",
            options=["(todas)"] + by_city["Municipio_norm"].tolist(),
        )
        if selected_city == "(todas)":
            selected_city = None

with right_city:
    if selected_city is None:
        df_nb = buyers_bairro.sort_values("Faturamento", ascending=False).head(25)
        title = "Top bairros por faturamento (todas as cidades)"
    else:
        df_nb = buyers_bairro[buyers_bairro["Municipio_norm"] == selected_city] \
            .sort_values("Faturamento", ascending=False).head(25)
        title = f"Top bairros por faturamento - {selected_city}"

    fig_nb = px.bar(df_nb, x="Faturamento", y="Bairro_norm", orientation="h", title=title)
    fig_nb.update_xaxes(tickprefix="R$ ", tickformat=",.2f")
    st.plotly_chart(fig_nb, use_container_width=True)

with st.expander("Debug / Cache meta"):
    st.json(meta)