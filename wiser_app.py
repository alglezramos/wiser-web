"""Wiser - Full Pallet Picking (archivo único: núcleo + app Streamlit).
Ejecutar:  streamlit run wiser_app.py

Warehouse slotting.

Metodología (según el resumen ejecutivo de la tesis):
  1. Clasificación ABC por volumen de salidas acumulado (umbrales 80% / 95%).
  2. Zonificación: la clase A recibe los mejores slots, luego B, luego C.
  3. Dentro de cada clase, asignación óptima (LAP) con el algoritmo húngaro
     (scipy.optimize.linear_sum_assignment).
  4. Pasada de consolidación: agrupa los pallets de un mismo SKU sin superar
     una tolerancia de sobrecosto.

Modelo de costo (por pallet-instancia i en slot j):
    costo[i, j] = in_i * T_in[j] + out_i * T_out[j]
    T_punto[j]  = round_trip * distancia_rectilinea(slot, punto) / velocidad
                  + (vertical_fijo + vertical_por_nivel * nivel)   si nivel > 0
El tiempo de manipulación (handling) es constante por movimiento y no cambia
la asignación; se reporta aparte.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import streamlit as st

SKU_COLS = ["sku", "pallets", "in_moves", "out_moves"]
SLOT_COLS = ["slot_id", "x", "y", "level"]


@dataclass
class Params:
    # Seis parámetros físicos / económicos del modelo
    forklift_speed: float = 2.0       # m/s
    vertical_fixed: float = 8.0       # s por movimiento vertical (término fijo)
    vertical_per_level: float = 4.0   # s por nivel
    handling_time: float = 30.0       # s por ciclo (constante)
    round_trip: float = 2.0           # ida y vuelta
    labor_cost_h: float = 20.0        # EUR por hora de operario
    # Geometría: punto de entrada (producción) y de salida (expedición)
    in_x: float = 0.0
    in_y: float = 0.0
    out_x: float = 40.0
    out_y: float = 0.0
    # Clasificación ABC y consolidación
    abc_a: float = 0.80
    abc_b: float = 0.95
    consolidation_tol: float = 0.005  # sobrecosto máximo permitido (fracción)


# --------------------------------------------------------------------------- #
# Validación y preparación
# --------------------------------------------------------------------------- #
def validate_inputs(skus: pd.DataFrame, slots: pd.DataFrame):
    for df, cols, name in ((skus, SKU_COLS, "SKUs"), (slots, SLOT_COLS, "slots")):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Tabla {name}: faltan columnas {missing}")
    skus = skus[SKU_COLS].copy().reset_index(drop=True)
    slots = slots[SLOT_COLS].copy().reset_index(drop=True)
    if skus["sku"].duplicated().any():
        raise ValueError("SKUs duplicados")
    if slots["slot_id"].duplicated().any():
        raise ValueError("slot_id duplicados")
    for c in ("pallets", "in_moves", "out_moves"):
        skus[c] = pd.to_numeric(skus[c], errors="raise")
    for c in ("x", "y", "level"):
        slots[c] = pd.to_numeric(slots[c], errors="raise")
    if (skus[["pallets", "in_moves", "out_moves"]] < 0).any().any():
        raise ValueError("Valores negativos en la tabla de SKUs")
    if (skus["pallets"] < 1).any() or (skus["pallets"] % 1 != 0).any():
        raise ValueError("'pallets' debe ser un entero >= 1")
    skus["pallets"] = skus["pallets"].astype(int)
    if (slots["level"] < 0).any():
        raise ValueError("'level' no puede ser negativo (0 = nivel de suelo)")
    n_inst = int(skus["pallets"].sum())
    if n_inst > len(slots):
        raise ValueError(
            f"Capacidad insuficiente: {n_inst} pallets para {len(slots)} slots"
        )
    return skus, slots


def abc_classify(skus: pd.DataFrame, a: float, b: float) -> pd.Series:
    """Clase por volumen de salidas acumulado. Un SKU es A mientras el
    acumulado ANTES de incluirlo sea < a (el que cruza el umbral entra)."""
    vol = skus["out_moves"].astype(float)
    if vol.sum() <= 0:
        vol = skus["in_moves"].astype(float) + skus["out_moves"].astype(float)
    order = vol.sort_values(ascending=False, kind="stable").index
    total = vol.sum()
    if total <= 0:
        return pd.Series("C", index=skus.index)
    prev = vol.loc[order].cumsum() / total - vol.loc[order] / total
    cls = pd.Series("C", index=skus.index)
    cls.loc[order[prev.to_numpy() < b]] = "B"
    cls.loc[order[prev.to_numpy() < a]] = "A"
    return cls


def expand_instances(skus: pd.DataFrame) -> pd.DataFrame:
    inst = skus.loc[skus.index.repeat(skus["pallets"])].copy()
    inst["instance"] = inst.groupby("sku").cumcount() + 1
    inst["in_i"] = inst["in_moves"] / inst["pallets"]
    inst["out_i"] = inst["out_moves"] / inst["pallets"]
    return inst.reset_index(drop=True)


def slot_times(slots: pd.DataFrame, p: Params):
    lvl = slots["level"].to_numpy(float)
    vert = np.where(lvl > 0, p.vertical_fixed + p.vertical_per_level * lvl, 0.0)

    def t(px, py):
        d = np.abs(slots["x"].to_numpy(float) - px) + np.abs(slots["y"].to_numpy(float) - py)
        return p.round_trip * d / p.forklift_speed + vert

    return t(p.in_x, p.in_y), t(p.out_x, p.out_y)


# --------------------------------------------------------------------------- #
# Asignación
# --------------------------------------------------------------------------- #
def _reference_time(inst, t_in, t_out):
    tin, tout = inst["in_i"].sum(), inst["out_i"].sum()
    tot = tin + tout
    if tot <= 0:
        return (t_in + t_out) / 2
    return (tin * t_in + tout * t_out) / tot


def hierarchical_assign(inst, cost, t_ref):
    """ABC zoning + LAP (húngaro) dentro de cada clase."""
    order = np.argsort(t_ref, kind="stable")
    n = {c: int((inst["cls"] == c).sum()) for c in "ABC"}
    zones = {
        "A": order[: n["A"]],
        "B": order[n["A"]: n["A"] + n["B"]],
        "C": order[n["A"] + n["B"]:],
    }
    assign = np.full(len(inst), -1, dtype=int)
    for c in "ABC":
        rows = np.where(inst["cls"].to_numpy() == c)[0]
        if len(rows) == 0:
            continue
        sub = cost[np.ix_(rows, zones[c])]
        r, cc = linear_sum_assignment(sub)
        assign[rows[r]] = zones[c][cc]
    return assign


def global_lap(cost):
    """LAP sin jerarquía (cota inferior de costo con este modelo)."""
    r, c = linear_sum_assignment(cost)
    out = np.empty(cost.shape[0], dtype=int)
    out[r] = c
    return out


def coi_assign(skus, inst, t_ref):
    """Benchmark Cube-per-Order Index (Heskett, 1963)."""
    moves = (skus["in_moves"] + skus["out_moves"]).astype(float)
    coi = skus["pallets"] / moves.replace(0, np.nan)
    coi = coi.fillna(np.inf)
    rank = dict(zip(skus.loc[np.argsort(coi.to_numpy(), kind="stable"), "sku"], range(len(skus))))
    inst_order = np.argsort(inst["sku"].map(rank).to_numpy(), kind="stable")
    slot_order = np.argsort(t_ref, kind="stable")
    assign = np.empty(len(inst), dtype=int)
    assign[inst_order] = slot_order[: len(inst)]
    return assign


def consolidate(assign, inst, slots, cost, tol):
    """Acerca los pallets de un mismo SKU intercambiando slots con pallets de
    otros SKUs de la MISMA clase, mientras el sobrecosto acumulado <= tol."""
    assign = assign.copy()
    idx = np.arange(len(assign))
    budget = tol * cost[idx, assign].sum()
    X, Y = slots["x"].to_numpy(float), slots["y"].to_numpy(float)
    cls, sku = inst["cls"].to_numpy(), inst["sku"].to_numpy()
    swaps = 0
    for s_id, grp in inst.groupby("sku").indices.items():
        if len(grp) < 2:
            continue
        grp = sorted(grp, key=lambda i: cost[i, assign[i]])
        anchor = assign[grp[0]]
        for k in grp[1:]:
            sk = assign[k]
            dk = abs(X[sk] - X[anchor]) + abs(Y[sk] - Y[anchor])
            if dk == 0:
                continue
            d_all = np.abs(X[assign] - X[anchor]) + np.abs(Y[assign] - Y[anchor])
            mask = (cls == cls[k]) & (sku != s_id) & (d_all < dk)
            cands = np.where(mask)[0]
            cands = cands[np.argsort(d_all[cands], kind="stable")][:30]
            for o in cands:
                so = assign[o]
                delta = cost[k, so] + cost[o, sk] - cost[k, sk] - cost[o, so]
                if delta <= budget:
                    assign[k], assign[o] = so, sk
                    budget -= max(delta, 0.0)
                    swaps += 1
                    break
    return assign, swaps


def scatter_metric(assign, inst, slots):
    """Dispersión media de los SKUs multi-pallet (rango en x + rango en y, m)."""
    X, Y = slots["x"].to_numpy(float), slots["y"].to_numpy(float)
    vals = []
    for _, g in inst.groupby("sku").indices.items():
        if len(g) < 2:
            continue
        xs, ys = X[assign[g]], Y[assign[g]]
        vals.append((xs.max() - xs.min()) + (ys.max() - ys.min()))
    return float(np.mean(vals)) if vals else 0.0


# --------------------------------------------------------------------------- #
# API principal
# --------------------------------------------------------------------------- #
def optimize(skus: pd.DataFrame, slots: pd.DataFrame, p: Params | None = None) -> dict:
    p = p or Params()
    skus, slots = validate_inputs(skus, slots)
    skus["cls"] = abc_classify(skus, p.abc_a, p.abc_b)
    inst = expand_instances(skus).merge(skus[["sku", "cls"]], on="sku", suffixes=("", "_"))
    inst["cls"] = inst["cls_"]
    inst = inst.drop(columns="cls_")

    t_in, t_out = slot_times(slots, p)
    cost = inst["in_i"].to_numpy()[:, None] * t_in[None, :] + inst["out_i"].to_numpy()[:, None] * t_out[None, :]
    t_ref = _reference_time(inst, t_in, t_out)
    idx = np.arange(len(inst))

    a_h = hierarchical_assign(inst, cost, t_ref)
    a_final, swaps = consolidate(a_h, inst, slots, cost, p.consolidation_tol)
    a_glob = global_lap(cost)
    a_coi = coi_assign(skus, inst, t_ref)

    seconds = {
        "optimizado": cost[idx, a_final].sum(),
        "optimizado_sin_consolidar": cost[idx, a_h].sum(),
        "lap_global": cost[idx, a_glob].sum(),
        "coi": cost[idx, a_coi].sum(),
        "aleatorio": cost.mean(axis=1).sum(),  # esperanza exacta de almacenamiento aleatorio
    }
    hours = {k: v / 3600 for k, v in seconds.items()}

    def saving(base):
        return 0.0 if hours[base] == 0 else 100 * (hours[base] - hours["optimizado"]) / hours[base]

    moves_total = float((skus["in_moves"] + skus["out_moves"]).sum())
    kpis = {
        "horas_viaje": hours,
        "ahorro_vs_aleatorio_pct": saving("aleatorio"),
        "ahorro_vs_coi_pct": saving("coi"),
        "ventaja_sobre_coi_pts": saving("aleatorio") - 100 * (hours["aleatorio"] - hours["coi"]) / hours["aleatorio"]
        if hours["aleatorio"] else 0.0,
        "ahorro_eur_vs_aleatorio": (hours["aleatorio"] - hours["optimizado"]) * p.labor_cost_h,
        "ahorro_eur_vs_coi": (hours["coi"] - hours["optimizado"]) * p.labor_cost_h,
        "horas_handling": p.handling_time * moves_total / 3600,
        "swaps_consolidacion": swaps,
        "dispersion_antes_m": scatter_metric(a_h, inst, slots),
        "dispersion_despues_m": scatter_metric(a_final, inst, slots),
        "brecha_jerarquia_vs_lap_global_pct": 100 * (seconds["optimizado_sin_consolidar"] - seconds["lap_global"]) / seconds["lap_global"]
        if seconds["lap_global"] else 0.0,
        "pallets": len(inst),
        "slots": len(slots),
        "ocupacion_pct": 100 * len(inst) / len(slots),
    }

    sl = slots.iloc[a_final].reset_index(drop=True)
    result = pd.DataFrame({
        "sku": inst["sku"], "instance": inst["instance"], "clase": inst["cls"],
        "slot_id": sl["slot_id"], "x": sl["x"], "y": sl["y"], "level": sl["level"],
        "tiempo_viaje_anual_s": cost[idx, a_final],
    }).sort_values(["clase", "sku", "instance"]).reset_index(drop=True)

    return {"assignment": result, "kpis": kpis, "skus": skus, "params": p,
            "_inst": inst, "_cost": cost, "_t_ref": t_ref, "_slots": slots,
            "_assign": {"hier": a_h, "final": a_final, "glob": a_glob, "coi": a_coi}}


def class_stability(res_a: dict, res_b: dict) -> float:
    """% de SKUs comunes que conservan su clase ABC entre dos ventanas."""
    a = res_a["skus"].set_index("sku")["cls"]
    b = res_b["skus"].set_index("sku")["cls"]
    common = a.index.intersection(b.index)
    return 100 * float((a[common] == b[common]).mean()) if len(common) else float("nan")


# --------------------------------------------------------------------------- #
# Datos de ejemplo
# --------------------------------------------------------------------------- #
def generate_sample(n_skus: int = 278, seed: int = 42, aisles: int = 10,
                    positions: int = 25, levels: int = 4):
    rng = np.random.default_rng(seed)
    out = np.sort(rng.lognormal(mean=6.0, sigma=1.2, size=n_skus))[::-1].round().clip(min=1)
    ratio = rng.uniform(0.2, 1.8, n_skus)  # mezcla de SKUs "de entrada" y "de salida"
    skus = pd.DataFrame({
        "sku": [f"SKU{i + 1:04d}" for i in range(n_skus)],
        "pallets": rng.choice([1, 2, 3, 4, 6], size=n_skus, p=[.35, .25, .2, .12, .08]),
        "out_moves": out.astype(int),
        "in_moves": (out * ratio).round().astype(int),
    })
    rows = []
    for a in range(aisles):
        for pos in range(positions):
            for lv in range(levels):
                rows.append((f"A{a + 1:02d}-{pos + 1:02d}-L{lv}", 4.0 + a * 3.5, 3.0 + pos * 1.2, lv))
    slots = pd.DataFrame(rows, columns=["slot_id", "x", "y", "level"])
    return skus, slots


# --------------------------------------------------------------------------- #
# App Streamlit
# --------------------------------------------------------------------------- #
def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def tool_full_pallet_picking():
    st.header("Full Pallet Picking · Slotting")
    st.caption("ABC por volumen de salidas → asignación óptima (húngaro) por clase → consolidación.")

    src = st.radio("Datos", ["Datos de ejemplo", "Subir CSV"], horizontal=True)
    if src == "Datos de ejemplo":
        n = st.slider("Número de SKUs", 20, 400, 278)
        skus, slots = generate_sample(n_skus=n)
        st.download_button("Descargar SKUs de ejemplo (plantilla)", _csv(skus), "skus.csv")
        st.download_button("Descargar slots de ejemplo (plantilla)", _csv(slots), "slots.csv")
    else:
        st.write("**skus.csv**: sku, pallets, in_moves, out_moves · **slots.csv**: slot_id, x, y, level")
        f1 = st.file_uploader("skus.csv", type="csv")
        f2 = st.file_uploader("slots.csv", type="csv")
        if not (f1 and f2):
            st.info("Sube ambos archivos para continuar.")
            return
        skus, slots = pd.read_csv(f1), pd.read_csv(f2)

    with st.expander("Parámetros del modelo", expanded=False):
        c1, c2, c3 = st.columns(3)
        p = Params(
            forklift_speed=c1.number_input("Velocidad montacargas (m/s)", 0.1, 10.0, 2.0),
            vertical_fixed=c1.number_input("Vertical fijo (s)", 0.0, 120.0, 8.0),
            vertical_per_level=c1.number_input("Vertical por nivel (s)", 0.0, 60.0, 4.0),
            handling_time=c2.number_input("Handling por ciclo (s)", 0.0, 300.0, 30.0),
            round_trip=c2.number_input("Factor ida y vuelta", 1.0, 2.0, 2.0),
            labor_cost_h=c2.number_input("Costo laboral (EUR/h)", 0.0, 200.0, 20.0),
            in_x=c3.number_input("Entrada x (m)", value=0.0), in_y=c3.number_input("Entrada y (m)", value=0.0),
            out_x=c3.number_input("Salida x (m)", value=40.0), out_y=c3.number_input("Salida y (m)", value=0.0),
            abc_a=st.slider("Umbral A", 0.5, 0.95, 0.80),
            abc_b=st.slider("Umbral B", 0.80, 0.99, 0.95),
            consolidation_tol=st.slider("Tolerancia de consolidación", 0.0, 0.05, 0.005, 0.001, format="%.3f"),
        )

    if st.button("Optimizar", type="primary"):
        try:
            st.session_state["res_fpp"] = optimize(skus, slots, p)
        except ValueError as e:
            st.session_state.pop("res_fpp", None)
            st.error(str(e))
            return

    res = st.session_state.get("res_fpp")
    if res is None:
        return
    k = res["kpis"]
    h = k["horas_viaje"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ahorro vs. aleatorio", f"{k['ahorro_vs_aleatorio_pct']:.1f}%")
    m2.metric("Ahorro vs. COI", f"{k['ahorro_vs_coi_pct']:.1f}%")
    m3.metric("Ahorro anual vs. COI", f"€{k['ahorro_eur_vs_coi']:,.0f}")
    m4.metric("Ocupación", f"{k['ocupacion_pct']:.0f}%")
    st.bar_chart(pd.Series({"Aleatorio": h["aleatorio"], "COI": h["coi"],
                            "Optimizado": h["optimizado"], "LAP global": h["lap_global"]},
                           name="Horas de viaje / año"))
    st.caption(f"Consolidación: {k['swaps_consolidacion']} intercambios · dispersión media por SKU "
               f"{k['dispersion_antes_m']:.1f} → {k['dispersion_despues_m']:.1f} m · "
               f"la jerarquía ABC cuesta {k['brecha_jerarquia_vs_lap_global_pct']:.2f}% frente al LAP global.")
    st.dataframe(res["assignment"], width="stretch", height=320)
    st.download_button("Descargar mapa de slots (CSV)", _csv(res["assignment"]), "slot_map.csv")


TOOLS = {
    "Full Pallet Picking (Slotting)": tool_full_pallet_picking,
    # "Case Picking": tool_case_picking,   # siguiente módulo
}


def main():
    st.set_page_config(page_title="Wiser", layout="wide")
    st.sidebar.title("Wiser")
    choice = st.sidebar.radio("Herramienta", list(TOOLS))
    TOOLS[choice]()


main()
