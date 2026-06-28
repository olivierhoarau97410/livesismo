"""
fetch.py — Récupération des événements sismiques
Combine l'historique local (CSV) avec les événements frais (FDSN IPGP).
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st

DATA_DIR = Path(__file__).parent.parent / "data"
HIST_CSV  = DATA_DIR / "seismes_historique.csv"

# Coordonnées La Réunion (zone élargie PF + île entière)
ZONE_ILE = dict(minlat=-22.15, maxlat=-20.07, minlon=54.43, maxlon=56.65)  # boîte 230 km
ZONE_PF  = dict(minlat=-21.35, maxlat=-21.15, minlon=55.60, maxlon=55.85)

# ── Chargement de l'historique local ─────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_historique() -> pd.DataFrame:
    """Charge le CSV des 20 ans (embarqué dans le repo)."""
    if not HIST_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(HIST_CSV, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "date"])
    df["source"] = "historique"
    return df

# ── Fetch FDSN (caché par session navigateur) ─────────────────────────────────

def fetch_fdsn(days: int = 90, zone: str = "ile") -> pd.DataFrame:
    """
    Télécharge les événements récents depuis FDSN IPGP.
    Résultat mis en cache dans st.session_state :
      - 1 seul fetch par session (par onglet navigateur)
      - le bouton 'Forcer le rafraîchissement' vide ce cache pour l'user courant
    """
    cache_key = f"fdsn_cache_{days}_{zone}"

    if cache_key in st.session_state:
        return st.session_state[cache_key]

    try:
        from obspy.clients.fdsn import Client
        from obspy import UTCDateTime

        client = Client("IPGP")
        coords = ZONE_PF if zone == "pf" else ZONE_ILE

        fin    = UTCDateTime.now()
        debut  = fin - days * 86400

        cat = client.get_events(
            starttime=debut, endtime=fin,
            **coords,
            orderby="time",
        )

        rows = []
        for ev in cat:
            orig = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
            if not orig:
                continue
            mag_obj = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
            rows.append({
                "date"        : str(orig.time),
                "latitude"    : orig.latitude,
                "longitude"   : orig.longitude,
                "profondeur_km": round(orig.depth / 1000, 2) if orig.depth else np.nan,
                "magnitude"   : round(mag_obj.mag, 2) if mag_obj else np.nan,
                "type_magnitude": mag_obj.magnitude_type if mag_obj else "",
                "evaluation"  : orig.evaluation_mode or "",
                "source"      : "fdsn_live",
            })

        df = pd.DataFrame(rows)
        if df.empty:
            st.session_state[cache_key] = df
            return df
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df["annee"] = df["date"].dt.year
        st.session_state[cache_key] = df
        return df

    except Exception as e:
        st.warning(f"⚠️ FDSN IPGP inaccessible : {e}")
        return pd.DataFrame()


# ── Fusion historique + live ──────────────────────────────────────────────────

def get_all_events(days_live: int = 90) -> pd.DataFrame:
    """
    Retourne un DataFrame unifié :
      - Historique complet (CSV local)
      - Événements frais FDSN (derniers N jours), dédupliqués
    """
    hist = load_historique()
    live = fetch_fdsn(days=days_live)

    if hist.empty and live.empty:
        return pd.DataFrame()

    if hist.empty:
        return live
    if live.empty:
        return hist

    # Dédupliquation : on retire de l'historique les dates déjà dans live
    cutoff = live["date"].min()
    hist_old = hist[hist["date"] < cutoff].copy()

    combined = pd.concat([hist_old, live], ignore_index=True)
    combined = combined.sort_values("date", ascending=False).reset_index(drop=True)

    # Colonne annee au cas où absente
    if "annee" not in combined.columns:
        combined["annee"] = combined["date"].dt.year

    return combined


# ── Filtre utilitaire ─────────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame,
                  mag_min: float, mag_max: float,
                  depth_min: float, depth_max: float,
                  date_start, date_end) -> pd.DataFrame:
    """Applique les filtres sidebar sur le DataFrame complet."""
    mask = (
        (df["magnitude"].between(mag_min, mag_max, inclusive="both")) &
        (df["profondeur_km"].between(depth_min, depth_max, inclusive="both"))
    )
    if date_start and date_end:
        ds = pd.Timestamp(date_start, tz="UTC")
        de = pd.Timestamp(date_end,   tz="UTC")
        mask &= (df["date"] >= ds) & (df["date"] <= de)
    return df[mask].copy()
