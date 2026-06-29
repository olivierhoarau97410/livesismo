"""
LIVESISMO — Sismicité live · île de La Réunion
=======================================================================
Auteur : Olivier Hoarau — juin 2026
Stack  : Streamlit · ObsPy (FDSN IPGP) · Plotly · Folium

Lancement local  : streamlit run app.py
Données terrain  : générer data/terrain_250m.npz avec tools/build_dem.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, timezone
import plotly.graph_objects as go
import plotly.express as px

# ── Config page (doit être le 1er appel Streamlit) ───────────────────────────
st.set_page_config(
    page_title="LIVESISMO — île de La Réunion",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Fond général clair */
[data-testid="stAppViewContainer"] {
    background: #f5f5f7;
    color: #1a1a2e;
}
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e0e0e8;
}

/* Titres */
h1, h2, h3 { color: #1a1a2e; }

/* Métriques */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #dde0ea;
    border-radius: 10px;
    padding: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] { color: #666680; font-size: 0.8rem; }
[data-testid="stMetricValue"] { color: #1a1a2e; font-size: 1.5rem; }
[data-testid="stMetricDelta"] { font-size: 0.8rem; }

/* Onglets */
[data-testid="stTabs"] button {
    color: #666680;
    font-weight: 600;
    font-size: 0.9rem;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ef5350;
    border-bottom: 3px solid #ef5350;
}

/* Séparateur sidebar */
hr { border-color: #e0e0e8; }

/* Desktop : header propre, sidebar toujours ouverte */
@media (min-width: 768px) {
    [data-testid="stToolbar"]               { display: none !important; }
    [data-testid="stDecoration"]            { display: none !important; }
    header[data-testid="stHeader"]          { display: none !important; }
    /* Cacher le bouton fermer dans la sidebar */
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    /* Forcer la sidebar ouverte même si localStorage dit "collapsed" */
    [data-testid="stSidebar"][aria-expanded="false"] {
        transform: none !important;
        width: 21rem !important;
        visibility: visible !important;
    }
    /* Cacher le bouton hamburger desktop (sidebar toujours là) */
    [data-testid="collapsedControl"] { display: none !important; }
}
/* Mobile : on ne touche à RIEN — Streamlit gère nativement le hamburger */
[data-testid="stAppViewContainer"] > .main { padding-top: 0.5rem !important; }

/* Badge source live */
.badge-live {
    display:inline-block; padding:2px 8px; border-radius:12px;
    background:#ef5350; color:white; font-size:0.7rem; font-weight:700;
    margin-left:6px; vertical-align:middle;
}
.badge-hist {
    display:inline-block; padding:2px 8px; border-radius:12px;
    background:#eeeef5; color:#666; font-size:0.7rem;
    margin-left:6px; vertical-align:middle;
}
</style>
""", unsafe_allow_html=True)

# ── Imports modules internes ──────────────────────────────────────────────────
from modules.fetch   import get_all_events, apply_filters, load_historique, fetch_fdsn
from modules.carte2d import make_carte
from modules.vue3d   import make_vue3d_html
import streamlit.components.v1 as components


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Filtres & contrôles
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🌋 LIVESISMO")
    st.markdown("Piton de la Fournaise · IPGP / OVPF")
    st.divider()

    # ── Bloc 1 : Sismicité récente ────────────────────────────────────────────
    st.markdown("""<div style='background:#fff3f3;border-left:4px solid #ef5350;
        border-radius:6px;padding:8px 12px;margin-bottom:8px'>
        <b style='color:#c62828'>⚡ SISMICITÉ RÉCENTE</b><br>
        <small style='color:#888'>Source : FDSN IPGP (live, cache 1h)</small>
        </div>""", unsafe_allow_html=True)

    days_live = st.slider("Fenêtre temporelle (jours)", 7, 365, 90, step=7,
                          help="Données téléchargées depuis FDSN IPGP, rechargées automatiquement toutes les heures")

    if st.button("🔄 Forcer le rafraîchissement", use_container_width=True):
        # Vide le cache FDSN (session) + le cache CSV (server)
        keys_to_clear = [k for k in st.session_state if k.startswith("fdsn_cache_")]
        for k in keys_to_clear:
            del st.session_state[k]
        load_historique.clear()
        st.rerun()

    st.divider()

    # ── Bloc 2 : Sismicité pluridécennale ────────────────────────────────────
    st.markdown("""<div style='background:#f3f6ff;border-left:4px solid #5c6bc0;
        border-radius:6px;padding:8px 12px;margin-bottom:8px'>
        <b style='color:#283593'>📚 SISMICITÉ PLURIDÉCENNALE</b><br>
        <small style='color:#888'>Source : catalogue IPGP 2005–2025</small>
        </div>""", unsafe_allow_html=True)

    show_historique = st.checkbox("Afficher les 20 ans de données", value=True,
                                  help="Décocher pour ne voir que la sismicité récente (fenêtre FDSN ci-dessus)")

    if show_historique:
        _hist_info = load_historique()
        if not _hist_info.empty:
            _d0 = _hist_info["date"].min().strftime("%d/%m/%Y")
            _d1 = _hist_info["date"].max().strftime("%d/%m/%Y")
            st.caption(f"Base disponible : {_d0} → {_d1} · {len(_hist_info):,} événements")
        col1, col2 = st.columns(2)
        with col1:
            d_start = st.date_input("Depuis", date(2005, 1, 1),
                                    min_value=date(2005, 1, 1),
                                    max_value=date.today(),
                                    format="DD/MM/YYYY")
        with col2:
            d_end = st.date_input("Jusqu'au", date.today(),
                                  min_value=date(2005, 1, 1),
                                  max_value=date.today(),
                                  format="DD/MM/YYYY")
    else:
        d_start, d_end = None, None

    st.divider()

    # ── Filtres communs ───────────────────────────────────────────────────────
    st.markdown("**🎚 Filtres communs**")
    mag_range = st.slider("Magnitude", -1.0, 7.0, (0.0, 7.0), step=0.1)
    dep_range = st.slider("Profondeur (km)", 0, 60, (0, 40))

    st.divider()

    # ── Options d'affichage ───────────────────────────────────────────────────
    st.markdown("**🗺 Carte 2D**")
    basemap = st.selectbox("Fond de carte",
                           ["CartoDB positron", "OpenStreetMap",
                            "CartoDB dark_matter",
                            "Esri World Imagery", "Esri Topo"])
    cluster_pts = st.checkbox("Regrouper les points", value=True)

    st.divider()
    st.markdown("**🏔 Vue 3D**")
    vert_exag = st.slider("Exagération verticale", 1.0, 10.0, 4.0, step=0.5)

    st.divider()
    st.caption("Données : FDSN IPGP · RGEALTI IGN 1 m · OVPF")

# ═══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

with st.spinner("Chargement des données…"):
    # Sismicité récente (FDSN live, toujours chargée)
    df_live = fetch_fdsn(days=days_live)
    if "source" not in df_live.columns:
        df_live["source"] = "fdsn_live"

    # Sismicité historique (CSV local, seulement si activée)
    if show_historique:
        df_hist = load_historique()
        # Éviter les doublons avec le live : on garde l'historique avant le début du live
        if not df_live.empty and not df_hist.empty:
            cutoff = df_live["date"].min()
            df_hist = df_hist[df_hist["date"] < cutoff]
        df_all = pd.concat([df_hist, df_live], ignore_index=True).sort_values("date", ascending=False)
    else:
        df_all = df_live.copy()

if df_all.empty:
    st.error("Aucune donnée disponible. Vérifiez la connexion FDSN et le fichier data/seismes_historique.csv")
    st.stop()

# Filtres magnitude, profondeur et dates (les dates filtrent tout le dataset)
df = apply_filters(df_all, mag_range[0], mag_range[1],
                   dep_range[0], dep_range[1], d_start, d_end)

n_live = int((df["source"] == "fdsn_live").sum()) if "source" in df.columns else 0
n_hist = len(df) - n_live

# ═══════════════════════════════════════════════════════════════════════════════
#  EN-TÊTE — métriques
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🌋 LIVESISMO — île de La Réunion")
st.markdown("Sismicité live en temps différé · Réseau PF · IPGP / OVPF")

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("Séismes (filtrés)", f"{len(df):,}",
              help="Nombre d'événements correspondant aux filtres actifs")
with m2:
    mag_max = df["magnitude"].max() if not df.empty else 0
    st.metric("Magnitude max", f"{mag_max:.1f}" if not np.isnan(mag_max) else "—")
with m3:
    last_date = df["date"].max() if not df.empty else None
    st.metric("Dernier event", str(last_date)[:16] if last_date else "—")
with m4:
    st.metric("Live (FDSN)", f"{n_live}", delta=f"derniers {days_live}j",
              delta_color="normal")
with m5:
    st.metric("Historique", f"{n_hist:,}", delta="CSV local")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
#  ONGLETS PRINCIPAUX
# ═══════════════════════════════════════════════════════════════════════════════

tab_carte, tab_3d, tab_stats, tab_data = st.tabs(
    ["🗺  Carte régionale", "🌋  Vue 3D profondeur", "📊  Statistiques", "📋  Données brutes"]
)

# ── Tab 1 : Carte 2D Folium ────────────────────────────────────────────────────
with tab_carte:
    from streamlit_folium import st_folium

    st.markdown(f"### Carte sismique — {len(df)} événements")
    if not df.empty:
        recent = df[df["date"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)]
        if not recent.empty:
            st.info(f"⚡ **{len(recent)} séismes ces 7 derniers jours** "
                    f"— dernier : M{recent.iloc[0]['magnitude']:.1f} "
                    f"le {str(recent.iloc[0]['date'])[:16]} UTC")

    carte = make_carte(df, cluster=cluster_pts, basemap=basemap)
    st_folium(carte, width="100%", height=600, returned_objects=[])

    with st.expander("ℹ️ À propos de la carte"):
        st.markdown("""
        - **Cadre blanc pointillé** : zone de 230 km × 230 km centrée sur La Réunion
        - **Taille des cercles** proportionnelle à la magnitude
        - **Couleur** : gris < M1 · bleu M1–2 · vert M2–3 · orange M3–4 · rouge M4–5 · bordeaux ≥ M5
        - Cliquer sur un point pour voir les détails (date, magnitude, profondeur)
        """)

# ── Tab 2 : Vue 3D ────────────────────────────────────────────────────────────
with tab_3d:
    st.markdown("### Vue 3D — Séismes localisés sous la surface volcanique")
    st.caption("Les séismes apparaissent à leur profondeur réelle sous le relief — "
               "plus le point est bas, plus le foyer est profond.")

    src_label = "historique + récents" if show_historique else f"récents ({days_live}j)"
    st.caption(f"Affichage : {len(df)} événements ({src_label}) · "
               "🖱 Glisser=rotation · ⇧+Glisser=panoramique · Molette=zoom · Double-clic=reset")

    html_3d = make_vue3d_html(df, vert_exag=vert_exag)
    components.html(html_3d, height=640, scrolling=False)

# ── Tab 3 : Statistiques ──────────────────────────────────────────────────────
with tab_stats:
    st.markdown("### Analyse statistique de la sismicité")

    if df.empty:
        st.warning("Aucun événement pour les filtres actifs.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            # Timeline : nb de séismes par semaine
            df_t = df.copy()
            df_t["semaine"] = df_t["date"].dt.to_period("W").apply(lambda p: p.start_time)
            df_t["semaine"] = pd.to_datetime(df_t["semaine"])
            timeline = df_t.groupby("semaine").size().reset_index(name="count")
            fig_tl = px.bar(
                timeline, x="semaine", y="count",
                title="Nombre de séismes par semaine",
                labels={"semaine": "Semaine", "count": "Nombre"},
                color_discrete_sequence=["#ef5350"],
            )
            fig_tl.update_layout(
                paper_bgcolor="#ffffff", plot_bgcolor="#f9f9fb",
                font_color="#444", title_font_color="#1a1a2e",
            )
            st.plotly_chart(fig_tl, use_container_width=True)

        with col_b:
            # Distribution magnitudes
            fig_mag = px.histogram(
                df.dropna(subset=["magnitude"]),
                x="magnitude",
                nbins=30,
                title="Distribution des magnitudes",
                labels={"magnitude": "Magnitude"},
                color_discrete_sequence=["#4fc3f7"],
            )
            fig_mag.update_layout(
                paper_bgcolor="#ffffff", plot_bgcolor="#f9f9fb",
                font_color="#444", title_font_color="#1a1a2e",
            )
            st.plotly_chart(fig_mag, use_container_width=True)

        col_c, col_d = st.columns(2)

        with col_c:
            # Profondeur vs Magnitude
            fig_sc = px.scatter(
                df.dropna(subset=["magnitude", "profondeur_km"]),
                x="magnitude", y="profondeur_km",
                color="magnitude",
                color_continuous_scale="RdYlGn_r",
                title="Profondeur vs Magnitude",
                labels={"profondeur_km": "Profondeur (km)", "magnitude": "Magnitude"},
                opacity=0.6,
            )
            fig_sc.update_yaxes(autorange="reversed")
            fig_sc.update_layout(
                paper_bgcolor="#ffffff", plot_bgcolor="#f9f9fb",
                font_color="#444", title_font_color="#1a1a2e",
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        with col_d:
            # Sismicité par année
            if "annee" in df.columns:
                par_an = df.groupby("annee").agg(
                    nb=("magnitude", "count"),
                    mag_moy=("magnitude", "mean"),
                    mag_max=("magnitude", "max"),
                ).reset_index()
                fig_an = go.Figure()
                fig_an.add_bar(x=par_an["annee"], y=par_an["nb"],
                               name="Nb séismes", marker_color="#6f42c1")
                fig_an.add_scatter(x=par_an["annee"], y=par_an["mag_max"],
                                   name="Mag max", yaxis="y2",
                                   line=dict(color="#ef5350", width=2), mode="lines+markers")
                fig_an.update_layout(
                    title="Sismicité annuelle",
                    paper_bgcolor="#ffffff", plot_bgcolor="#f9f9fb",
                    font_color="#444", title_font_color="#1a1a2e",
                    yaxis=dict(title="Nb séismes", color="#6f42c1"),
                    yaxis2=dict(title="Magnitude max", overlaying="y",
                                side="right", color="#ef5350"),
                    legend=dict(bgcolor="rgba(255,255,255,0.8)"),
                )
                st.plotly_chart(fig_an, use_container_width=True)

        # Statistiques textuelles
        st.markdown("#### Résumé")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Magnitude moyenne", f"{df['magnitude'].mean():.2f}")
        with col_s2:
            st.metric("Prof. moyenne", f"{df['profondeur_km'].mean():.1f} km")
        with col_s3:
            st.metric("M ≥ 3", f"{len(df[df['magnitude'] >= 3])}")
        with col_s4:
            st.metric("Période couverte",
                      f"{(df['date'].max() - df['date'].min()).days} jours")

# ── Tab 4 : Données brutes ────────────────────────────────────────────────────
with tab_data:
    st.markdown(f"### Données brutes — {len(df)} événements")

    cols_show = [c for c in ["date", "magnitude", "type_magnitude",
                              "profondeur_km", "latitude", "longitude",
                              "evaluation", "source"] if c in df.columns]

    df_show = df[cols_show].copy()
    df_show["date"] = df_show["date"].astype(str).str[:19]

    st.dataframe(
        df_show,
        use_container_width=True,
        height=500,
        column_config={
            "magnitude"     : st.column_config.NumberColumn("Mag.", format="%.1f"),
            "profondeur_km" : st.column_config.NumberColumn("Prof. (km)", format="%.1f"),
            "latitude"      : st.column_config.NumberColumn("Lat.", format="%.4f"),
            "longitude"     : st.column_config.NumberColumn("Lon.", format="%.4f"),
        }
    )

    csv_bytes = df_show.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Télécharger CSV",
        data=csv_bytes,
        file_name=f"seismes_reunion_{date.today()}.csv",
        mime="text/csv",
    )
