"""
carte2d.py — Carte Folium 230 km × 230 km centrée sur La Réunion
Séismes colorés par magnitude, taille proportionnelle.
"""

import folium
import pandas as pd
import numpy as np
from folium.plugins import MarkerCluster, MiniMap

# ── Paramètres géographiques ──────────────────────────────────────────────────
CENTER      = (-21.11, 55.54)   # centre de La Réunion
# 230 km → ≈ 1.04° lat, 1.12° lon (à lat 21°)
BOX_DLAT    = 1.036
BOX_DLON    = 1.112
BOX_BOUNDS  = [
    [CENTER[0] - BOX_DLAT, CENTER[1] - BOX_DLON],   # SW
    [CENTER[0] + BOX_DLAT, CENTER[1] + BOX_DLON],   # NE
]

# ── Couleur selon magnitude ───────────────────────────────────────────────────
def _mag_color(mag: float) -> str:
    if pd.isna(mag) or mag < 1:   return "#a0a0a0"
    if mag < 2:                    return "#4fc3f7"
    if mag < 3:                    return "#81c784"
    if mag < 4:                    return "#ffb74d"
    if mag < 5:                    return "#ef5350"
    return "#b71c1c"

def _mag_radius(mag: float) -> float:
    if pd.isna(mag): return 3
    return max(3, min(22, 2 + 3 ** (mag - 0.5)))

# ── Construction de la carte ──────────────────────────────────────────────────

def make_carte(df: pd.DataFrame,
               cluster: bool = True,
               show_box: bool = True,
               basemap: str = "CartoDB dark_matter") -> folium.Map:
    """
    Retourne une carte Folium prête à injecter dans Streamlit.
    df doit contenir : latitude, longitude, magnitude, profondeur_km, date
    """
    # Fonds de carte natifs folium (sans attribution externe requise)
    TILES_NATIFS = {"CartoDB positron", "CartoDB dark_matter", "OpenStreetMap"}

    # Fonds personnalisés avec attribution
    TILES_CUSTOM = {
        "Esri World Imagery": {
            "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "attr": "Esri, Maxar, Earthstar Geographics",
            "name": "Esri World Imagery",
        },
        "Esri Topo": {
            "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            "attr": "Esri, HERE, Garmin, FAO, NOAA, USGS",
            "name": "Esri Topo",
        },
    }

    if basemap in TILES_NATIFS:
        m = folium.Map(location=CENTER, zoom_start=10, tiles=basemap, max_bounds=True)
    elif basemap in TILES_CUSTOM:
        cfg = TILES_CUSTOM[basemap]
        m = folium.Map(location=CENTER, zoom_start=10, tiles=None, max_bounds=True)
        folium.TileLayer(tiles=cfg["tiles"], attr=cfg["attr"], name=cfg["name"]).add_to(m)
    else:
        m = folium.Map(location=CENTER, zoom_start=9, tiles="CartoDB positron", max_bounds=True)

    # Rectangle 230 km
    if show_box:
        folium.Rectangle(
            bounds=BOX_BOUNDS,
            color="#ffffff",
            weight=1.5,
            dash_array="6 4",
            fill=False,
            tooltip="Zone 230 km × 230 km",
        ).add_to(m)

    # Marqueur sommet Piton de la Fournaise
    folium.Marker(
        location=(-21.244, 55.712),
        tooltip="🌋 Piton de la Fournaise (2 632 m)",
        icon=folium.Icon(color="red", icon="fire", prefix="fa"),
    ).add_to(m)

    # Marqueur Piton des Neiges
    folium.Marker(
        location=(-21.091, 55.483),
        tooltip="⛰ Piton des Neiges (3 070 m)",
        icon=folium.Icon(color="darkblue", icon="mountain", prefix="fa"),
    ).add_to(m)

    if df.empty:
        MiniMap(toggle_display=True).add_to(m)
        return m

    # Couche séismes
    layer = MarkerCluster(name="Séismes", disableClusteringAtZoom=10) if cluster else folium.FeatureGroup(name="Séismes")

    for _, row in df.iterrows():
        lat = row.get("latitude")
        lon = row.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            continue

        mag   = row.get("magnitude", np.nan)
        depth = row.get("profondeur_km", np.nan)
        date  = row.get("date", "")
        src   = row.get("source", "")

        depth_str = f"{depth:.1f} km" if not pd.isna(depth) else "?"
        mag_str   = f"{mag:.1f}" if not pd.isna(mag) else "?"
        date_str  = str(date)[:16] if date else "?"

        popup_html = f"""
        <div style='font-family:monospace;font-size:12px;min-width:180px'>
          <b style='color:#ef5350'>🌊 Séisme</b><br>
          <b>Date :</b> {date_str} UTC<br>
          <b>Magnitude :</b> {mag_str}<br>
          <b>Profondeur :</b> {depth_str}<br>
          <b>Lat / Lon :</b> {lat:.4f} / {lon:.4f}<br>
          <small style='color:#999'>{src}</small>
        </div>"""

        folium.CircleMarker(
            location=(lat, lon),
            radius=_mag_radius(mag),
            color=_mag_color(mag),
            fill=True,
            fill_color=_mag_color(mag),
            fill_opacity=0.75,
            weight=0.8,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"M{mag_str} — {depth_str}",
        ).add_to(layer)

    layer.add_to(m)

    # Légende magnitude (coin bas-gauche, au-dessus du zoom)
    legend_html = """
    <div style='
        position:fixed; bottom:30px; left:12px; z-index:1000;
        background:rgba(255,255,255,0.93); border-radius:8px;
        padding:10px 14px; font-family:monospace; font-size:12px; color:#333;
        border:1px solid #ccc; box-shadow:0 2px 8px rgba(0,0,0,0.12);'>
      <b style='color:#1a1a2e'>Magnitude</b><br>
      <span style='color:#4fc3f7'>●</span> &lt; 2 &nbsp;
      <span style='color:#43a047'>●</span> 2–3 &nbsp;
      <span style='color:#fb8c00'>●</span> 3–4<br>
      <span style='color:#ef5350'>●</span> 4–5 &nbsp;
      <span style='color:#b71c1c'>●</span> ≥ 5
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    MiniMap(toggle_display=True).add_to(m)

    return m
