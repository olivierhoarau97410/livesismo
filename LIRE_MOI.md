# LIVESISMO — Sismicité live Piton de la Fournaise

## Lancement rapide (local)

```bash
cd /Desktop/mes-applis/LIVESISMO
pip install -r requirements.txt
streamlit run app.py
```

## Étape obligatoire avant la vue 3D réaliste : générer le DEM

Le fichier `data/terrain_250m.npz` n'est pas dans le repo (trop lourd).
Il se génère **une seule fois** en local à partir des tuiles IGN :

```bash
python tools/build_dem.py
# Durée : 3 à 8 min — génère data/terrain_250m.npz (~6 Mo)
```

Sans ce fichier, la vue 3D fonctionne quand même avec un terrain synthétique
(deux gaussiennes pour le Piton des Neiges et le Piton de la Fournaise).

## Déploiement Streamlit Cloud

1. `git init` puis push sur GitHub (inclure `data/terrain_250m.npz` si < 100 Mo)
2. Connecter le repo sur share.streamlit.io
3. Les events frais sont rechargés automatiquement toutes les heures via `@st.cache_data(ttl=3600)`

## Structure

```
LIVESISMO/
├── app.py                      ← point d'entrée Streamlit
├── requirements.txt
├── modules/
│   ├── fetch.py                ← FDSN IPGP + fusion CSV historique
│   ├── carte2d.py              ← carte Folium 230 km × 230 km
│   └── vue3d.py                ← surface 3D Plotly + séismes en profondeur
├── data/
│   ├── seismes_historique.csv  ← 20 ans de sismicité (2005-2025)
│   └── terrain_250m.npz        ← DEM Réunion 250 m (généré par build_dem.py)
└── tools/
    └── build_dem.py            ← mosaïque des 2 641 tuiles ASC IGN → .npz
```

## Données

- **Historique** : 2 068 événements 2005–2025 (FDSN IPGP, extrait via sismoILO.py)
- **Live** : réseau PF, FDSN IPGP, fenêtre glissante configurable (7 à 365 jours)
- **Terrain** : RGEALTI IGN 1 m (tuiles ASC dans `../Carto RUN/RGEALTI_1M/`)
