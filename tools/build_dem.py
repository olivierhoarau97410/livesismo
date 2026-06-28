"""
build_dem.py — Outil de préparation du DEM (usage unique, local)
=================================================================
Lit les 2 641 tuiles ASC (RGEALTI 1 m, RGR92/UTM40S) présentes dans
../Carto RUN/RGEALTI_1M/ et génère un fichier data/terrain_250m.npz
contenant :
  - lat   : tableau 1D de latitudes  (nord→sud)
  - lon   : tableau 1D de longitudes (ouest→est)
  - Z     : matrice 2D d'altitudes (m)  shape=(len(lat), len(lon))

Exécution (depuis le dossier LIVESISMO/) :
    python tools/build_dem.py

Dépendances : numpy, pyproj  (pip install numpy pyproj)
Durée : 3 à 8 minutes selon le Mac (lecture de ~2 600 fichiers).
"""

import sys
import time
from pathlib import Path

import numpy as np
from pyproj import Transformer

# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
ASC_DIR  = ROOT.parent / "Carto RUN" / "RGEALTI_1M"
OUT_FILE = ROOT / "data" / "terrain_250m.npz"
RESOL    = 250   # résolution de sortie en mètres (UTM)

# ── Projection RGR92/UTM40S → WGS84 ──────────────────────────────────────────
# EPSG:2975 = RGR92 / UTM zone 40S
transformer = Transformer.from_crs("EPSG:2975", "EPSG:4326", always_xy=True)


def read_asc_header(path: Path) -> dict:
    """Lit uniquement l'en-tête d'un fichier ASC (6 lignes)."""
    h = {}
    with open(path, "r") as f:
        for _ in range(6):
            key, val = f.readline().split()
            h[key.lower()] = float(val)
    return h


def read_asc_data(path: Path, header: dict) -> np.ndarray:
    """Charge la grille de valeurs d'un fichier ASC."""
    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    data  = np.loadtxt(path, skiprows=6, max_rows=nrows)
    nodata = header.get("nodata_value", -99999)
    data = data.astype(np.float32)
    data[data == nodata] = np.nan
    return data.reshape(nrows, ncols)


def main():
    print("=" * 60)
    print("  build_dem.py — Mosaïque DEM La Réunion")
    print("=" * 60)

    asc_files = sorted(ASC_DIR.glob("*.asc"))
    if not asc_files:
        print(f"❌ Aucun fichier .asc trouvé dans {ASC_DIR}")
        sys.exit(1)

    print(f"📂 {len(asc_files)} tuiles trouvées dans {ASC_DIR.name}/")

    # ── Passe 1 : déterminer le bounding-box global UTM ──────────────────────
    print("🔍 Passe 1 : lecture des en-têtes…")
    t0 = time.time()
    x_min_g = y_min_g = +1e12
    x_max_g = y_max_g = -1e12
    cell_size = None

    for p in asc_files:
        try:
            h = read_asc_header(p)
        except Exception:
            continue
        cs = h["cellsize"]
        if cell_size is None:
            cell_size = cs
        x0  = h["xllcorner"]
        y0  = h["yllcorner"]
        ncols = int(h["ncols"])
        nrows = int(h["nrows"])
        x_min_g = min(x_min_g, x0)
        y_min_g = min(y_min_g, y0)
        x_max_g = max(x_max_g, x0 + ncols * cs)
        y_max_g = max(y_max_g, y0 + nrows * cs)

    print(f"   Emprise UTM : E {x_min_g:.0f}–{x_max_g:.0f}  N {y_min_g:.0f}–{y_max_g:.0f}")
    print(f"   Résolution source : {cell_size:.0f} m → sortie : {RESOL} m")

    # ── Grille de sortie en UTM ───────────────────────────────────────────────
    step = RESOL
    xs = np.arange(x_min_g, x_max_g, step)
    ys = np.arange(y_max_g, y_min_g, -step)    # nord → sud
    ncols_out = len(xs)
    nrows_out = len(ys)
    print(f"   Grille de sortie : {nrows_out} × {ncols_out} ({nrows_out*ncols_out/1e6:.1f} Mpix)")

    Z_out = np.full((nrows_out, ncols_out), np.nan, dtype=np.float32)
    Z_cnt = np.zeros_like(Z_out, dtype=np.uint8)

    # ── Passe 2 : remplissage ─────────────────────────────────────────────────
    print(f"\n📦 Passe 2 : remplissage de la grille…")
    for i, p in enumerate(asc_files):
        if i % 200 == 0:
            pct = i / len(asc_files) * 100
            elapsed = time.time() - t0
            print(f"   {i}/{len(asc_files)}  ({pct:.0f}%)  —  {elapsed:.0f}s écoulées")

        try:
            h   = read_asc_header(p)
            data = read_asc_data(p, h)
        except Exception:
            continue

        cs    = h["cellsize"]
        x0    = h["xllcorner"]
        y0    = h["yllcorner"]
        nr, nc = data.shape

        # Coordonnées UTM des coins de la tuile
        x_tile_max = x0 + nc * cs
        y_tile_max = y0 + nr * cs   # coin NE

        # Indices dans la grille de sortie
        ix0 = int((x0 - x_min_g) / step)
        ix1 = int((x_tile_max - x_min_g) / step)
        iy0 = int((y_max_g - y_tile_max) / step)
        iy1 = int((y_max_g - y0) / step)

        ix1 = min(ix1, ncols_out)
        iy1 = min(iy1, nrows_out)
        if ix0 >= ncols_out or iy0 >= nrows_out or ix1 <= 0 or iy1 <= 0:
            continue

        out_r = iy1 - iy0
        out_c = ix1 - ix0

        # Rééchantillonnage grossier : sous-échantillon de la tuile
        row_idx = np.round(np.linspace(0, nr - 1, out_r)).astype(int)
        col_idx = np.round(np.linspace(0, nc - 1, out_c)).astype(int)
        patch   = data[np.ix_(row_idx, col_idx)]

        iy0c = max(iy0, 0); iy1c = min(iy1, nrows_out)
        ix0c = max(ix0, 0); ix1c = min(ix1, ncols_out)
        pr = slice(iy0c - iy0, iy0c - iy0 + (iy1c - iy0c))
        pc = slice(ix0c - ix0, ix0c - ix0 + (ix1c - ix0c))

        mask_valid = ~np.isnan(patch[pr, pc])
        Z_out[iy0c:iy1c, ix0c:ix1c] = np.where(
            mask_valid,
            patch[pr, pc],
            Z_out[iy0c:iy1c, ix0c:ix1c]
        )
        Z_cnt[iy0c:iy1c, ix0c:ix1c] += mask_valid.astype(np.uint8)

    # Lissage léger pour combler les trous résiduels
    from scipy.ndimage import uniform_filter
    Z_out = np.where(np.isnan(Z_out), 0.0, Z_out)
    Z_out = uniform_filter(Z_out, size=3).astype(np.float32)

    # ── Conversion UTM → lat/lon ──────────────────────────────────────────────
    print("\n🌍 Conversion UTM → WGS84 lat/lon…")
    # On convertit uniquement les axes (1D)
    lon_1d, _ = transformer.transform(xs, np.full_like(xs, ys[len(ys)//2]))
    _, lat_1d = transformer.transform(np.full_like(ys, xs[len(xs)//2]), ys)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_FILE, lat=lat_1d.astype(np.float32),
                                  lon=lon_1d.astype(np.float32),
                                  Z=Z_out)

    total = time.time() - t0
    fsize = OUT_FILE.stat().st_size / 1e6
    print(f"\n✅ Fichier généré : {OUT_FILE}")
    print(f"   Taille : {fsize:.1f} Mo  |  Grille : {nrows_out}×{ncols_out}")
    print(f"   Durée totale : {total:.0f} s")
    print("\nTu peux maintenant lancer : streamlit run app.py")


if __name__ == "__main__":
    main()
