"""
scripts/generate_boulangerie.py
Génère data/boulangerie.csv — dataset complet pour Boulangerie Atlas.
"""
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path

np.random.seed(1337)

ROOT  = Path(__file__).parent.parent
OUT   = ROOT / "data" / "boulangerie.csv"

# ── Période ───────────────────────────────────────────────────────────────
START = date(2023, 1, 1)
END   = date(2025, 3, 31)

# ── Produits : nom → (prix MAD, base_qty/jour, catégorie) ────────────────
PRODUCTS = {
    # Pains & galettes
    "Pain Traditionnel":   (4.0,  95, "pain"),
    "Msemen":              (2.0,  55, "galette"),
    "Harcha":              (3.0,  40, "galette"),
    "Beghrir":             (4.0,  22, "galette"),
    # Viennoiseries
    "Croissant":           (5.5,  32, "viennoiserie"),
    "Pain au Chocolat":    (6.5,  24, "viennoiserie"),
    "Brioche":             (9.0,  16, "viennoiserie"),
    "Chausson aux Pommes": (7.0,  14, "viennoiserie"),
    # Patisseries
    "Mille-feuille":       (13.0, 12, "patisserie"),
    "Tarte aux Fraises":   (12.0, 10, "patisserie"),
    "Eclair au Chocolat":  (9.0,  11, "patisserie"),
    "Sellou":              (10.0,  8, "patisserie"),   # specialite Ramadan
}

# ── Ramadan ───────────────────────────────────────────────────────────────
RAMADAN = [
    (date(2023,  3, 23), date(2023,  4, 20)),
    (date(2024,  3, 11), date(2024,  4,  9)),
    (date(2025,  3,  1), date(2025,  3, 29)),
]

# ── Fetes marocaines → multiplicateur ─────────────────────────────────────
FETES = {
    date(2023,  1,  1): 1.8, date(2023,  1, 11): 1.3,
    date(2023,  4, 21): 2.4, date(2023,  4, 22): 2.0,
    date(2023,  5,  1): 1.4,
    date(2023,  6, 28): 2.2, date(2023,  6, 29): 1.9,
    date(2023,  7, 30): 1.6,
    date(2023,  8, 20): 1.4, date(2023,  8, 21): 1.3,
    date(2023,  9, 27): 1.5,
    date(2023, 11,  6): 1.3, date(2023, 11, 18): 1.5,
    date(2023, 12, 31): 1.5,
    date(2024,  1,  1): 1.9, date(2024,  1, 11): 1.3,
    date(2024,  4, 10): 2.5, date(2024,  4, 11): 2.1,
    date(2024,  5,  1): 1.4,
    date(2024,  6, 17): 2.3, date(2024,  6, 18): 2.0,
    date(2024,  7,  8): 1.4, date(2024,  7, 30): 1.7,
    date(2024,  8, 20): 1.4, date(2024,  8, 21): 1.3,
    date(2024,  9, 15): 1.5,
    date(2024, 11,  6): 1.3, date(2024, 11, 18): 1.5,
    date(2024, 12, 31): 1.6,
    date(2025,  1,  1): 2.0, date(2025,  1, 11): 1.3,
    date(2025,  3, 30): 2.5, date(2025,  3, 31): 2.0,
}

# ── Vacances scolaires ─────────────────────────────────────────────────────
VACANCES = [
    (date(2023,  3,  6), date(2023,  3, 19)),
    (date(2023,  6, 30), date(2023,  9,  4)),
    (date(2023, 11,  6), date(2023, 11, 12)),
    (date(2023, 12, 25), date(2024,  1,  7)),
    (date(2024,  3,  4), date(2024,  3, 17)),
    (date(2024,  6, 28), date(2024,  9,  2)),
    (date(2024, 11,  4), date(2024, 11, 10)),
    (date(2024, 12, 23), date(2025,  1,  6)),
    (date(2025,  3,  3), date(2025,  3, 16)),
]

# ── Distributions horaires par categorie et type de journee ───────────────
# pain: 6h-9h (matin) + 17h-19h (soir)
# galette: 6h-9h tres fort
# viennoiserie: 7h-10h + 16h-18h
# patisserie: 9h-11h + 15h-18h
HOUR_DIST = {
    "pain": {
        "normal":  {6:.12, 7:.22, 8:.20, 9:.10, 12:.08,
                    13:.05, 17:.12, 18:.08, 19:.03},
        "weekend": {6:.06, 7:.18, 8:.22, 9:.18, 10:.08,
                    12:.07, 13:.06, 17:.10, 18:.05},
        "ramadan": {3:.10, 4:.08, 6:.05,
                    17:.20, 18:.30, 19:.20, 20:.07},
    },
    "galette": {
        "normal":  {6:.18, 7:.30, 8:.25, 9:.12, 10:.05,
                    17:.06, 18:.04},
        "weekend": {6:.10, 7:.24, 8:.28, 9:.20, 10:.10,
                    11:.05, 17:.03},
        "ramadan": {3:.12, 4:.10, 17:.18, 18:.32, 19:.20, 20:.08},
    },
    "viennoiserie": {
        "normal":  {7:.20, 8:.28, 9:.18, 10:.08,
                    16:.12, 17:.10, 18:.04},
        "weekend": {7:.14, 8:.24, 9:.22, 10:.18, 11:.08,
                    16:.08, 17:.06},
        "ramadan": {7:.06, 8:.08,
                    17:.20, 18:.35, 19:.22, 20:.09},
    },
    "patisserie": {
        "normal":  {9:.12, 10:.18, 11:.12,
                    15:.18, 16:.22, 17:.14, 18:.04},
        "weekend": {9:.10, 10:.16, 11:.16, 12:.10,
                    15:.16, 16:.18, 17:.12, 18:.02},
        "ramadan": {9:.04, 10:.04,
                    17:.18, 18:.34, 19:.26, 20:.14},
    },
}

MONTH_MULT = {
    1:1.05, 2:1.02, 3:1.00, 4:0.98, 5:1.00, 6:0.97,
    7:0.90, 8:0.88, 9:1.02, 10:1.05, 11:1.08, 12:1.12,
}

def in_ramadan(d):
    return any(s <= d <= e for s, e in RAMADAN)

def in_vacances(d):
    return any(s <= d <= e for s, e in VACANCES)

total_days = (END - START).days + 1

# ~25 jours de fermeture aleatoire sur 27 mois
closed_days = set()
for _ in range(25):
    closed_days.add(START + timedelta(days=int(np.random.randint(0, total_days))))

rows = []
cur = START

while cur <= END:
    if cur in closed_days:
        cur += timedelta(days=1)
        continue

    dow    = cur.weekday()   # 0=Lun … 6=Dim
    is_fri = dow == 4
    is_sat = dow == 5
    is_wknd = is_fri or is_sat
    is_ram  = in_ramadan(cur)
    is_hol  = cur in FETES
    is_vac  = in_vacances(cur)
    n_days  = (cur - START).days

    # Tendance : +30 % sur 27 mois
    trend = 1.0 + (n_days / (total_days - 1)) * 0.30

    # Saisonnalite mensuelle
    month_m = MONTH_MULT[cur.month]

    # Multiplicateur global
    mult = 1.0 * month_m
    if is_wknd:          mult *= 1.42
    if is_ram:           mult *= 1.25
    if is_hol:           mult *= FETES[cur]
    if is_vac and not is_ram:
        mult *= 1.08

    # Mauvaise journee ~5 %
    if np.random.random() < 0.05:
        mult *= np.random.uniform(0.25, 0.55)

    for prod, (prix, base, cat) in PRODUCTS.items():
        slot   = "ramadan" if is_ram else ("weekend" if is_wknd else "normal")
        dist   = HOUR_DIST[cat][slot]
        hours  = list(dist.keys())
        weights = np.array(list(dist.values()), dtype=float)
        weights /= weights.sum()

        pm = mult
        if is_ram:
            if prod in ("Msemen", "Beghrir"):            pm *= 3.5
            if prod == "Harcha":                         pm *= 2.0
            if prod == "Sellou":                         pm *= 5.0
            if prod in ("Croissant", "Pain au Chocolat",
                        "Chausson aux Pommes"):          pm *= 0.55
        if is_hol and cat == "patisserie":               pm *= 1.40
        if is_wknd and cat == "viennoiserie":            pm *= 1.18
        if is_vac  and cat in ("viennoiserie",
                                "patisserie"):           pm *= 1.15

        noise   = 1 + np.random.normal(0, 0.13)
        qty_day = max(1, int(round(base * pm * trend * noise)))

        hourly = np.random.multinomial(qty_day, weights)
        for h, q in zip(hours, hourly):
            if q > 0:
                rows.append({
                    "date":            cur.strftime("%Y-%m-%d"),
                    "produit":         prod,
                    "heure":           int(h),
                    "quantite_vendue": int(q),
                    "prix_unitaire":   float(prix),
                })

    cur += timedelta(days=1)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False, encoding="utf-8")

# ── Resume ────────────────────────────────────────────────────────────────
daily = df.groupby("date")["quantite_vendue"].sum()
ca    = (df["quantite_vendue"] * df["prix_unitaire"]).sum()

print(f"Fichier  : {OUT}")
print(f"Lignes   : {len(df):,}")
print(f"Jours    : {df['date'].nunique()}")
print(f"Periode  : {df['date'].min()} -> {df['date'].max()}")
print(f"Produits : {df['produit'].nunique()}")
print(f"Heures   : {sorted(df['heure'].unique())}")
print(f"CA total : {ca:,.0f} MAD")
print(f"Moy/jour : {daily.mean():.0f} unites")
print(f"Record   : {daily.max():.0f} unites le {daily.idxmax()}")
print(f"Pire     : {daily.min():.0f} unites le {daily.idxmin()}")
print()
print("=== Ventes par produit ===")
pt = df.groupby("produit")["quantite_vendue"].sum()
px = df.drop_duplicates("produit").set_index("produit")["prix_unitaire"]
for p in pt.sort_values(ascending=False).index:
    ca_p = pt[p] * px[p]
    print(f"  {p:<24} {int(pt[p]):>7} unites   {ca_p:>10,.0f} MAD   @{px[p]:.1f} MAD")
print()
print("=== Ramadan 2024 vs Normal ===")
ram24 = df[(df["date"] >= "2024-03-11") & (df["date"] <= "2024-04-09")]
hors  = df[(df["date"] < "2024-03-11") | (df["date"] > "2024-04-09")]
nd_r  = ram24["date"].nunique()
nd_h  = hors["date"].nunique()
for p in ["Msemen", "Sellou", "Beghrir", "Croissant", "Pain Traditionnel"]:
    r = ram24[ram24["produit"] == p]["quantite_vendue"].sum() / max(nd_r, 1)
    h = hors[hors["produit"]   == p]["quantite_vendue"].sum() / max(nd_h, 1)
    ratio = r / h if h > 0 else 0
    print(f"  {p:<24} Ramadan:{r:5.1f}/j  Normal:{h:5.1f}/j  x{ratio:.1f}")
