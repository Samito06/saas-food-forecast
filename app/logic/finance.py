import pandas as pd


def build_fin_df(df: pd.DataFrame, price_map: dict, cost_map: dict) -> pd.DataFrame:
    """Attach per-row revenue, cost, and gross-margin columns."""
    out = df.copy()
    out["prix_unit"]   = out["produit"].map(price_map).fillna(0.0)
    out["cout_unit"]   = out["produit"].map(cost_map).fillna(0.0)
    out["recette"]     = out["quantite_vendue"] * out["prix_unit"]
    out["cout_mp"]     = out["quantite_vendue"] * out["cout_unit"]
    out["marge_brute"] = out["recette"] - out["cout_mp"]
    return out
