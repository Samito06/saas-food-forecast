"""
app/main.py — FoodCast | Production-ready SaaS dashboard
Predict · Decide · Grow
Run: streamlit run app/main.py
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.util import hash_pandas_object
from prophet import Prophet

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
DEFAULT_CSV  = ROOT / "data" / "boulangerie.csv"
REQUIRED_COLS = {"date", "produit", "quantite_vendue"}
WEEKEND_DAYS  = {"Friday", "Saturday"}
BUFFER_RATE   = 1.15
PEAK_THRESH   = 1.30
UNIT_OPTIONS  = ["g", "kg", "ml", "L", "piece", "tbsp", "tsp", "oz", "portion"]

# ── Moroccan fixed public holidays (month, day, name) ─────────────────────
MAROC_FIXED_HOLIDAYS = [
    (1,  1,  "Jour de l'An"),
    (1,  11, "Manifeste de l'Indépendance"),
    (5,  1,  "Fête du Travail"),
    (7,  30, "Fête du Trône"),
    (8,  14, "Allégeance des Provinces du Sud"),
    (8,  20, "Révolution du Roi et du Peuple"),
    (8,  21, "Fête de la Jeunesse"),
    (11, 6,  "Marche Verte"),
    (11, 18, "Fête de l'Indépendance"),
]

# Islamic moveable holidays — updated through 2027
ISLAMIC_HOLIDAYS: dict[int, list[tuple[str, str]]] = {
    2022: [("2022-05-02","Aïd al-Fitr"),("2022-05-03","Aïd al-Fitr J2"),
           ("2022-07-09","Aïd al-Adha"),("2022-07-10","Aïd al-Adha J2"),
           ("2022-07-30","Aïd al-Hijra"),("2022-10-08","Mawlid")],
    2023: [("2023-04-21","Aïd al-Fitr"),("2023-04-22","Aïd al-Fitr J2"),
           ("2023-06-28","Aïd al-Adha"),("2023-06-29","Aïd al-Adha J2"),
           ("2023-07-19","Aïd al-Hijra"),("2023-09-27","Mawlid")],
    2024: [("2024-04-10","Aïd al-Fitr"),("2024-04-11","Aïd al-Fitr J2"),
           ("2024-06-17","Aïd al-Adha"),("2024-06-18","Aïd al-Adha J2"),
           ("2024-07-08","Aïd al-Hijra"),("2024-09-15","Mawlid")],
    2025: [("2025-03-31","Aïd al-Fitr"),("2025-04-01","Aïd al-Fitr J2"),
           ("2025-06-07","Aïd al-Adha"),("2025-06-08","Aïd al-Adha J2"),
           ("2025-06-26","Aïd al-Hijra"),("2025-09-04","Mawlid")],
    2026: [("2026-03-20","Aïd al-Fitr"),("2026-03-21","Aïd al-Fitr J2"),
           ("2026-05-27","Aïd al-Adha"),("2026-05-28","Aïd al-Adha J2"),
           ("2026-06-16","Aïd al-Hijra"),("2026-08-25","Mawlid")],
    2027: [("2027-03-09","Aïd al-Fitr"),("2027-03-10","Aïd al-Fitr J2"),
           ("2027-05-16","Aïd al-Adha"),("2027-05-17","Aïd al-Adha J2"),
           ("2027-06-06","Aïd al-Hijra"),("2027-08-14","Mawlid")],
}

# Ramadan start/end dates — first day of fasting → last day of fasting
RAMADAN_DATES: dict[int, tuple[str, str]] = {
    2022: ("2022-04-02", "2022-05-01"),
    2023: ("2023-03-23", "2023-04-20"),
    2024: ("2024-03-11", "2024-04-09"),
    2025: ("2025-03-01", "2025-03-29"),
    2026: ("2026-02-18", "2026-03-18"),
    2027: ("2027-02-07", "2027-03-08"),
}

# ── i18n: French & Arabic translations ───────────────────────────────────
EN_DAYS_IDX = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}

TRANS: dict[str, dict] = {
    "fr": {
        # Navigation
        "nav_dashboard":       "Tableau de bord",
        "nav_saisie":          "Saisie des ventes",
        "nav_overview":        "Vue d'ensemble",
        "nav_forecast":        "Prévisions",
        "nav_analyse":         "Analyse",
        "nav_finances":        "Finances",
        "nav_recommendations": "Recommandations",
        "nav_products":        "Produits",
        "nav_shopping":        "Liste de courses",
        "nav_conseils":        "Conseils",
        "nav_about":           "À propos",
        # Sidebar
        "sidebar_file":        "FICHIER DE VENTES",
        "sidebar_demo":        "Boulangerie Atlas (démo) — importez votre fichier ci-dessus.",
        "sidebar_ramadan":     "Mode Ramadan",
        "sidebar_ramadan_help":"Active les alertes Iftar, produits spéciaux et indicateurs sur les graphiques.",
        "sidebar_lite":        "Mode économique",
        "sidebar_lite_help":   "Désactive les prévisions IA (Prophet) pour économiser la bande passante. Idéal avec une connexion faible.",
        # Dashboard
        "dash_title":          "Tableau de bord",
        "sec_today":           "Résumé du jour",
        "sec_perf":            "Indicateur de performance",
        "sec_ranking":         "Classement des produits",
        "sec_week":            "Ventes de la semaine",
        "sec_history":         "Historique — 30 derniers jours",
        # KPIs
        "kpi_sold":            "Unités vendues",
        "kpi_yesterday":       "vs hier",
        "kpi_avg":             "vs moyenne",
        "kpi_best":            "Meilleur produit du jour",
        "kpi_avg_day":         "Moyenne / jour ouvert",
        "kpi_best_day_ever":   "Record historique",
        "kpi_open_days":       "Jours ouverts",
        "kpi_yesterday_sub":   "hier : {qty} {u}",
        "kpi_avg_sub":         "moy. : {avg:.0f} {u}/j",
        # Performance labels
        "perf_good":           "Bonne journée",
        "perf_avg":            "Journée moyenne",
        "perf_bad":            "Mauvaise journée",
        "perf_good_sub":       "Bien au-dessus de la moyenne",
        "perf_avg_sub":        "Dans la fourchette habituelle",
        "perf_bad_sub":        "En dessous de la moyenne",
        "perf_vs":             "de la moyenne",
        # Ranking
        "rank_pos":            "#",
        "rank_product":        "Produit",
        "rank_total":          "Total vendu",
        "rank_share":          "Part",
        "rank_trend":          "Tendance",
        "rank_caption":        "↗ hausse · → stable · ↘ baisse (comparaison 30 derniers jours vs 30 jours précédents)",
        # History table
        "hist_date":           "Date",
        "hist_day":            "Jour",
        "hist_qty":            "Quantité",
        "hist_vs_avg":         "vs moy.",
        "hist_perf":           "Bilan",
        # Days (Mon … Sun)
        "days":                ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"],
        "units":               "unités",
        "lang_label":          "Langue",
        # Saisie page
        "saisie_title":        "Saisie des ventes",
        "saisie_sub":          "Entrez vos ventes du jour sans modifier votre fichier CSV. Téléchargez ensuite le fichier complet pour le réimporter.",
        "saisie_nofile":       "Aucun fichier chargé — vous pouvez saisir vos ventes manuellement. Téléchargez ensuite le CSV généré et importez-le comme fichier de données.",
        "saisie_new_prod":     "Ajouter un produit à la liste",
        "saisie_add_btn":      "Ajouter",
        "saisie_date_label":   "Date des ventes",
        "saisie_save_btn":     "Enregistrer les ventes",
        "saisie_save_btn2":    "Enregistrer",
        "saisie_sec_entry":    "Ventes du jour",
        "saisie_history":      "Historique des saisies manuelles",
        "saisie_export":       "Exporter / Synchroniser",
        "saisie_sync_tip":     "💡 Vos saisies sont conservées pendant cette session. Téléchargez le fichier complet ci-dessous pour ne pas les perdre, puis importez-le au prochain démarrage.",
        "saisie_download":     "Télécharger fichier complet (CSV + saisies)",
        "saisie_clear":        "Tout effacer",
        "saisie_empty":        "Aucune saisie manuelle pour le moment. Utilisez le formulaire ci-dessus.",
        "saisie_success":      "{n} produit(s) enregistré(s) pour le {d}.",
        "saisie_warn":         "Entrez au moins une quantité > 0.",
        "saisie_kpi_days":     "Jours saisis",
        "saisie_kpi_days_sub": "via saisie manuelle",
        "saisie_kpi_total":    "Total saisi",
        "saisie_kpi_total_sub":"unités au total",
        "saisie_already_csv":  "Déjà dans votre CSV pour cette date.",
        # Forecast page
        "forecast_title":      "Prévisions des ventes",
        "forecast_sub":        "Prévisions IA Prophet avec prise en compte des fêtes marocaines.",
        "forecast_lite_banner":"⚡ Mode économique activé — Prévisions IA désactivées. Désactivez-le dans la barre latérale pour voir les prévisions Prophet.",
        "forecast_lite_section":"Tendance récente (30 derniers jours)",
        "forecast_horizon":    "Combien de jours ?",
        "forecast_horizon_fmt":"{n} jours",
        "forecast_acc_good":   "Très bonne",
        "forecast_acc_ok":     "Bonne",
        "forecast_acc_fair":   "Correcte",
        "forecast_acc_na":     "N/D",
        "forecast_acc_sub":    "erreur moy. : {mae} unités/j",
        "forecast_acc_label":  "Précision des prévisions",
        "forecast_acc_nodata": "Pas assez de données",
        "forecast_spinner":    "Chargement de votre prévision...",
        "forecast_chart_sec":  "Historique + prévision {n} jours",
        "forecast_history":    "Ventes passées",
        "forecast_pred":       "Prévision IA",
        "forecast_range":      "Plage prévue",
        "forecast_fc_start":   "Début prévision ({d})",
        "forecast_peaks":      "Alertes — Pics imminents",
        "forecast_ram_sec":    "Mode Ramadan — Jours concernés",
        "forecast_ram_msg":    "**{n} jours** de la prévision tombent pendant le Ramadan. Les ventes du soir (Iftar) peuvent représenter 60–80% du chiffre journalier. Préparez davantage de **Harira, Chebakia, Briouates** et boissons sucrées.",
        "forecast_weekly":     "Résumé hebdomadaire",
        "forecast_monthly":    "Résumé mensuel",
        "forecast_weekly_cols":["Semaine","Jours","Total prévu","Moy. jour","Min prévu","Max prévu"],
        "forecast_monthly_cols":["Mois","Total prévu","Moy. jour","Min prévu","Max prévu"],
        # Analyse page
        "analyse_title":       "Analyse avancée",
        "analyse_sub":         "Comparaison semaines, prévisions par produit, calendrier des fêtes marocaines et meilleure heure de vente.",
        "analyse_spinner":     "Calcul des prévisions...",
        "analyse_week":        "Semaine en cours vs semaine dernière",
        "analyse_cur_week":    "Semaine en cours",
        "analyse_cur_week_sub":"7 derniers jours",
        "analyse_prev_week":   "Semaine dernière",
        "analyse_prev_week_sub":"7 jours précédents",
        "analyse_delta":       "Évolution",
        "analyse_units_sold":  "Unités vendues",
        "analyse_hour":        "Meilleure heure de vente",
        "analyse_best_hour":   "Meilleure heure",
        "analyse_peak_avg":    "Pic moyen : {qty:.1f} unités",
        "analyse_no_hour":     "Pour activer cette fonctionnalité, ajoutez une colonne **`heure`** (0–23) à votre fichier CSV. Exemple : `2024-01-15, Pizza Margherita, 12, 8`",
        "analyse_prod":        "Prévision par produit",
        "analyse_prod_pick":   "Choisir un produit",
        "analyse_horizon":     "Horizon",
        "analyse_horizon_fmt": "{n} jours",
        "analyse_prod_lite":   "⚡ Mode économique — prévisions par produit désactivées.",
        "analyse_prod_empty":  "Aucun produit trouvé dans vos données.",
        "analyse_prod_nodata": "Pas assez de données pour **{p}** ({n} jours — minimum 30).",
        "analyse_prod_spinner":"Prévision pour {p}...",
        "analyse_prod_hist":   "Moy. historique",
        "analyse_prod_next7":  "Prévu (7 prochains j.)",
        "analyse_prod_units":  "unités/jour (jours ouverts)",
        "analyse_prod_total":  "unités au total",
        "analyse_interval":    "Intervalle",
        "analyse_pred_label":  "Prévision IA",
        "analyse_hist_label":  "Historique",
        "analyse_fc_start":    "Début prévision",
        "analyse_calendar":    "Calendrier des fêtes marocaines",
        "analyse_cal_nat":     "Fête nationale",
        "analyse_cal_isl":     "Fête islamique",
        "analyse_cal_ram":     "Ramadan",
        "analyse_ram_msg":     "**Mode Ramadan activé** — Pendant le Ramadan, vos ventes se concentrent sur la tranche **19h – 23h** (Iftar & Tarawih). Préparez davantage de : Harira, Chebakia, Sellou, Briouates, Shabakiya, jus frais et dattes. Réduisez les préparations matinales de 30–40%.",
        "analyse_trend_up":    "hausse",
        "analyse_trend_stable":"stable",
        "analyse_trend_down":  "baisse",
        "analyse_trend_caption":"↗ {up} · → {stable} · ↘ {down} (comparaison 30 derniers jours vs 30 jours précédents)",
        # Finances page
        "fin_title":           "Finances & Rentabilité",
        "fin_sub":             "Chiffre d'affaires, marges, objectifs et estimation du bénéfice mensuel.",
        "fin_config":          "Configurer les prix de vente et coûts de revient",
        "fin_config_caption":  "Entrez le **prix de vente** (ce que vous encaissez) et le **coût unitaire** (ingrédients + emballage) pour chaque produit. Toutes les valeurs sont en **MAD**.",
        "fin_auto_price":      "Colonne `prix_unitaire` détectée dans votre fichier — prix pré-remplis.",
        "fin_col_prod":        "Produit",
        "fin_col_price":       "Prix vente (MAD)",
        "fin_col_cost":        "Coût unitaire (MAD)",
        "fin_col_price_help":  "Prix auquel vous vendez une portion/unité",
        "fin_col_cost_help":   "Coût total d'ingrédients + emballage par unité produite",
        "fin_warn_no_price":   "Aucun prix de vente configuré — les métriques financières afficheront 0 MAD. Renseignez les prix dans le tableau ci-dessus pour activer toutes les analyses.",
        "fin_ca_section":      "Chiffre d'affaires",
        "fin_kpi_day":         "CA du jour",
        "fin_kpi_week":        "CA de la semaine",
        "fin_kpi_month":       "CA du mois",
        "fin_kpi_margin":      "Marge brute mois",
        "fin_margin_sub":      "marge : {m}",
        "fin_margin_pct":      "{pct:.1f}% du CA",
        "fin_target_section":  "Objectif de CA journalier",
        "fin_target_input":    "Objectif (MAD/jour)",
        "fin_target_above":    "{pct:.0f}% atteint · {ca} / objectif {target}",
        "fin_target_days":     "Jours ayant atteint l'objectif dans l'historique : **{n} / {total}**",
        "fin_target_empty":    "Entrez un objectif à gauche pour voir votre progression.",
        "fin_ca_series":       "CA journalier",
        "fin_cost_series":     "Coût matières premières",
        "fin_target_line":     "Objectif {t}/j",
        "fin_prod_section":    "Produit le plus rentable vs le plus vendu",
        "fin_best_seller":     "Plus vendu",
        "fin_best_margin":     "Plus rentable",
        "fin_bubble_x":        "Unités vendues",
        "fin_bubble_y":        "Marge / unité (MAD)",
        "fin_bubble_size":     "CA total (MAD)",
        "fin_table_cols":      ["Produit","Unités vendues","CA total","Coût total MP","Marge brute","Marge %","Marge/unité"],
        "fin_no_price_info":   "Configurez les prix ci-dessus pour voir cette analyse.",
        "fin_monthly_section": "Recettes vs dépenses matières premières",
        "fin_rev_series":      "Recettes (CA)",
        "fin_mp_series":       "Coût MP",
        "fin_margin_series":   "Marge brute",
        "fin_profit_section":  "Estimation du bénéfice mensuel",
        "fin_month_cols":      ["Mois","CA (MAD)","MP (MAD)","Marge (MAD)","Marge %"],
        "fin_total_sub":       "CA : {ca} · Coût MP : {cout} · Taux : {pct:.1f}%",
        "fin_total_label":     "Marge brute totale sur toute la période",
        "fin_no_config_info":  "Configurez les prix et coûts ci-dessus pour activer cette section.",
        "fin_footer":          "Marge brute = CA − Coût matières premières. Elle ne tient pas compte du loyer, de la masse salariale ni des charges fixes. Une marge brute ≥ 60% est généralement saine pour la restauration rapide.",
        # Conseils page
        "conseils_hero_title": "Aide à la décision",
        "conseils_hero_sub":   "Conseils personnalisés basés sur votre historique — mis à jour en temps réel selon vos données.",
        "conseils_pdj":        "⭐ Produit du jour à mettre en avant",
        "conseils_pdj_badge":  "Mettre en avant aujourd'hui",
        "conseils_pdj_score":  "Score {s}/100 | Moyenne {d} : {qty:.0f} unités{margin}",
        "conseils_pdj_margin": " | Marge : {m:.1f} MAD/unité",
        "conseils_pdj_empty":  "Pas assez de données pour calculer le produit du jour.",
        "conseils_prep":       "📋 Prépare plus — les 7 prochains jours",
        "conseils_prep_normal":"Journée habituelle — pas de pic identifié.",
        "conseils_promo":      "🏷️ Suggestions de prix promotionnel",
        "conseils_promo_thresh":"Seuil de baisse pour déclencher une promo (%)",
        "conseils_promo_help": "Un produit est signalé si ses ventes des 7 derniers jours ont chuté de X% par rapport aux 30 derniers jours.",
        "conseils_promo_note": "Prix promotionnel calculé en appliquant la remise au prix unitaire configuré dans Finances.",
        "conseils_promo_config":"configurez le prix",
        "conseils_promo_ok":   "✅ Aucun produit en baisse significative sur les 7 derniers jours.",
        "conseils_dormant":    "😴 Alertes — produits non vendus récemment",
        "conseils_dormant_slider":"Signaler si absent depuis (jours)",
        "conseils_dormant_help":"Un produit est signalé s'il n'apparaît pas dans vos ventes depuis X jours.",
        "conseils_dormant_since":"Dernière vente : {d} — absent depuis {n} jours",
        "conseils_dormant_never":"Dernière vente : jamais — absent depuis +{n} jours",
        "conseils_dormant_ok": "✅ Tous les produits ont été vendus dans les derniers jours.",
        "conseils_seasonal":   "🌿 Idées de nouveaux produits — saison & événements",
        "conseils_seasonal_note":"Suggestions basées sur les tendances du marché marocain et la saisonnalité locale. Adaptez-les à votre clientèle.",
        "conseils_seasonal_empty":"Aucune suggestion saisonnière disponible pour cette période.",
        "conseils_ram_current":"Ramadan — en ce moment",
        "conseils_ram_soon":   "Ramadan — dans 30 jours",
        "conseils_month_now":  "{m} — produits de saison",
        "conseils_month_next": "{m} — anticipez le mois prochain",
    },
    "ar": {
        # Navigation
        "nav_dashboard":       "لوحة القيادة",
        "nav_saisie":          "إدخال المبيعات",
        "nav_overview":        "نظرة عامة",
        "nav_forecast":        "التوقعات",
        "nav_analyse":         "التحليل",
        "nav_finances":        "الماليات",
        "nav_recommendations": "التوصيات",
        "nav_products":        "المنتجات",
        "nav_shopping":        "قائمة التسوق",
        "nav_conseils":        "النصائح",
        "nav_about":           "حول",
        # Sidebar
        "sidebar_file":        "ملف المبيعات",
        "sidebar_demo":        "مخبزة أطلس (تجريبي) — استورد ملفك أعلاه.",
        "sidebar_ramadan":     "وضع رمضان",
        "sidebar_ramadan_help":"تفعيل تنبيهات الإفطار والمنتجات الخاصة.",
        "sidebar_lite":        "الوضع الاقتصادي",
        "sidebar_lite_help":   "تعطيل توقعات الذكاء الاصطناعي لتوفير النطاق الترددي.",
        # Dashboard
        "dash_title":          "لوحة القيادة",
        "sec_today":           "ملخص اليوم",
        "sec_perf":            "مؤشر الأداء",
        "sec_ranking":         "ترتيب المنتجات",
        "sec_week":            "مبيعات الأسبوع",
        "sec_history":         "سجل آخر 30 يوماً",
        "kpi_sold":            "الوحدات المباعة",
        "kpi_yesterday":       "مقارنة بالأمس",
        "kpi_avg":             "مقارنة بالمعدل",
        "kpi_best":            "أفضل منتج اليوم",
        "kpi_avg_day":         "المعدل اليومي",
        "kpi_best_day_ever":   "أفضل يوم",
        "kpi_open_days":       "أيام العمل",
        "kpi_yesterday_sub":   "أمس : {qty} {u}",
        "kpi_avg_sub":         "المعدل : {avg:.0f} {u}/ي",
        "perf_good":           "يوم ممتاز ⭐",
        "perf_avg":            "يوم عادي 📊",
        "perf_bad":            "يوم ضعيف ⚠️",
        "perf_good_sub":       "أعلى بكثير من المعدل",
        "perf_avg_sub":        "ضمن النطاق المعتاد",
        "perf_bad_sub":        "أقل من المعدل",
        "perf_vs":             "من المعدل",
        "rank_pos":            "#",
        "rank_product":        "المنتج",
        "rank_total":          "إجمالي المبيعات",
        "rank_share":          "الحصة",
        "rank_trend":          "الاتجاه",
        "rank_caption":        "↗ ارتفاع · → مستقر · ↘ انخفاض (مقارنة آخر 30 يوم بـ30 يوم سابقة)",
        "hist_date":           "التاريخ",
        "hist_day":            "اليوم",
        "hist_qty":            "الكمية",
        "hist_vs_avg":         "مقارنة",
        "hist_perf":           "التقييم",
        "days":                ["إث","ثل","أر","خم","جم","سب","أح"],
        "units":               "وحدة",
        "lang_label":          "اللغة",
        # Saisie
        "saisie_title":        "إدخال المبيعات",
        "saisie_sub":          "أدخل مبيعات اليوم دون تعديل ملف CSV. ثم حمّل الملف الكامل لإعادة استيراده.",
        "saisie_nofile":       "لا يوجد ملف — يمكنك إدخال المبيعات يدوياً.",
        "saisie_new_prod":     "إضافة منتج إلى القائمة",
        "saisie_add_btn":      "إضافة",
        "saisie_date_label":   "تاريخ المبيعات",
        "saisie_save_btn":     "حفظ المبيعات",
        "saisie_save_btn2":    "حفظ",
        "saisie_sec_entry":    "مبيعات اليوم",
        "saisie_history":      "سجل الإدخالات اليدوية",
        "saisie_export":       "تصدير / مزامنة",
        "saisie_sync_tip":     "💡 يتم حفظ إدخالاتك خلال هذه الجلسة. حمّل الملف الكامل أدناه وأعد استيراده في الجلسة التالية.",
        "saisie_download":     "تحميل الملف الكامل (CSV + الإدخالات)",
        "saisie_clear":        "مسح الكل",
        "saisie_empty":        "لا توجد إدخالات يدوية حتى الآن. استخدم النموذج أعلاه.",
        "saisie_success":      "تم حفظ {n} منتج(ات) بتاريخ {d}.",
        "saisie_warn":         "أدخل كمية > 0 على الأقل.",
        "saisie_kpi_days":     "أيام مُدخلة",
        "saisie_kpi_days_sub": "عبر الإدخال اليدوي",
        "saisie_kpi_total":    "إجمالي مُدخل",
        "saisie_kpi_total_sub":"وحدات إجمالاً",
        "saisie_already_csv":  "موجود بالفعل في CSV لهذا التاريخ.",
        # Forecast
        "forecast_title":      "توقعات المبيعات",
        "forecast_sub":        "توقعات مدعومة بالذكاء الاصطناعي مع مراعاة العطل المغربية.",
        "forecast_lite_banner":"⚡ الوضع الاقتصادي مفعّل — توقعات الذكاء الاصطناعي معطلة.",
        "forecast_lite_section":"الاتجاه الأخير (آخر 30 يوم)",
        "forecast_horizon":    "كم عدد الأيام؟",
        "forecast_horizon_fmt":"{n} يوم",
        "forecast_acc_good":   "ممتازة",
        "forecast_acc_ok":     "جيدة",
        "forecast_acc_fair":   "مقبولة",
        "forecast_acc_na":     "غير متاح",
        "forecast_acc_sub":    "خطأ متوسط : {mae} وحدة/يوم",
        "forecast_acc_label":  "دقة التوقعات",
        "forecast_acc_nodata": "بيانات غير كافية",
        "forecast_spinner":    "جارٍ تحميل التوقعات...",
        "forecast_chart_sec":  "السجل التاريخي + توقع {n} يوم",
        "forecast_history":    "مبيعات سابقة",
        "forecast_pred":       "توقع الذكاء الاصطناعي",
        "forecast_range":      "النطاق المتوقع",
        "forecast_fc_start":   "بداية التوقع ({d})",
        "forecast_peaks":      "تنبيهات — ذروات وشيكة",
        "forecast_ram_sec":    "وضع رمضان — الأيام المعنية",
        "forecast_ram_msg":    "**{n} يوم** من التوقع يقع في رمضان. مبيعات المساء (الإفطار) قد تمثل 60-80% من الرقم اليومي. جهّز المزيد من **الحريرة والشباكية والبريوات** والمشروبات الحلوة.",
        "forecast_weekly":     "ملخص أسبوعي",
        "forecast_monthly":    "ملخص شهري",
        "forecast_weekly_cols":["الأسبوع","الأيام","الإجمالي","معدل/يوم","الحد الأدنى","الحد الأقصى"],
        "forecast_monthly_cols":["الشهر","الإجمالي","معدل/يوم","الحد الأدنى","الحد الأقصى"],
        # Analyse
        "analyse_title":       "تحليل متقدم",
        "analyse_sub":         "مقارنة الأسابيع، توقعات المنتجات، تقويم الأعياد المغربية وأفضل ساعة بيع.",
        "analyse_spinner":     "جارٍ حساب التوقعات...",
        "analyse_week":        "الأسبوع الحالي مقابل الأسبوع الماضي",
        "analyse_cur_week":    "الأسبوع الحالي",
        "analyse_cur_week_sub":"آخر 7 أيام",
        "analyse_prev_week":   "الأسبوع الماضي",
        "analyse_prev_week_sub":"7 أيام سابقة",
        "analyse_delta":       "التطور",
        "analyse_units_sold":  "وحدات مباعة",
        "analyse_hour":        "أفضل ساعة بيع",
        "analyse_best_hour":   "أفضل ساعة",
        "analyse_peak_avg":    "ذروة متوسطة : {qty:.1f} وحدة",
        "analyse_no_hour":     "لتفعيل هذه الميزة، أضف عموداً **`heure`** (0-23) إلى ملف CSV.",
        "analyse_prod":        "توقع حسب المنتج",
        "analyse_prod_pick":   "اختر منتجاً",
        "analyse_horizon":     "الأفق",
        "analyse_horizon_fmt": "{n} يوم",
        "analyse_prod_lite":   "⚡ الوضع الاقتصادي — توقعات المنتجات معطلة.",
        "analyse_prod_empty":  "لا توجد منتجات في بياناتك.",
        "analyse_prod_nodata": "بيانات غير كافية لـ **{p}** ({n} يوم — الحد الأدنى 30).",
        "analyse_prod_spinner":"توقع لـ {p}...",
        "analyse_prod_hist":   "المتوسط التاريخي",
        "analyse_prod_next7":  "متوقع (7 أيام القادمة)",
        "analyse_prod_units":  "وحدة/يوم (أيام مفتوحة)",
        "analyse_prod_total":  "وحدات إجمالاً",
        "analyse_interval":    "النطاق",
        "analyse_pred_label":  "توقع ذكاء اصطناعي",
        "analyse_hist_label":  "السجل التاريخي",
        "analyse_fc_start":    "بداية التوقع",
        "analyse_calendar":    "تقويم الأعياد المغربية",
        "analyse_cal_nat":     "عيد وطني",
        "analyse_cal_isl":     "عيد إسلامي",
        "analyse_cal_ram":     "رمضان",
        "analyse_ram_msg":     "**وضع رمضان مفعّل** — خلال رمضان، تتركز مبيعاتك في الفترة **19:00 – 23:00** (الإفطار والتراويح). جهّز المزيد من: الحريرة، الشباكية، السلو، البريوات، العصائر الطازجة والتمر. قلّل التحضيرات الصباحية بنسبة 30–40%.",
        "analyse_trend_up":    "ارتفاع",
        "analyse_trend_stable":"مستقر",
        "analyse_trend_down":  "انخفاض",
        "analyse_trend_caption":"↗ {up} · → {stable} · ↘ {down}",
        # Finances
        "fin_title":           "المالية والربحية",
        "fin_sub":             "رقم الأعمال والهوامش والأهداف وتقدير الربح الشهري.",
        "fin_config":          "تهيئة أسعار البيع وتكاليف الإنتاج",
        "fin_config_caption":  "أدخل **سعر البيع** (ما تحصله) و**التكلفة الوحدوية** (مكونات + تغليف) لكل منتج. جميع القيم بالدرهم (MAD).",
        "fin_auto_price":      "تم اكتشاف عمود `prix_unitaire` — تم تعبئة الأسعار تلقائياً.",
        "fin_col_prod":        "المنتج",
        "fin_col_price":       "سعر البيع (MAD)",
        "fin_col_cost":        "التكلفة الوحدوية (MAD)",
        "fin_col_price_help":  "السعر الذي تبيع به الوحدة",
        "fin_col_cost_help":   "إجمالي تكلفة المكونات والتغليف لكل وحدة",
        "fin_warn_no_price":   "لم يتم تهيئة أي أسعار — ستعرض المقاييس المالية 0 درهم.",
        "fin_ca_section":      "رقم الأعمال",
        "fin_kpi_day":         "CA اليوم",
        "fin_kpi_week":        "CA الأسبوع",
        "fin_kpi_month":       "CA الشهر",
        "fin_kpi_margin":      "الهامش الشهري",
        "fin_margin_sub":      "الهامش : {m}",
        "fin_margin_pct":      "{pct:.1f}% من CA",
        "fin_target_section":  "هدف CA اليومي",
        "fin_target_input":    "الهدف (MAD/يوم)",
        "fin_target_above":    "{pct:.0f}% محقق · {ca} / الهدف {target}",
        "fin_target_days":     "أيام تحقق الهدف في السجل : **{n} / {total}**",
        "fin_target_empty":    "أدخل هدفاً على اليسار لرؤية تقدمك.",
        "fin_ca_series":       "CA اليومي",
        "fin_cost_series":     "تكلفة المواد",
        "fin_target_line":     "الهدف {t}/يوم",
        "fin_prod_section":    "المنتج الأكثر ربحاً مقابل الأكثر مبيعاً",
        "fin_best_seller":     "الأكثر مبيعاً",
        "fin_best_margin":     "الأكثر ربحاً",
        "fin_bubble_x":        "الوحدات المباعة",
        "fin_bubble_y":        "الهامش / الوحدة (MAD)",
        "fin_bubble_size":     "إجمالي CA (MAD)",
        "fin_table_cols":      ["المنتج","الوحدات المباعة","إجمالي CA","إجمالي التكلفة","الهامش الإجمالي","الهامش %","الهامش/وحدة"],
        "fin_no_price_info":   "أعد تهيئة الأسعار أعلاه لرؤية هذا التحليل.",
        "fin_monthly_section": "الإيرادات مقابل تكاليف المواد",
        "fin_rev_series":      "الإيرادات (CA)",
        "fin_mp_series":       "تكلفة المواد",
        "fin_margin_series":   "الهامش الإجمالي",
        "fin_profit_section":  "تقدير الربح الشهري",
        "fin_month_cols":      ["الشهر","CA (MAD)","المواد (MAD)","الهامش (MAD)","الهامش %"],
        "fin_total_sub":       "CA : {ca} · تكلفة المواد : {cout} · النسبة : {pct:.1f}%",
        "fin_total_label":     "إجمالي الهامش على كامل الفترة",
        "fin_no_config_info":  "أعد تهيئة الأسعار والتكاليف أعلاه لتفعيل هذا القسم.",
        "fin_footer":          "الهامش الإجمالي = CA − تكلفة المواد. لا يشمل الإيجار أو الرواتب أو التكاليف الثابتة.",
        # Conseils
        "conseils_hero_title": "مساعدة في اتخاذ القرار",
        "conseils_hero_sub":   "نصائح مخصصة بناءً على سجلك — تُحدَّث في الوقت الفعلي.",
        "conseils_pdj":        "⭐ منتج اليوم المقترح للإبراز",
        "conseils_pdj_badge":  "أبرزه اليوم",
        "conseils_pdj_score":  "النقاط {s}/100 | متوسط {d} : {qty:.0f} وحدة{margin}",
        "conseils_pdj_margin": " | الهامش : {m:.1f} درهم/وحدة",
        "conseils_pdj_empty":  "بيانات غير كافية لحساب منتج اليوم.",
        "conseils_prep":       "📋 جهّز أكثر — الأيام السبعة القادمة",
        "conseils_prep_normal":"يوم عادي — لا توجد ذروات محددة.",
        "conseils_promo":      "🏷️ اقتراحات أسعار ترويجية",
        "conseils_promo_thresh":"عتبة الانخفاض لتشغيل العرض (%)",
        "conseils_promo_help": "يُشار إلى المنتج إذا انخفضت مبيعاته خلال 7 أيام بنسبة X% مقارنة بـ30 يوماً.",
        "conseils_promo_note": "السعر الترويجي محسوب بتطبيق الخصم على سعر الوحدة المُهيأ في المالية.",
        "conseils_promo_config":"أعد تهيئة السعر",
        "conseils_promo_ok":   "✅ لا يوجد منتج في انخفاض ملحوظ خلال 7 أيام الأخيرة.",
        "conseils_dormant":    "😴 تنبيهات — منتجات لم تُباع مؤخراً",
        "conseils_dormant_slider":"التنبيه إذا كان غائباً منذ (أيام)",
        "conseils_dormant_help":"يُشار إلى المنتج إذا لم يظهر في مبيعاتك منذ X أيام.",
        "conseils_dormant_since":"آخر بيع : {d} — غائب منذ {n} أيام",
        "conseils_dormant_never":"آخر بيع : لم يُباع قط — غائب منذ +{n} أيام",
        "conseils_dormant_ok": "✅ جميع المنتجات بيعت خلال الأيام الأخيرة.",
        "conseils_seasonal":   "🌿 أفكار منتجات جديدة — الموسم والمناسبات",
        "conseils_seasonal_note":"اقتراحات بناءً على اتجاهات السوق المغربي والموسمية المحلية.",
        "conseils_seasonal_empty":"لا توجد اقتراحات موسمية لهذه الفترة.",
        "conseils_ram_current":"رمضان — الآن",
        "conseils_ram_soon":   "رمضان — خلال 30 يوماً",
        "conseils_month_now":  "{m} — منتجات موسمية",
        "conseils_month_next": "{m} — استعد للشهر القادم",
    },
    "en": {
        # Navigation
        "nav_dashboard":       "Dashboard",
        "nav_saisie":          "Sales Entry",
        "nav_overview":        "Overview",
        "nav_forecast":        "Forecast",
        "nav_analyse":         "Analytics",
        "nav_finances":        "Finances",
        "nav_recommendations": "Recommendations",
        "nav_products":        "Products",
        "nav_shopping":        "Shopping List",
        "nav_conseils":        "Insights",
        "nav_about":           "About",
        # Sidebar
        "sidebar_file":        "SALES FILE",
        "sidebar_demo":        "Boulangerie Atlas (demo) — upload your file above.",
        "sidebar_ramadan":     "Ramadan Mode",
        "sidebar_ramadan_help":"Enable Iftar alerts, special product indicators and Ramadan markers on charts.",
        "sidebar_lite":        "Lite Mode",
        "sidebar_lite_help":   "Disable AI forecasts (Prophet) to save bandwidth. Ideal on slow connections.",
        # Dashboard
        "dash_title":          "Dashboard",
        "sec_today":           "Today's Summary",
        "sec_perf":            "Performance Indicator",
        "sec_ranking":         "Product Ranking",
        "sec_week":            "This Week's Sales",
        "sec_history":         "History — Last 30 Days",
        "kpi_sold":            "Units Sold",
        "kpi_yesterday":       "vs yesterday",
        "kpi_avg":             "vs average",
        "kpi_best":            "Best Product Today",
        "kpi_avg_day":         "Avg / open day",
        "kpi_best_day_ever":   "All-time record",
        "kpi_open_days":       "Open days",
        "kpi_yesterday_sub":   "yesterday: {qty} {u}",
        "kpi_avg_sub":         "avg: {avg:.0f} {u}/day",
        "perf_good":           "Great day",
        "perf_avg":            "Average day",
        "perf_bad":            "Slow day",
        "perf_good_sub":       "Well above average",
        "perf_avg_sub":        "Within normal range",
        "perf_bad_sub":        "Below average",
        "perf_vs":             "of average",
        "rank_pos":            "#",
        "rank_product":        "Product",
        "rank_total":          "Total sold",
        "rank_share":          "Share",
        "rank_trend":          "Trend",
        "rank_caption":        "↗ rising · → stable · ↘ declining (last 30 days vs previous 30 days)",
        "hist_date":           "Date",
        "hist_day":            "Day",
        "hist_qty":            "Quantity",
        "hist_vs_avg":         "vs avg",
        "hist_perf":           "Rating",
        "days":                ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        "units":               "units",
        "lang_label":          "Language",
        # Saisie
        "saisie_title":        "Sales Entry",
        "saisie_sub":          "Enter today's sales without editing your CSV. Download the full file afterwards to re-import it.",
        "saisie_nofile":       "No file loaded — you can enter sales manually. Download the generated CSV and import it as your data file.",
        "saisie_new_prod":     "Add a product to the list",
        "saisie_add_btn":      "Add",
        "saisie_date_label":   "Sales date",
        "saisie_save_btn":     "Save sales",
        "saisie_save_btn2":    "Save",
        "saisie_sec_entry":    "Today's sales",
        "saisie_history":      "Manual entry history",
        "saisie_export":       "Export / Sync",
        "saisie_sync_tip":     "💡 Your entries are kept during this session. Download the full file below to keep them, then import it at next launch.",
        "saisie_download":     "Download full file (CSV + entries)",
        "saisie_clear":        "Clear all",
        "saisie_empty":        "No manual entries yet. Use the form above.",
        "saisie_success":      "{n} product(s) saved for {d}.",
        "saisie_warn":         "Enter at least one quantity > 0.",
        "saisie_kpi_days":     "Days entered",
        "saisie_kpi_days_sub": "via manual entry",
        "saisie_kpi_total":    "Total entered",
        "saisie_kpi_total_sub":"units total",
        "saisie_already_csv":  "Already in your CSV for this date.",
        # Forecast
        "forecast_title":      "Sales Forecast",
        "forecast_sub":        "AI-powered forecasts with Moroccan holiday awareness.",
        "forecast_lite_banner":"⚡ Lite mode on — AI forecasts disabled. Turn it off in the sidebar to see Prophet forecasts.",
        "forecast_lite_section":"Recent trend (last 30 days)",
        "forecast_horizon":    "How many days ahead?",
        "forecast_horizon_fmt":"{n} days",
        "forecast_acc_good":   "Very Good",
        "forecast_acc_ok":     "Good",
        "forecast_acc_fair":   "Fair",
        "forecast_acc_na":     "N/A",
        "forecast_acc_sub":    "avg error: {mae} units/day",
        "forecast_acc_label":  "Forecast accuracy",
        "forecast_acc_nodata": "Not enough data",
        "forecast_spinner":    "Loading your forecast...",
        "forecast_chart_sec":  "Sales History + {n}-Day Forecast",
        "forecast_history":    "Past sales",
        "forecast_pred":       "AI prediction",
        "forecast_range":      "Expected range",
        "forecast_fc_start":   "Forecast start ({d})",
        "forecast_peaks":      "Alerts — Upcoming peaks",
        "forecast_ram_sec":    "Ramadan Mode — Affected Days",
        "forecast_ram_msg":    "**{n} days** in the forecast fall during Ramadan. Evening sales (Iftar) may represent 60–80% of daily revenue. Prepare more **Harira, Chebakia, Briouates** and sweet drinks.",
        "forecast_weekly":     "Weekly summary",
        "forecast_monthly":    "Monthly summary",
        "forecast_weekly_cols":["Week","Days","Total forecast","Avg/day","Min forecast","Max forecast"],
        "forecast_monthly_cols":["Month","Total forecast","Avg/day","Min forecast","Max forecast"],
        # Analyse
        "analyse_title":       "Advanced Analytics",
        "analyse_sub":         "Week comparison, per-product forecasts, Moroccan holiday calendar and best selling hour.",
        "analyse_spinner":     "Computing forecasts...",
        "analyse_week":        "This week vs last week",
        "analyse_cur_week":    "This week",
        "analyse_cur_week_sub":"Last 7 days",
        "analyse_prev_week":   "Last week",
        "analyse_prev_week_sub":"Previous 7 days",
        "analyse_delta":       "Change",
        "analyse_units_sold":  "Units sold",
        "analyse_hour":        "Best selling hour",
        "analyse_best_hour":   "Best hour",
        "analyse_peak_avg":    "Peak avg: {qty:.1f} units",
        "analyse_no_hour":     "To enable this, add an **`heure`** column (0–23) to your CSV. Example: `2024-01-15, Pizza Margherita, 12, 8`",
        "analyse_prod":        "Per-product forecast",
        "analyse_prod_pick":   "Choose a product",
        "analyse_horizon":     "Horizon",
        "analyse_horizon_fmt": "{n} days",
        "analyse_prod_lite":   "⚡ Lite mode — per-product forecasts disabled.",
        "analyse_prod_empty":  "No products found in your data.",
        "analyse_prod_nodata": "Not enough data for **{p}** ({n} days — minimum 30).",
        "analyse_prod_spinner":"Forecasting {p}...",
        "analyse_prod_hist":   "Historical avg",
        "analyse_prod_next7":  "Forecast (next 7 days)",
        "analyse_prod_units":  "units/day (open days)",
        "analyse_prod_total":  "units total",
        "analyse_interval":    "Expected range",
        "analyse_pred_label":  "AI Forecast",
        "analyse_hist_label":  "History",
        "analyse_fc_start":    "Forecast start",
        "analyse_calendar":    "Moroccan Holiday Calendar",
        "analyse_cal_nat":     "National holiday",
        "analyse_cal_isl":     "Islamic holiday",
        "analyse_cal_ram":     "Ramadan",
        "analyse_ram_msg":     "**Ramadan mode on** — During Ramadan, sales concentrate in the **7 PM – 11 PM** window (Iftar & Tarawih). Prepare more: Harira, Chebakia, Sellou, Briouates, Shabakiya, fresh juices and dates. Reduce morning prep by 30–40%.",
        "analyse_trend_up":    "rising",
        "analyse_trend_stable":"stable",
        "analyse_trend_down":  "declining",
        "analyse_trend_caption":"↗ {up} · → {stable} · ↘ {down} (last 30 days vs previous 30 days)",
        # Finances
        "fin_title":           "Finances & Profitability",
        "fin_sub":             "Revenue, margins, targets and monthly profit estimate.",
        "fin_config":          "Configure selling prices and unit costs",
        "fin_config_caption":  "Enter the **selling price** (what you collect) and **unit cost** (ingredients + packaging) for each product. All values in **MAD**.",
        "fin_auto_price":      "`prix_unitaire` column detected — prices pre-filled.",
        "fin_col_prod":        "Product",
        "fin_col_price":       "Selling price (MAD)",
        "fin_col_cost":        "Unit cost (MAD)",
        "fin_col_price_help":  "Price at which you sell one portion/unit",
        "fin_col_cost_help":   "Total ingredient + packaging cost per unit produced",
        "fin_warn_no_price":   "No selling prices set — financial metrics will show 0 MAD. Enter prices in the table above to unlock all analyses.",
        "fin_ca_section":      "Revenue",
        "fin_kpi_day":         "Today's revenue",
        "fin_kpi_week":        "This week's revenue",
        "fin_kpi_month":       "This month's revenue",
        "fin_kpi_margin":      "Monthly gross margin",
        "fin_margin_sub":      "margin: {m}",
        "fin_margin_pct":      "{pct:.1f}% of revenue",
        "fin_target_section":  "Daily revenue target",
        "fin_target_input":    "Target (MAD/day)",
        "fin_target_above":    "{pct:.0f}% achieved · {ca} / target {target}",
        "fin_target_days":     "Days that hit the target in history: **{n} / {total}**",
        "fin_target_empty":    "Enter a target on the left to track your progress.",
        "fin_ca_series":       "Daily revenue",
        "fin_cost_series":     "Raw material cost",
        "fin_target_line":     "Target {t}/day",
        "fin_prod_section":    "Most profitable vs best-selling product",
        "fin_best_seller":     "Best seller",
        "fin_best_margin":     "Most profitable",
        "fin_bubble_x":        "Units sold",
        "fin_bubble_y":        "Margin / unit (MAD)",
        "fin_bubble_size":     "Total revenue (MAD)",
        "fin_table_cols":      ["Product","Units sold","Total revenue","Total cost","Gross margin","Margin %","Margin/unit"],
        "fin_no_price_info":   "Set up prices above to see this analysis.",
        "fin_monthly_section": "Revenue vs raw material costs",
        "fin_rev_series":      "Revenue",
        "fin_mp_series":       "Raw material cost",
        "fin_margin_series":   "Gross margin",
        "fin_profit_section":  "Monthly profit estimate",
        "fin_month_cols":      ["Month","Revenue (MAD)","Materials (MAD)","Margin (MAD)","Margin %"],
        "fin_total_sub":       "Revenue: {ca} · Materials: {cout} · Rate: {pct:.1f}%",
        "fin_total_label":     "Total gross margin over the entire period",
        "fin_no_config_info":  "Set up prices and costs above to enable this section.",
        "fin_footer":          "Gross margin = Revenue − Raw material costs. Does not include rent, payroll or fixed costs. A gross margin ≥ 60% is generally healthy for food businesses.",
        # Conseils
        "conseils_hero_title": "Decision Support",
        "conseils_hero_sub":   "Personalised insights based on your history — updated in real time.",
        "conseils_pdj":        "⭐ Featured Product of the Day",
        "conseils_pdj_badge":  "Feature today",
        "conseils_pdj_score":  "Score {s}/100 | {d} avg: {qty:.0f} units{margin}",
        "conseils_pdj_margin": " | Margin: {m:.1f} MAD/unit",
        "conseils_pdj_empty":  "Not enough data to compute the product of the day.",
        "conseils_prep":       "📋 Prepare More — Next 7 Days",
        "conseils_prep_normal":"Normal day — no surge identified.",
        "conseils_promo":      "🏷️ Promotional Price Suggestions",
        "conseils_promo_thresh":"Decline threshold to trigger a promo (%)",
        "conseils_promo_help": "A product is flagged if its last-7-day sales dropped by X% vs the previous 30 days.",
        "conseils_promo_note": "Promo price calculated by applying the discount to the unit price set in Finances.",
        "conseils_promo_config":"set price in Finances",
        "conseils_promo_ok":   "✅ No product in significant decline over the last 7 days.",
        "conseils_dormant":    "😴 Alerts — Products Not Sold Recently",
        "conseils_dormant_slider":"Flag if absent for (days)",
        "conseils_dormant_help":"A product is flagged if it hasn't appeared in your sales for X days.",
        "conseils_dormant_since":"Last sale: {d} — absent for {n} days",
        "conseils_dormant_never":"Last sale: never — absent for +{n} days",
        "conseils_dormant_ok": "✅ All products have been sold in recent days.",
        "conseils_seasonal":   "🌿 New Product Ideas — Season & Events",
        "conseils_seasonal_note":"Suggestions based on Moroccan market trends and local seasonality. Adapt them to your clientele.",
        "conseils_seasonal_empty":"No seasonal suggestions available for this period.",
        "conseils_ram_current":"Ramadan — now",
        "conseils_ram_soon":   "Ramadan — in 30 days",
        "conseils_month_now":  "{m} — seasonal products",
        "conseils_month_next": "{m} — prepare for next month",
    },
}


def T(key: str) -> str:
    """Return translated string for the active UI language."""
    lang = st.session_state.get("lang", "fr")
    val  = TRANS.get(lang, TRANS["fr"]).get(key) or TRANS["fr"].get(key, key)
    return val if isinstance(val, str) else str(val)


def Tlist(key: str) -> list:
    """Return translated list for the active UI language."""
    lang = st.session_state.get("lang", "fr")
    val  = TRANS.get(lang, TRANS["fr"]).get(key) or TRANS["fr"].get(key, [])
    return val if isinstance(val, list) else []


def Tdays() -> list[str]:
    lang = st.session_state.get("lang", "fr")
    return TRANS.get(lang, TRANS["fr"]).get("days", TRANS["fr"]["days"])


def day_label(en_name: str) -> str:
    days = Tdays()
    return days[EN_DAYS_IDX.get(en_name, 0)]


# ── Recipes: ingredient quantities per unit sold ───────────────────────────
# Structure: {product: {ingredient: (qty_per_unit, unit)}}
RECIPES: dict[str, dict[str, tuple[float, str]]] = {
    "Pain Traditionnel": {
        "Farine de blé":     (250, "g"),
        "Eau":               (150, "ml"),
        "Levure fraîche":    (5,   "g"),
        "Sel":               (4,   "g"),
        "Huile d'olive":     (8,   "ml"),
        "Semoule fine":      (30,  "g"),
    },
    "Msemen": {
        "Farine de blé":     (100, "g"),
        "Semoule fine":      (50,  "g"),
        "Eau tiède":         (70,  "ml"),
        "Sel":               (2,   "g"),
        "Huile végétale":    (15,  "ml"),
        "Beurre":            (10,  "g"),
    },
    "Harcha": {
        "Semoule fine":      (150, "g"),
        "Beurre fondu":      (40,  "g"),
        "Lait":              (60,  "ml"),
        "Sel":               (2,   "g"),
        "Levure chimique":   (3,   "g"),
    },
    "Beghrir": {
        "Semoule fine":      (80,  "g"),
        "Farine de blé":     (40,  "g"),
        "Eau tiède":         (200, "ml"),
        "Levure fraîche":    (5,   "g"),
        "Sel":               (2,   "g"),
        "Sucre":             (5,   "g"),
    },
    "Croissant": {
        "Farine T45":        (100, "g"),
        "Beurre de tourage": (55,  "g"),
        "Lait":              (35,  "ml"),
        "Sucre":             (12,  "g"),
        "Levure fraîche":    (3,   "g"),
        "Sel":               (2,   "g"),
        "Oeuf":              (0.2, "piece"),
    },
    "Pain au Chocolat": {
        "Farine T45":        (100, "g"),
        "Beurre de tourage": (55,  "g"),
        "Lait":              (35,  "ml"),
        "Sucre":             (12,  "g"),
        "Levure fraîche":    (3,   "g"),
        "Sel":               (2,   "g"),
        "Bâton chocolat":    (2,   "piece"),
    },
    "Brioche": {
        "Farine T45":        (150, "g"),
        "Beurre":            (60,  "g"),
        "Oeuf":              (1.5, "piece"),
        "Sucre":             (25,  "g"),
        "Levure fraîche":    (5,   "g"),
        "Lait":              (30,  "ml"),
        "Sel":               (3,   "g"),
    },
    "Chausson aux Pommes": {
        "Pâte feuilletée":   (120, "g"),
        "Pomme":             (80,  "g"),
        "Sucre":             (15,  "g"),
        "Cannelle":          (1,   "g"),
        "Beurre":            (8,   "g"),
        "Oeuf":              (0.3, "piece"),
    },
    "Mille-feuille": {
        "Pâte feuilletée":   (150, "g"),
        "Crème pâtissière":  (120, "g"),
        "Sucre glace":       (20,  "g"),
        "Fondant blanc":     (30,  "g"),
    },
    "Tarte aux Fraises": {
        "Pâte sablée":       (100, "g"),
        "Crème pâtissière":  (80,  "g"),
        "Fraises":           (120, "g"),
        "Nappage":           (20,  "g"),
        "Beurre":            (10,  "g"),
    },
    "Eclair au Chocolat": {
        "Pâte à choux":      (80,  "g"),
        "Crème pâtissière":  (90,  "g"),
        "Chocolat noir":     (30,  "g"),
        "Beurre":            (8,   "g"),
        "Oeuf":              (0.5, "piece"),
    },
    "Sellou": {
        "Farine de blé":     (200, "g"),
        "Sésame":            (100, "g"),
        "Amandes":           (80,  "g"),
        "Miel":              (60,  "g"),
        "Beurre fondu":      (80,  "g"),
        "Anis":              (5,   "g"),
        "Sucre glace":       (40,  "g"),
    },
}

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="FoodCast — Boulangerie Atlas",
    page_icon="🥐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Injection ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]        { font-family: 'Inter', sans-serif !important; }
.stApp                            { background-color: #f0f4f8; }
.block-container                  { padding-top: 1.4rem !important;
                                    padding-bottom: 2.5rem !important;
                                    max-width: 1240px; }

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"]         { background: linear-gradient(170deg, #0f2541 0%, #1e3a5f 60%, #155e75 100%) !important; }
[data-testid="stSidebar"] *       { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio > label           { font-weight: 600; font-size: 0.78rem;
                                    text-transform: uppercase; letter-spacing: .07em; color: #94a3b8 !important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label  { font-size: 0.95rem !important;
                                    text-transform: none !important; letter-spacing: normal !important;
                                    font-weight: 500 !important; color: #e2e8f0 !important; }
[data-testid="stSidebar"] hr      { border-color: rgba(255,255,255,.12) !important; }
[data-testid="stSidebar"] .stFileUploader label      { color: #94a3b8 !important;
                                    font-size: 0.78rem !important; text-transform: uppercase;
                                    letter-spacing: .07em; font-weight: 600 !important; }

/* ── KPI card ─────────────────────────────────────────────────────────────── */
.kpi-card   { background: #fff; border-radius: 14px; padding: 20px 22px;
              box-shadow: 0 2px 14px rgba(0,0,0,.07); border-top: 3px solid #14b8a6;
              transition: transform .18s ease, box-shadow .18s ease; }
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,.11); }
.kpi-label  { font-size: .72rem; font-weight: 700; color: #64748b;
              text-transform: uppercase; letter-spacing: .07em; margin-bottom: 5px; }
.kpi-value  { font-size: 1.85rem; font-weight: 800; color: #1e3a5f; line-height: 1.1; }
.kpi-sub    { font-size: .77rem; color: #94a3b8; margin-top: 5px; }
.kpi-icon   { font-size: 1.5rem; float: right; opacity: .25; margin-top: -32px; }

/* ── Section header ───────────────────────────────────────────────────────── */
.sec-head   { font-size: 1.05rem; font-weight: 700; color: #1e3a5f;
              margin: 1.6rem 0 .55rem; padding-bottom: 7px;
              border-bottom: 2px solid #e2e8f0; }

/* ── Health badges ────────────────────────────────────────────────────────── */
.badge      { display:inline-block; padding: 2px 11px; border-radius: 20px;
              font-size: .72rem; font-weight: 700; }
.badge-ok   { background:#dcfce7; color:#166534; }
.badge-warn { background:#fef9c3; color:#854d0e; }
.badge-err  { background:#fee2e2; color:#991b1b; }
.health-row { display:flex; align-items:center; gap:10px; padding: 7px 0;
              border-bottom: 1px solid #f1f5f9; font-size:.85rem; color:#475569; }
.health-row:last-child { border-bottom: none; }
.health-label { min-width: 175px; font-weight: 500; }

/* ── Accuracy badge ───────────────────────────────────────────────────────── */
.acc-box    { background:#fff; border-radius:12px; padding:14px 20px;
              box-shadow:0 2px 12px rgba(0,0,0,.07); text-align:center;
              border-top: 3px solid #14b8a6; }
.acc-val    { font-size:1.55rem; font-weight:800; color:#14b8a6; display:block; }
.acc-lbl    { font-size:.7rem; font-weight:700; color:#94a3b8;
              text-transform:uppercase; letter-spacing:.07em; }

/* ── Summary box ──────────────────────────────────────────────────────────── */
.sum-box    { background: linear-gradient(135deg, #1e3a5f 0%, #0e7490 100%);
              border-radius:16px; padding:26px 32px; color:#fff; text-align:center; margin:1rem 0; }
.sum-big    { font-size:3rem; font-weight:800; display:block; line-height:1.1; }
.sum-sub    { font-size:.92rem; opacity:.82; margin-top:4px; }

/* ── Step cards (About page) ──────────────────────────────────────────────── */
.step-card  { background:#fff; border-radius:14px; padding:26px 22px;
              box-shadow:0 2px 14px rgba(0,0,0,.07); text-align:center; height:100%; }
.step-num   { width:46px; height:46px; border-radius:50%; color:#fff; font-weight:800;
              font-size:1.1rem; display:inline-flex; align-items:center; justify-content:center;
              margin-bottom:14px;
              background: linear-gradient(135deg, #1e3a5f, #14b8a6); }
.step-title { font-size:1.0rem; font-weight:700; color:#1e3a5f; margin-bottom:7px; }
.step-desc  { font-size:.84rem; color:#64748b; line-height:1.55; }

/* ── Hero banner ──────────────────────────────────────────────────────────── */
.hero       { background: linear-gradient(135deg, #0f2541 0%, #14b8a6 100%);
              border-radius:18px; padding:48px 40px; color:#fff; margin-bottom:1.8rem; }
.hero h1    { font-size:2.4rem; font-weight:800; margin:0 0 8px; }
.hero .tag  { font-size:1.05rem; opacity:.85; letter-spacing:.12em; text-transform:uppercase; }
.hero .desc { font-size:1.0rem; opacity:.9; margin-top:16px; max-width:600px; line-height:1.6; }

/* ── Finance progress bar ─────────────────────────────────────────────────── */
.prog-wrap  { background:#e2e8f0; border-radius:20px; height:22px; overflow:hidden;
              margin:6px 0 2px; }
.prog-bar   { height:100%; border-radius:20px; transition:width .4s ease;
              background: linear-gradient(90deg,#14b8a6,#0e7490); }
.prog-bar-warn { background: linear-gradient(90deg,#f59e0b,#d97706); }
.prog-bar-ok   { background: linear-gradient(90deg,#22c55e,#16a34a); }
.prog-label { font-size:.72rem; color:#64748b; font-weight:600; }

/* ── Finance card (green accent) ─────────────────────────────────────────── */
.fin-card   { background:#fff; border-radius:14px; padding:18px 20px;
              box-shadow:0 2px 14px rgba(0,0,0,.07); border-top:3px solid #22c55e; }
.fin-label  { font-size:.72rem; font-weight:700; color:#64748b;
              text-transform:uppercase; letter-spacing:.07em; margin-bottom:4px; }
.fin-value  { font-size:1.7rem; font-weight:800; color:#166534; line-height:1.1; }
.fin-sub    { font-size:.75rem; color:#94a3b8; margin-top:4px; }

/* ── Info pill ────────────────────────────────────────────────────────────── */
.info-pill  { display:inline-block; background:#e0f2fe; color:#0369a1;
              border-radius:20px; padding:3px 13px; font-size:.78rem; font-weight:600;
              margin:2px; }

/* ── RTL / Arabic support ─────────────────────────────────────────────────── */
.rtl        { direction:rtl; text-align:right; }
.ar-font    { font-family:'Segoe UI','Arial','Tahoma',sans-serif !important; }
.rtl .kpi-label, .rtl .kpi-value, .rtl .kpi-sub,
.rtl .fin-label, .rtl .fin-value, .rtl .fin-sub { text-align:right; }

/* ── Performance banner ──────────────────────────────────────────────────── */
.perf-banner{ border-radius:18px; padding:28px 36px; margin:0.6rem 0 1.2rem;
              display:flex; align-items:center; gap:24px; }
.perf-icon  { font-size:3.2rem; line-height:1; flex-shrink:0; }
.perf-label { font-size:2rem; font-weight:800; line-height:1.1; }
.perf-sub   { font-size:.92rem; opacity:.82; margin-top:5px; }
.perf-pct   { font-size:1.1rem; font-weight:700; margin-top:8px; }

/* ── Product ranking ─────────────────────────────────────────────────────── */
.rank-row   { display:flex; align-items:center; gap:14px; padding:10px 16px;
              background:#fff; border-radius:12px; margin-bottom:8px;
              box-shadow:0 1px 6px rgba(0,0,0,.06); }
.rank-medal { font-size:1.5rem; width:32px; text-align:center; flex-shrink:0; }
.rank-name  { flex:1; font-weight:600; color:#1e3a5f; font-size:.95rem; }
.rank-qty   { font-weight:800; color:#14b8a6; font-size:1.05rem; min-width:60px; text-align:right; }
.rank-share { font-size:.78rem; color:#94a3b8; min-width:40px; text-align:right; }
.rank-trend { font-size:1.1rem; min-width:28px; text-align:center; }

/* ── Day badge ────────────────────────────────────────────────────────────── */
.day-good   { color:#166534; font-weight:700; }
.day-avg    { color:#854d0e; font-weight:700; }
.day-bad    { color:#991b1b; font-weight:700; }

/* ── Mobile responsive (≤ 768 px) ────────────────────────────────────────── */
@media (max-width: 768px) {
    .block-container    { padding: 0.4rem 0.7rem !important; }
    .kpi-card           { padding: 12px 12px; }
    .kpi-value          { font-size: 1.3rem; }
    .kpi-icon           { display: none; }
    .fin-card           { padding: 12px 12px; }
    .fin-value          { font-size: 1.3rem; }
    .perf-banner        { flex-direction: column; text-align: center; padding: 16px 14px; gap: 10px; }
    .perf-label         { font-size: 1.4rem; }
    .perf-pct           { font-size: .95rem; }
    .sum-box            { padding: 18px 16px; }
    .sum-big            { font-size: 2rem; }
    .sum-sub            { font-size: .78rem; }
    .sec-head           { font-size: .92rem; }
    .rank-row           { padding: 8px 10px; gap: 8px; }
    .rank-qty           { font-size: .9rem; }
    h1                  { font-size: 1.5rem !important; }
    /* Touch-friendly number inputs */
    input[type="number"]{ font-size: 1.1rem !important; min-height: 42px; }
}

/* ── Connection status badge (injected by JS) ─────────────────────────────── */
#fc-conn-badge {
    position: fixed; bottom: 14px; right: 14px; z-index: 9999;
    padding: 5px 14px; border-radius: 20px; font-size: 12px;
    font-weight: 700; font-family: sans-serif;
    box-shadow: 0 2px 10px rgba(0,0,0,.18);
    transition: background .4s, color .4s;
    cursor: default; user-select: none;
}

/* ── Saisie rapide ────────────────────────────────────────────────────────── */
.entry-card { background:#fff; border-radius:14px; padding:18px 20px;
              box-shadow:0 2px 12px rgba(0,0,0,.07);
              border-left:4px solid #14b8a6; margin-bottom:10px; }
.entry-prod { font-weight:700; color:#1e3a5f; font-size:.97rem; }
.entry-qty  { color:#14b8a6; font-weight:800; font-size:1.1rem; }
.entry-date { color:#94a3b8; font-size:.78rem; }

/* ── Lite mode banner ────────────────────────────────────────────────────── */
.lite-banner{ background:#fef3c7; border:1px solid #fcd34d; border-radius:10px;
              padding:10px 16px; color:#92400e; font-size:.85rem; font-weight:600;
              margin-bottom:.8rem; }

/* ── Conseils cards ──────────────────────────────────────────────────────── */
.conseil-hero { background:linear-gradient(135deg,#1e3a5f 0%,#0e7490 100%);
               border-radius:18px; padding:28px 32px; color:#fff;
               margin-bottom:1.2rem; }
.conseil-hero-title { font-size:1.8rem; font-weight:800; margin-bottom:4px; }
.conseil-hero-sub   { font-size:.9rem; opacity:.85; }
.conseil-section { background:#fff; border-radius:16px; padding:24px 26px;
                   box-shadow:0 2px 14px rgba(0,0,0,.07);
                   border-left:4px solid #14b8a6; margin-bottom:1.2rem; }
.conseil-section-title { font-size:1.05rem; font-weight:700; color:#1e3a5f;
                          margin-bottom:12px; }
.produit-du-jour { text-align:center; padding:24px 16px; }
.pdj-name   { font-size:2rem; font-weight:800; color:#1e3a5f; margin:10px 0 4px; }
.pdj-score  { font-size:.82rem; color:#94a3b8; font-weight:600;
              text-transform:uppercase; letter-spacing:.05em; }
.pdj-badge  { display:inline-block; background:#dcfce7; color:#166534;
              border-radius:20px; padding:4px 16px; font-size:.85rem;
              font-weight:700; margin-top:8px; }
.alert-row  { display:flex; align-items:center; gap:12px; padding:10px 14px;
              background:#fff7ed; border:1px solid #fed7aa; border-radius:10px;
              margin-bottom:8px; }
.alert-icon { font-size:1.4rem; flex-shrink:0; }
.alert-text { flex:1; }
.alert-prod { font-weight:700; color:#9a3412; font-size:.95rem; }
.alert-detail { font-size:.8rem; color:#c2410c; }
.promo-row  { display:flex; align-items:center; gap:14px; padding:10px 14px;
              background:#fefce8; border:1px solid #fde68a; border-radius:10px;
              margin-bottom:8px; }
.promo-prod { font-weight:700; color:#854d0e; flex:1; font-size:.95rem; }
.promo-price { font-size:1.1rem; font-weight:800; color:#d97706; min-width:90px; text-align:right; }
.promo-pct  { font-size:.78rem; color:#92400e; min-width:55px; text-align:right; }
.prep-row   { display:flex; align-items:center; gap:12px; padding:10px 14px;
              background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px;
              margin-bottom:8px; }
.prep-prod  { font-weight:700; color:#166534; flex:1; font-size:.95rem; }
.prep-qty   { font-size:1.1rem; font-weight:800; color:#16a34a; min-width:70px; text-align:right; }
.prep-ratio { font-size:.8rem; color:#15803d; min-width:65px; text-align:right; }
.season-card { background:#fff; border:1px solid #e2e8f0; border-radius:12px;
               padding:16px 18px; }
.season-card-title { font-weight:700; color:#1e3a5f; margin-bottom:6px; font-size:.95rem; }
.season-idea { display:flex; align-items:baseline; gap:8px; padding:4px 0; }
.season-idea-name { font-weight:600; color:#0e7490; font-size:.9rem; }
.season-idea-why  { font-size:.78rem; color:#64748b; }

/* ── Hide Streamlit chrome ────────────────────────────────────────────────── */
#MainMenu   { visibility:hidden; }
footer      { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═════════════════════════════════════════════════════════════════════════════

def _kpi(label: str, value: str, sub: str = "", icon: str = "") -> str:
    icon_html = f'<div class="kpi-icon">{icon}</div>' if icon else ""
    sub_html  = f'<div class="kpi-sub">{sub}</div>'  if sub  else ""
    return (f'<div class="kpi-card">{icon_html}'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>{sub_html}</div>')


def _sec(title: str) -> None:
    st.markdown(f'<div class="sec-head">{title}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_and_validate(file_bytes: bytes | None, file_ext: str = "csv") -> tuple[pd.DataFrame | None, list[str]]:
    """Load CSV or Excel, run validation checks. Returns (df_or_None, error_list)."""
    errors: list[str] = []
    try:
        if file_bytes is None:
            df = pd.read_csv(DEFAULT_CSV, parse_dates=["date"])
        elif file_ext in ("xlsx", "xls"):
            df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl" if file_ext == "xlsx" else None)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df = pd.read_csv(io.BytesIO(file_bytes), parse_dates=["date"])

        missing_cols = REQUIRED_COLS - set(df.columns)
        if missing_cols:
            errors.append(f"Your file is missing these required columns: **{', '.join(sorted(missing_cols))}**. "
                          f"The file must have exactly these column names: `date`, `produit`, `quantite_vendue`. "
                          f"Download the sample file on the About page to see the correct format.")
            return None, errors

        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            errors.append("The `date` column couldn't be read. Please use the format `YYYY-MM-DD` (e.g. 2024-01-15).")
            return None, errors

        if not pd.api.types.is_numeric_dtype(df["quantite_vendue"]):
            errors.append("The `quantite_vendue` column must contain numbers only (e.g. 28, not 'twenty-eight').")
            return None, errors

        n_days = df["date"].nunique()
        if n_days < 30:
            errors.append(f"Your file only has **{n_days} days** of data. "
                          "At least 30 days are needed to make reliable predictions.")
            return None, errors

        return df, errors

    except Exception as exc:
        errors.append(f"Could not read your file. Make sure it is a valid CSV file. Details: {exc}")
        return None, errors


@st.cache_data(show_spinner=False)
def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("date")["quantite_vendue"]
        .sum().reset_index()
        .rename(columns={"date": "ds", "quantite_vendue": "y"})
        .sort_values("ds").reset_index(drop=True)
    )


def get_moroccan_holidays(years: list[int]) -> pd.DataFrame:
    """Build a Prophet-compatible holiday DataFrame for Morocco."""
    rows: list[dict] = []
    for year in years:
        for month, day, name in MAROC_FIXED_HOLIDAYS:
            try:
                rows.append({"ds": pd.Timestamp(year, month, day), "holiday": name})
            except ValueError:
                pass
        for date_str, name in ISLAMIC_HOLIDAYS.get(year, []):
            rows.append({"ds": pd.Timestamp(date_str), "holiday": name})
    return pd.DataFrame(rows).drop_duplicates("ds").reset_index(drop=True)


def _get_holiday_name(date: pd.Timestamp) -> str | None:
    """Return Moroccan holiday name for a given date, or None."""
    for month, day, name in MAROC_FIXED_HOLIDAYS:
        if date.month == month and date.day == day:
            return name
    for date_str, name in ISLAMIC_HOLIDAYS.get(date.year, []):
        if pd.Timestamp(date_str).date() == date.date():
            return name
    return None


def is_ramadan_period(date: pd.Timestamp) -> bool:
    """Return True if the date falls during Ramadan."""
    for start_str, end_str in RAMADAN_DATES.values():
        if pd.Timestamp(start_str) <= date <= pd.Timestamp(end_str):
            return True
    return False


def get_upcoming_peaks(pred_future: pd.DataFrame, avg: float, n_days: int = 7) -> list[dict]:
    """Return upcoming days that warrant a special alert."""
    peaks = []
    for _, row in pred_future.head(n_days).iterrows():
        reasons: list[str] = []
        day_name = row["ds"].day_name()
        if day_name in WEEKEND_DAYS:
            reasons.append("week-end")
        ratio = row["yhat"] / max(avg, 1)
        if ratio > PEAK_THRESH:
            reasons.append(f"+{(ratio - 1)*100:.0f}% vs moyenne")
        holiday = _get_holiday_name(row["ds"])
        if holiday:
            reasons.append(f"Fête : {holiday}")
        if is_ramadan_period(row["ds"]):
            reasons.append("Ramadan")
        if reasons:
            peaks.append({
                "date":    row["ds"],
                "yhat":    int(max(row["yhat"], 0).round()),
                "reasons": reasons,
            })
    return peaks


@st.cache_resource(show_spinner=False)
def get_model(data_hash: str, _daily: pd.DataFrame) -> Prophet:
    """Train Prophet on the full daily series with Moroccan holidays."""
    years = list(range(_daily["ds"].dt.year.min(), _daily["ds"].dt.year.max() + 3))
    holidays_df = get_moroccan_holidays(years)
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.95,
        holidays=holidays_df,
    )
    m.fit(_daily)
    return m


@st.cache_data(show_spinner=False)
def aggregate_by_product(df: pd.DataFrame, product: str) -> pd.DataFrame:
    """Daily series for a single product."""
    return (
        df[df["produit"] == product]
        .groupby("date")["quantite_vendue"]
        .sum().reset_index()
        .rename(columns={"date": "ds", "quantite_vendue": "y"})
        .sort_values("ds").reset_index(drop=True)
    )


@st.cache_resource(show_spinner=False)
def get_product_model(data_hash: str, product: str, _prod_daily: pd.DataFrame) -> Prophet:
    years = list(range(_prod_daily["ds"].dt.year.min(), _prod_daily["ds"].dt.year.max() + 3))
    holidays_df = get_moroccan_holidays(years)
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.90,
        holidays=holidays_df,
    )
    m.fit(_prod_daily)
    return m


@st.cache_data(show_spinner=False)
def run_product_forecast(data_hash: str, product: str, horizon: int,
                         _model: Prophet, _prod_daily: pd.DataFrame) -> pd.DataFrame:
    future = pd.DataFrame({
        "ds": pd.date_range(_prod_daily["ds"].min(),
                            _prod_daily["ds"].max() + pd.Timedelta(days=horizon), freq="D")
    })
    pred = _model.predict(future)
    return pred[["ds", "yhat", "yhat_lower", "yhat_upper"]]


@st.cache_data(show_spinner=False)
def run_forecast(data_hash: str, horizon: int, _model: Prophet,
                 _daily: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions from history start through history end + horizon days."""
    last_date = _daily["ds"].max()
    future = pd.DataFrame({
        "ds": pd.date_range(_daily["ds"].min(),
                            last_date + pd.Timedelta(days=horizon), freq="D")
    })
    pred = _model.predict(future)
    return pred[["ds", "yhat", "yhat_lower", "yhat_upper"]]


@st.cache_data(show_spinner=False)
def compute_mae(data_hash: str, _daily: pd.DataFrame) -> float | None:
    """Holdout MAE: train on [all - last 30], evaluate on last 30 days."""
    if len(_daily) < 61:
        return None
    cutoff  = len(_daily) - 30
    train   = _daily.iloc[:cutoff].copy()
    test    = _daily.iloc[cutoff:].copy()
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False, seasonality_mode="multiplicative")
    m.fit(train)
    pred = m.predict(test[["ds"]])
    return round(float(np.mean(np.abs(pred["yhat"].values - test["y"].values))), 1)


def check_data_health(df: pd.DataFrame) -> list[dict]:
    checks = []

    nulls = int(df.isnull().sum().sum())
    checks.append({"label": "Missing values",
                   "status": "ok" if nulls == 0 else "warn",
                   "detail": "None detected" if nulls == 0 else f"{nulls} null values"})

    neg = int((df["quantite_vendue"] < 0).sum())
    checks.append({"label": "Negative quantities",
                   "status": "ok" if neg == 0 else "warn",
                   "detail": "None" if neg == 0 else f"{neg} rows with qty < 0"})

    expected_days = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    gaps = len(expected_days) - df["date"].dt.date.nunique()
    checks.append({"label": "Date continuity",
                   "status": "ok" if gaps == 0 else "warn",
                   "detail": "No gaps" if gaps == 0 else f"{gaps} missing dates"})

    n_products = df["produit"].nunique()
    checks.append({"label": "Products detected",
                   "status": "ok",
                   "detail": f"{n_products} unique product(s): " +
                              ", ".join(sorted(df["produit"].unique()))})

    zero_days = int((df.groupby("date")["quantite_vendue"].sum() == 0).sum())
    checks.append({"label": "Zero-sales days (closed)",
                   "status": "ok",
                   "detail": f"{zero_days} day(s) with total sales = 0"})

    return checks


def make_sample_csv() -> bytes:
    rows = []
    products = ["Pizza Margherita", "Chicken Sandwich", "Burger", "Caesar Salad"]
    qtys     = [28, 22, 17, 13]
    for i in range(5):
        date_str = f"2024-01-0{i+1}"
        for p, q in zip(products, qtys):
            rows.append({"date": date_str, "produit": p, "quantite_vendue": q + i})
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def compute_stockout_date(current_stock: float,
                          daily_qtys: list[float],
                          dates: list) -> str:
    """
    Simulate day-by-day consumption. Return the first date stock hits <= 0,
    or 'OK (>N j)' if it lasts the full period.
    """
    stock = float(current_stock)
    for date, qty in zip(dates, daily_qtys):
        stock -= max(float(qty), 0)
        if stock <= 0:
            return pd.Timestamp(date).strftime("%Y-%m-%d")
    return f"OK (>{len(dates)} days)"


def _fin_kpi(label: str, value: str, sub: str = "", icon: str = "") -> str:
    icon_html = f'<div class="kpi-icon">{icon}</div>' if icon else ""
    sub_html  = f'<div class="fin-sub">{sub}</div>'  if sub  else ""
    return (f'<div class="fin-card">{icon_html}'
            f'<div class="fin-label">{label}</div>'
            f'<div class="fin-value">{value}</div>{sub_html}</div>')


def _progress_bar(pct: float, label: str) -> str:
    clamped = min(pct, 100)
    cls = "prog-bar-ok" if pct >= 100 else ("prog-bar-warn" if pct >= 70 else "prog-bar")
    return (
        f'<div class="prog-label">{label}</div>'
        f'<div class="prog-wrap"><div class="{cls}" style="width:{clamped:.1f}%"></div></div>'
    )


def build_fin_df(df: pd.DataFrame, price_map: dict, cost_map: dict) -> pd.DataFrame:
    """Attach per-row revenue, cost, and gross margin columns."""
    out = df.copy()
    out["prix_unit"]   = out["produit"].map(price_map).fillna(0.0)
    out["cout_unit"]   = out["produit"].map(cost_map).fillna(0.0)
    out["recette"]     = out["quantite_vendue"] * out["prix_unit"]
    out["cout_mp"]     = out["quantite_vendue"] * out["cout_unit"]
    out["marge_brute"] = out["recette"] - out["cout_mp"]
    return out


def _mad(v: float) -> str:
    """Format a float as Moroccan dirham string."""
    return f"{v:,.0f} MAD"


def _risk_level(row: pd.Series, avg: float) -> str:
    """
    Risk is primarily demand-driven (ratio to average) so it stays
    meaningful regardless of CI width, which varies with forecast horizon.
    """
    ratio = row["yhat"] / max(avg, 1)
    if ratio > PEAK_THRESH:           # >130 % of average
        return "High"
    if ratio > 1.10 or row["is_weekend"]:   # >110 % or weekend
        return "Medium"
    return "Low"


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 4px">
      <div style="font-size:1.6rem;font-weight:800;color:#fff;letter-spacing:-.5px;">
        🥐 FoodCast
      </div>
      <div style="font-size:.72rem;color:#14b8a6;font-weight:700;letter-spacing:.18em;
                  text-transform:uppercase;margin-top:2px;">
        Predict &middot; Decide &middot; Grow
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Language selector (drives T() for the rest of the sidebar + pages) ──
    if "lang" not in st.session_state:
        st.session_state["lang"] = "fr"

    _lang_opts = ["fr", "ar", "en"]
    _lang_labels = {"fr": "🇫🇷 Français", "ar": "🇲🇦 العربية", "en": "🇬🇧 English"}
    _lang_idx = _lang_opts.index(st.session_state["lang"]) if st.session_state["lang"] in _lang_opts else 0
    _lang_choice = st.radio(
        "🌐",
        _lang_opts,
        format_func=lambda l: _lang_labels[l],
        horizontal=True,
        index=_lang_idx,
        key="lang_radio",
    )
    st.session_state["lang"] = _lang_choice

    st.divider()

    _PAGE_KEYS = [
        "Tableau de bord", "Saisie", "Overview", "Forecast", "Analyse", "Finances",
        "Recommendations", "Products", "Shopping List", "Conseils", "About",
    ]
    _NAV_TKEYS = [
        "nav_dashboard", "nav_saisie", "nav_overview", "nav_forecast", "nav_analyse",
        "nav_finances", "nav_recommendations", "nav_products", "nav_shopping",
        "nav_conseils", "nav_about",
    ]
    _nav_labels = {k: T(tk) for k, tk in zip(_PAGE_KEYS, _NAV_TKEYS)}

    page = st.radio(
        "NAVIGATION",
        _PAGE_KEYS,
        format_func=lambda k: _nav_labels[k],
    )
    st.divider()

    uploaded = st.file_uploader(
        T("sidebar_file"),
        type=["csv", "xlsx", "xls"],
        help=(
            "Formats acceptés : CSV, Excel (.xlsx, .xls). "
            "Colonnes requises : date · produit · quantite_vendue."
        ),
    )
    if uploaded is None:
        st.markdown(
            f'<p style="font-size:.78rem;color:#64748b;margin-top:6px;">'
            f'{T("sidebar_demo")}</p>',
            unsafe_allow_html=True,
        )
    else:
        ext_icon = "📊" if uploaded.name.endswith((".xlsx", ".xls")) else "📄"
        st.markdown(
            f'<p style="font-size:.78rem;color:#34d399;margin-top:6px;">'
            f'{ext_icon} <b>{uploaded.name}</b></p>',
            unsafe_allow_html=True,
        )

    st.divider()
    ramadan_mode = st.checkbox(
        T("sidebar_ramadan"),
        value=False,
        help=T("sidebar_ramadan_help"),
    )
    lite_mode = st.checkbox(
        T("sidebar_lite"),
        value=False,
        help=T("sidebar_lite_help"),
    )
    st.markdown(
        '<p style="font-size:.72rem;color:#64748b;">v1.2 &nbsp;|&nbsp; '
        'FoodCast &copy; 2025</p>',
        unsafe_allow_html=True,
    )

# ── Connection status badge (injected once, outside sidebar) ─────────────
st.components.v1.html("""
<script>
(function() {
  try {
    var par = window.parent.document;
    var el = par.getElementById('fc-conn-badge');
    if (!el) {
      el = par.createElement('div');
      el.id = 'fc-conn-badge';
      par.body.appendChild(el);
    }
    function update() {
      var on = window.parent.navigator.onLine;
      el.textContent = on ? '🟢 En ligne' : '🔴 Hors ligne';
      el.style.background = on ? '#dcfce7' : '#fee2e2';
      el.style.color      = on ? '#166534' : '#991b1b';
    }
    window.parent.addEventListener('online',  update);
    window.parent.addEventListener('offline', update);
    update();
  } catch(e) {}
})();
</script>
""", height=0)


# ═════════════════════════════════════════════════════════════════════════════
# Data loading & validation gate
# ═════════════════════════════════════════════════════════════════════════════
file_bytes = uploaded.read() if uploaded is not None else None
file_ext   = uploaded.name.rsplit(".", 1)[-1].lower() if uploaded is not None else "csv"
df, load_errors = load_and_validate(file_bytes, file_ext)

if df is None:
    if page == "Saisie":
        # ── Manual-only mode: no historical file loaded ───────────────────
        st.markdown(
            f'<h1 style="color:#1e3a5f;font-weight:800;margin-bottom:.3rem;">{T("saisie_title")}</h1>',
            unsafe_allow_html=True,
        )
        st.info(T("saisie_nofile"))
        # Free-text product input when no CSV exists
        _man_prods_key = "manual_product_list"
        st.session_state.setdefault(_man_prods_key, ["Produit 1", "Produit 2"])
        _man_entries_key = "manual_entries"
        st.session_state.setdefault(_man_entries_key, [])

        _new_prod = st.text_input(T("saisie_new_prod"), placeholder="ex. Pizza, Burger…")
        if st.button(T("saisie_add_btn")) and _new_prod.strip():
            if _new_prod.strip() not in st.session_state[_man_prods_key]:
                st.session_state[_man_prods_key].append(_new_prod.strip())
                st.rerun()

        import datetime as _dt2
        _entry_date2 = st.date_input(T("saisie_date_label"), value=_dt2.date.today())
        _qty_inputs2 = {}
        for _p in st.session_state[_man_prods_key]:
            _qty_inputs2[_p] = st.number_input(_p, min_value=0, step=1, value=0, key=f"man_q2_{_p}")

        if st.button(T("saisie_save_btn2"), type="primary", use_container_width=True):
            _saved2 = False
            for _p, _q in _qty_inputs2.items():
                if _q > 0:
                    st.session_state[_man_entries_key].append({
                        "date": pd.Timestamp(_entry_date2),
                        "produit": _p,
                        "quantite_vendue": _q,
                    })
                    _saved2 = True
            if _saved2:
                st.success("Ventes enregistrées !")
                st.rerun()

        if st.session_state[_man_entries_key]:
            _exp_df2 = pd.DataFrame(st.session_state[_man_entries_key])
            _exp_df2["date"] = _exp_df2["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(_exp_df2, use_container_width=True, hide_index=True)
            st.download_button(
                T("saisie_download"),
                data=_exp_df2.to_csv(index=False).encode("utf-8"),
                file_name="mes_ventes.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if st.button(T("saisie_clear")):
                st.session_state[_man_entries_key] = []
                st.rerun()
        st.stop()
    else:
        st.error("### Problème avec votre fichier")
        for e in load_errors:
            st.markdown(f"- {e}")
        st.info(
            "**Format attendu :**\n\n"
            "| date | produit | quantite_vendue |\n"
            "|------|---------|----------------|\n"
            "| 2024-01-01 | Pizza Margherita | 28 |\n\n"
            "Formats acceptés : **CSV**, **Excel (.xlsx / .xls)**. "
            "Rendez-vous sur la page **À propos** pour télécharger un exemple."
        )
        st.stop()

# Shared derived data used by all pages
daily     = aggregate_daily(df)
data_hash = str(hash_pandas_object(daily).sum())
avg_open  = float(daily[daily["y"] > 0]["y"].mean())
last_date = daily["ds"].max()
fc_start  = last_date + pd.Timedelta(days=1)

# ── Product catalog (persisted across pages) — requires loaded df ─────────
_cat_key = f"catalog_{data_hash}"
if _cat_key not in st.session_state:
    st.session_state[_cat_key] = {
        p: {"active": True, "source": "csv"}
        for p in sorted(df["produit"].unique())
    }
else:
    # Ensure CSV products from re-upload are always present
    for p in sorted(df["produit"].unique()):
        st.session_state[_cat_key].setdefault(p, {"active": True, "source": "csv"})

catalog      = st.session_state[_cat_key]
active_prods = [p for p, v in catalog.items() if v["active"]]
active_csv   = [p for p, v in catalog.items() if v["active"] and v["source"] == "csv"]

# ── Financial data (prices, costs, daily target) ──────────────────────────
_prices_key = f"prices_{data_hash}"
_costs_key  = f"costs_{data_hash}"
_target_key = f"daily_target_{data_hash}"

_all_prods = sorted(df["produit"].unique())
if _prices_key not in st.session_state:
    init_px: dict[str, float] = {}
    for p in _all_prods:
        if "prix_unitaire" in df.columns:
            row_px = df[df["produit"] == p]["prix_unitaire"]
            init_px[p] = float(row_px.iloc[0]) if not row_px.empty else 0.0
        else:
            init_px[p] = 0.0
    st.session_state[_prices_key] = init_px

if _costs_key not in st.session_state:
    st.session_state[_costs_key] = {p: 0.0 for p in _all_prods}

if _target_key not in st.session_state:
    st.session_state[_target_key] = 0.0

# Ensure new products added later get a default
for p in _all_prods:
    st.session_state[_prices_key].setdefault(p, 0.0)
    st.session_state[_costs_key].setdefault(p, 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# Page 1 — Overview
# ═════════════════════════════════════════════════════════════════════════════
# Page — Saisie des ventes (quick daily entry)
# ═════════════════════════════════════════════════════════════════════════════
if page == "Saisie":
    import datetime as _dt

    st.markdown(
        f'<h1 style="color:#1e3a5f;font-weight:800;margin-bottom:.2rem;">{T("saisie_title")}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:#64748b;font-size:.88rem;margin-bottom:1.2rem;">{T("saisie_sub")}</p>',
        unsafe_allow_html=True,
    )

    _man_key = "manual_entries"
    st.session_state.setdefault(_man_key, [])

    # ── Entry form ────────────────────────────────────────────────────────────
    _sec(T("saisie_sec_entry"))

    # Determine default date: day after last known date, capped at today
    _today     = _dt.date.today()
    _default_d = min(last_date.date() + _dt.timedelta(days=1), _today)
    _entry_date = st.date_input(
        T("saisie_date_label"), value=_default_d,
        min_value=daily["ds"].min().date(),
        max_value=_today,
        label_visibility="collapsed",
    )

    # Check for duplicates (already in CSV or in manual entries for that date)
    _date_ts    = pd.Timestamp(_entry_date)
    _csv_prods_on_date = set(
        df[df["date"] == _date_ts]["produit"].unique()
    ) if df is not None else set()
    _man_prods_on_date = {
        e["produit"] for e in st.session_state[_man_key]
        if pd.Timestamp(e["date"]).date() == _entry_date
    }

    # Product inputs — one per row for mobile friendliness
    _products_for_entry = sorted(df["produit"].unique()) if df is not None else []
    _qty_inputs: dict[str, int] = {}

    cols_per_row = 2
    _prods_chunks = [
        _products_for_entry[i:i + cols_per_row]
        for i in range(0, len(_products_for_entry), cols_per_row)
    ]
    for _chunk in _prods_chunks:
        _row_cols = st.columns(len(_chunk))
        for _col, _prod in zip(_row_cols, _chunk):
            # Pre-fill if already entered today
            _existing = next(
                (e["quantite_vendue"] for e in st.session_state[_man_key]
                 if e["produit"] == _prod and pd.Timestamp(e["date"]).date() == _entry_date),
                0,
            )
            _already_in_csv = _prod in _csv_prods_on_date
            _help = T("saisie_already_csv") if _already_in_csv else ""
            _qty_inputs[_prod] = _col.number_input(
                _prod, min_value=0, step=1,
                value=int(_existing),
                key=f"saisie_{_prod}_{_entry_date}",
                help=_help if _help else None,
            )

    if st.button(T("saisie_save_btn"), type="primary", use_container_width=True):
        _saved_count = 0
        for _prod, _qty in _qty_inputs.items():
            if _qty > 0:
                # Remove existing manual entry for same date+product (update)
                st.session_state[_man_key] = [
                    e for e in st.session_state[_man_key]
                    if not (e["produit"] == _prod and pd.Timestamp(e["date"]).date() == _entry_date)
                ]
                st.session_state[_man_key].append({
                    "date":             pd.Timestamp(_entry_date),
                    "produit":          _prod,
                    "quantite_vendue":  _qty,
                })
                _saved_count += 1
        if _saved_count:
            st.success(T("saisie_success").format(n=_saved_count, d=_entry_date.strftime("%d/%m/%Y")))
        else:
            st.warning(T("saisie_warn"))

    # ── Manual entries history ─────────────────────────────────────────────────
    _man_entries = st.session_state[_man_key]
    if _man_entries:
        _sec(T("saisie_history"))
        _man_df = pd.DataFrame(_man_entries).copy()
        _man_df["date"] = pd.to_datetime(_man_df["date"]).dt.strftime("%Y-%m-%d")
        _man_df = _man_df.sort_values(["date", "produit"]).reset_index(drop=True)
        _man_df.columns = ["Date", "Produit", "Quantité saisie"]

        # KPIs
        n_days_manual  = _man_df["Date"].nunique()
        total_manual   = int(_man_df["Quantité saisie"].sum())
        k1, k2 = st.columns(2)
        k1.markdown(_kpi(T("saisie_kpi_days"), str(n_days_manual), T("saisie_kpi_days_sub"), "✏️"), unsafe_allow_html=True)
        k2.markdown(_kpi(T("saisie_kpi_total"), f"{total_manual:,}", T("saisie_kpi_total_sub"), "📦"), unsafe_allow_html=True)
        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

        st.dataframe(_man_df, use_container_width=True, hide_index=True)

        # ── Export ────────────────────────────────────────────────────────────
        _sec(T("saisie_export"))
        st.markdown(
            f'<div class="lite-banner">{T("saisie_sync_tip")}</div>',
            unsafe_allow_html=True,
        )

        # Build combined CSV (original + manual entries)
        _base_df = df[["date", "produit", "quantite_vendue"]].copy() if df is not None else pd.DataFrame()
        _base_df["date"] = pd.to_datetime(_base_df["date"]).dt.strftime("%Y-%m-%d")
        _new_rows = pd.DataFrame(_man_entries)
        _new_rows["date"] = pd.to_datetime(_new_rows["date"]).dt.strftime("%Y-%m-%d")
        _combined = pd.concat([_base_df, _new_rows], ignore_index=True)
        _combined = (
            _combined
            .drop_duplicates(subset=["date", "produit"], keep="last")
            .sort_values(["date", "produit"])
            .reset_index(drop=True)
        )

        dl_col, del_col = st.columns([3, 1])
        with dl_col:
            st.download_button(
                T("saisie_download"),
                data=_combined.to_csv(index=False).encode("utf-8"),
                file_name="ventes_complet.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )
        with del_col:
            if st.button(T("saisie_clear"), use_container_width=True):
                st.session_state[_man_key] = []
                st.rerun()
    else:
        st.info(T("saisie_empty"))

# ═════════════════════════════════════════════════════════════════════════════
if page == "Overview":

    st.markdown(
        '<h1 style="color:#1e3a5f;font-weight:800;margin-bottom:.2rem;">Sales Overview</h1>',
        unsafe_allow_html=True,
    )
    src_label = f"<b>{uploaded.name}</b>" if uploaded else "demo dataset"
    st.markdown(
        f'<p style="color:#64748b;font-size:.88rem;margin-bottom:1.2rem;">'
        f'Data source: {src_label} &nbsp;|&nbsp; '
        f'{len(df):,} rows &nbsp;|&nbsp; '
        f'{daily["ds"].min().date()} &rarr; {daily["ds"].max().date()}'
        f'</p>',
        unsafe_allow_html=True,
    )

    # ── 4 KPI cards ──────────────────────────────────────────────────────────
    total_sales  = int(df["quantite_vendue"].sum())
    best_product = df.groupby("produit")["quantite_vendue"].sum().idxmax()
    avg_daily    = round(float(daily[daily["y"] > 0]["y"].mean()), 1)

    df_open = df[df["quantite_vendue"] > 0].copy()
    df_open["day_name"] = df_open["date"].dt.day_name()
    busiest_day = df_open.groupby("day_name")["quantite_vendue"].mean().idxmax()

    c1, c2, c3, c4 = st.columns(4)
    for col, html in zip(
        [c1, c2, c3, c4],
        [
            _kpi("Total Units Sold",    f"{total_sales:,}",  f"{daily['ds'].min().year}–{daily['ds'].max().year}", "📦"),
            _kpi("Best-Selling Product", best_product,        "by total volume", "🏆"),
            _kpi("Avg Daily Sales",      f"{avg_daily}",      "units/day (open days)", "📈"),
            _kpi("Busiest Day",          busiest_day,         "highest avg sales", "📅"),
        ],
    ):
        col.markdown(html, unsafe_allow_html=True)

    # ── Sales trend with date range selector ─────────────────────────────────
    _sec("Sales Trend")
    min_d, max_d = daily["ds"].min().date(), daily["ds"].max().date()
    date_range = st.date_input(
        "Filter date range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
        label_visibility="collapsed",
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = date_range
        mask = (daily["ds"].dt.date >= start_d) & (daily["ds"].dt.date <= end_d)
        filtered = daily[mask]
    else:
        filtered = daily

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=filtered["ds"], y=filtered["y"],
        mode="lines", fill="tozeroy",
        fillcolor="rgba(20,184,166,.10)",
        line=dict(color="#14b8a6", width=1.8),
        name="Daily sales",
    ))
    fig_trend.update_layout(
        template="plotly_white", hovermode="x unified",
        xaxis_title="", yaxis_title="Units Sold",
        margin=dict(t=20, b=30),
        height=300,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Product breakdown: pie + bar ─────────────────────────────────────────
    _sec("Product Breakdown")
    by_product = (
        df[df["produit"].isin(active_csv)]
        .groupby("produit")["quantite_vendue"].sum()
        .reset_index().sort_values("quantite_vendue", ascending=False)
        .rename(columns={"produit": "Product", "quantite_vendue": "Total"})
    )
    PALETTE = ["#14b8a6", "#1e3a5f", "#0ea5e9", "#6366f1"]

    col_pie, col_bar = st.columns(2)
    with col_pie:
        fig_pie = px.pie(
            by_product, names="Product", values="Total",
            color_discrete_sequence=PALETTE,
            hole=0.42,
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, margin=dict(t=20, b=20), height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        fig_bar = px.bar(
            by_product, x="Product", y="Total",
            color="Product", color_discrete_sequence=PALETTE,
            text_auto=True,
        )
        fig_bar.update_layout(
            showlegend=False, template="plotly_white",
            margin=dict(t=20, b=20), height=320,
            xaxis_title="", yaxis_title="Total Units",
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Data health ──────────────────────────────────────────────────────────
    _sec("Data Health")
    checks = check_data_health(df)
    badge_map = {"ok": "badge-ok", "warn": "badge-warn", "err": "badge-err"}
    label_map = {"ok": "OK", "warn": "Warning", "err": "Error"}
    rows_html = ""
    for c in checks:
        cls = badge_map[c["status"]]
        lbl = label_map[c["status"]]
        rows_html += (
            f'<div class="health-row">'
            f'<span class="health-label">{c["label"]}</span>'
            f'<span class="badge {cls}">{lbl}</span>'
            f'<span style="color:#94a3b8">{c["detail"]}</span>'
            f'</div>'
        )
    st.markdown(
        f'<div style="background:#fff;border-radius:12px;padding:14px 20px;'
        f'box-shadow:0 2px 12px rgba(0,0,0,.06);">{rows_html}</div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Page 2 — Forecast
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Forecast":

    st.markdown(
        f'<h1 style="color:#1e3a5f;font-weight:800;margin-bottom:.2rem;">{T("forecast_title")}</h1>',
        unsafe_allow_html=True,
    )

    if lite_mode:
        st.markdown(
            f'<div class="lite-banner">{T("forecast_lite_banner")}</div>',
            unsafe_allow_html=True,
        )
        _sec(T("forecast_lite_section"))
        _lite_df = daily.tail(30).copy()
        fig_lite = go.Figure()
        fig_lite.add_trace(go.Scatter(
            x=_lite_df["ds"], y=_lite_df["y"],
            mode="lines+markers", fill="tozeroy",
            fillcolor="rgba(20,184,166,.10)",
            line=dict(color="#14b8a6", width=1.8),
        ))
        fig_lite.add_hline(y=avg_open, line=dict(color="#94a3b8", dash="dash"))
        fig_lite.update_layout(template="plotly_white", height=280, margin=dict(t=10, b=10),
                               xaxis_title="", yaxis_title=T("units"))
        st.plotly_chart(fig_lite, use_container_width=True)
        st.stop()

    # ── Controls row ─────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 4])
    with ctrl1:
        horizon = st.selectbox(
            T("forecast_horizon"),
            options=[7, 14, 30],
            index=2,
            format_func=lambda x: T("forecast_horizon_fmt").format(n=x),
        )
    with ctrl2:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        with st.spinner("..."):
            mae = compute_mae(data_hash, daily)
        if mae is None:
            acc_val = T("forecast_acc_na")
            acc_sub = T("forecast_acc_nodata")
        elif mae < 15:
            acc_val = T("forecast_acc_good")
            acc_sub = T("forecast_acc_sub").format(mae=mae)
        elif mae < 25:
            acc_val = T("forecast_acc_ok")
            acc_sub = T("forecast_acc_sub").format(mae=mae)
        else:
            acc_val = T("forecast_acc_fair")
            acc_sub = T("forecast_acc_sub").format(mae=mae)
        st.markdown(
            f'<div class="acc-box"><span class="acc-val">{acc_val}</span>'
            f'<span class="acc-lbl">{T("forecast_acc_label")} &mdash; {acc_sub}</span></div>',
            unsafe_allow_html=True,
        )

    # ── Train & forecast ──────────────────────────────────────────────────────
    with st.spinner(T("forecast_spinner")):
        model = get_model(data_hash, daily)
        pred  = run_forecast(data_hash, horizon, model, daily)

    pred_future = pred[pred["ds"] >= fc_start].copy()

    # ── Chart ─────────────────────────────────────────────────────────────────
    _sec(T("forecast_chart_sec").format(n=horizon))
    fig = go.Figure()

    # CI ribbon
    fig.add_trace(go.Scatter(
        x=pd.concat([pred["ds"], pred["ds"][::-1]]),
        y=pd.concat([pred["yhat_upper"], pred["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(20,184,166,.13)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip", name=T("forecast_range"),
    ))
    # Forecast line
    fig.add_trace(go.Scatter(
        x=pred["ds"], y=pred["yhat"],
        mode="lines", line=dict(color="#14b8a6", width=2, dash="dot"),
        name=T("forecast_pred"),
    ))
    # Actuals
    fig.add_trace(go.Scatter(
        x=daily["ds"], y=daily["y"],
        mode="lines", line=dict(color="#1e3a5f", width=1.2),
        name=T("forecast_history"),
    ))
    fig.add_vline(
        x=fc_start.timestamp() * 1000,
        line=dict(color="crimson", dash="dash", width=1.5),
        annotation_text=T("forecast_fc_start").format(d=fc_start.date()),
        annotation_position="top right",
    )
    fig.update_layout(
        template="plotly_white", hovermode="x unified",
        xaxis_title="", yaxis_title=T("units"),
        height=420, margin=dict(t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Alertes pics imminents ────────────────────────────────────────────────
    peaks = get_upcoming_peaks(pred_future, avg_open, n_days=horizon)
    if peaks:
        _sec(T("forecast_peaks"))
        for p in peaks:
            label = p["date"].strftime("%A %d %b")
            reasons_str = " · ".join(p["reasons"])
            st.warning(f"**{label}** — {p['yhat']} {T('units')} · {reasons_str}")

    if ramadan_mode:
        # Show which forecast days fall in Ramadan
        ram_days = [row for _, row in pred_future.iterrows() if is_ramadan_period(row["ds"])]
        if ram_days:
            _sec(T("forecast_ram_sec"))
            st.info(T("forecast_ram_msg").format(n=len(ram_days)))

    # ── Weekly summary table ──────────────────────────────────────────────────
    _sec(T("forecast_weekly"))
    wk = pred_future.copy()
    wk["week_start"] = wk["ds"].dt.to_period("W").apply(lambda p: p.start_time)
    wk["week_end"]   = wk["ds"].dt.to_period("W").apply(lambda p: p.end_time)
    weekly = (
        wk.groupby(["week_start", "week_end"])
        .agg(
            Days       = ("ds",   "count"),
            Total      = ("yhat", lambda x: int(x.clip(lower=0).sum().round())),
            Avg_Daily  = ("yhat", lambda x: round(float(x.clip(lower=0).mean()), 1)),
            CI_Lower   = ("yhat_lower", lambda x: int(x.clip(lower=0).sum().round())),
            CI_Upper   = ("yhat_upper", lambda x: int(x.clip(lower=0).sum().round())),
        )
        .reset_index()
    )
    weekly["Week"] = weekly.apply(
        lambda r: f"{r['week_start'].strftime('%b %d')} – {r['week_end'].strftime('%b %d')}", axis=1
    )
    weekly = weekly[["Week", "Days", "Total", "Avg_Daily", "CI_Lower", "CI_Upper"]]
    weekly.columns = Tlist("forecast_weekly_cols") or weekly.columns.tolist()
    st.dataframe(weekly.reset_index(drop=True), use_container_width=True, hide_index=True)

    # ── Monthly summary table ─────────────────────────────────────────────────
    _sec(T("forecast_monthly"))
    mo = pred_future.copy()
    mo["month"] = mo["ds"].dt.to_period("M")
    monthly = (
        mo.groupby("month")
        .agg(
            Total     = ("yhat", lambda x: int(x.clip(lower=0).sum().round())),
            Avg_Daily = ("yhat", lambda x: round(float(x.clip(lower=0).mean()), 1)),
            CI_Lower  = ("yhat_lower", lambda x: int(x.clip(lower=0).sum().round())),
            CI_Upper  = ("yhat_upper", lambda x: int(x.clip(lower=0).sum().round())),
        )
        .reset_index()
    )
    monthly["Mois"] = monthly["month"].dt.strftime("%B %Y")
    monthly = monthly[["Mois", "Total", "Avg_Daily", "CI_Lower", "CI_Upper"]]
    monthly.columns = Tlist("forecast_monthly_cols") or monthly.columns.tolist()
    st.dataframe(monthly, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# Page 3 — Recommendations
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Recommendations":

    st.markdown(
        '<h1 style="color:#1e3a5f;font-weight:800;margin-bottom:.2rem;">What to Prepare This Week</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#64748b;font-size:.88rem;margin-bottom:1rem;">'
        'Next 7 days &nbsp;|&nbsp; includes a +15% safety buffer so you never run short</p>',
        unsafe_allow_html=True,
    )

    with st.spinner("Calculating your stock plan..."):
        model = get_model(data_hash, daily)
        pred  = run_forecast(data_hash, 30, model, daily)

    next7 = pred[pred["ds"] >= fc_start].head(7).copy()
    next7["day_name"]      = next7["ds"].dt.day_name()
    next7["predicted_qty"] = next7["yhat"].clip(lower=0).round(0).astype(int)
    next7["suggested"]     = (next7["predicted_qty"] * BUFFER_RATE).round(0).astype(int)
    next7["is_weekend"]    = next7["day_name"].isin(WEEKEND_DAYS)
    next7["exceeds_avg"]   = next7["yhat"] > avg_open * PEAK_THRESH
    next7["risk"]          = next7.apply(lambda r: _risk_level(r, avg_open), axis=1)

    # ── Summary box ───────────────────────────────────────────────────────────
    total_stock = int(next7["suggested"].sum())
    total_pred  = int(next7["predicted_qty"].sum())
    st.markdown(
        f'<div class="sum-box">'
        f'<span class="sum-big">{total_stock:,} units</span>'
        f'<span class="sum-sub">Total stock to prepare this week '
        f'(forecast: {total_pred:,} + 15% buffer)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Alerts: high demand + holidays + Ramadan ──────────────────────────────
    high_days = next7[next7["exceeds_avg"]]
    alerted = False
    for _, row in high_days.iterrows():
        pct = (row["yhat"] / avg_open - 1) * 100
        extras: list[str] = []
        h = _get_holiday_name(row["ds"])
        if h:
            extras.append(f"public holiday: {h}")
        if is_ramadan_period(row["ds"]):
            extras.append("Ramadan")
        extra_str = f" · {', '.join(extras)}" if extras else ""
        st.warning(
            f"**{row['ds'].strftime('%A %d %b')}** — {row['predicted_qty']} units forecast "
            f"(**+{pct:.0f}%** vs avg){extra_str}. Prepare extra stock."
        )
        alerted = True

    # Holiday/Ramadan alerts for days that aren't necessarily peak but are special
    for _, row in next7[~next7["exceeds_avg"]].iterrows():
        h = _get_holiday_name(row["ds"])
        ram = is_ramadan_period(row["ds"])
        if h or ram:
            tags = ([f"Public holiday: **{h}**"] if h else []) + (["**Ramadan**"] if ram else [])
            st.info(f"**{row['ds'].strftime('%A %d %b')}** — {' · '.join(tags)}")
            alerted = True

    if ramadan_mode:
        ram_days_next7 = [row for _, row in next7.iterrows() if is_ramadan_period(row["ds"])]
        if ram_days_next7:
            st.warning(
                f"**Ramadan Mode** : {len(ram_days_next7)} day(s) this week fall during Ramadan. "
                "Focus production on the evening service (Iftar). "
                "Recommended products: Harira, Briouates, Chebakia."
            )
            alerted = True

    if not alerted:
        st.success("No extraordinary peaks forecast. Normal operations recommended.")

    # ── Color-coded table ─────────────────────────────────────────────────────
    _sec("Daily Stock Plan")

    display = next7[[
        "ds", "day_name", "predicted_qty", "suggested",
        "is_weekend", "risk",
        "yhat_lower", "yhat_upper",
    ]].copy()
    display["ds"]        = display["ds"].dt.strftime("%Y-%m-%d")
    display["yhat_lower"] = display["yhat_lower"].clip(lower=0).round(0).astype(int)
    display["yhat_upper"] = display["yhat_upper"].round(0).astype(int)
    display["is_weekend"] = display["is_weekend"].map({True: "Weekend", False: ""})
    display.columns = [
        "Date", "Day", "Expected Sales", "Stock to Prepare",
        "Peak Day", "Risk Level", "Min", "Max",
    ]

    RISK_COLOR  = {"High": "#fee2e2", "Medium": "#fef9c3", "Low": "#dcfce7"}
    PEAK_COLOR  = "#fee2e2"
    NORM_COLOR  = "#dcfce7"
    WKND_COLOR  = "#fef9c3"

    def _style_rows(row):
        if row["Risk Level"] == "High":
            bg = PEAK_COLOR
        elif row["Peak Day"] == "Weekend":
            bg = WKND_COLOR
        else:
            bg = NORM_COLOR
        return [f"background-color:{bg}"] * len(row)

    def _style_risk(val):
        colors = {"High": "color:#991b1b;font-weight:700",
                  "Medium": "color:#854d0e;font-weight:700",
                  "Low": "color:#166534;font-weight:700"}
        return colors.get(val, "")

    styled = (
        display.reset_index(drop=True)
        .style
        .apply(_style_rows, axis=1)
        .map(_style_risk, subset=["Risk Level"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown(
        f'<p style="font-size:.78rem;color:#94a3b8;">'
        f'Your typical daily sales: <b>{avg_open:.0f} units</b> &nbsp;|&nbsp; '
        f'Safety buffer: <b>+15%</b> &nbsp;|&nbsp; '
        f'<b>High risk</b> = more than 30% above your average &nbsp;|&nbsp; '
        f'Min/Max = best and worst case estimates</p>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Page 4 — Products (CRUD)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Products":

    st.markdown(
        '<h1 style="color:#1e3a5f;font-weight:800;margin-bottom:.2rem;">My Dishes</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#64748b;font-size:.88rem;margin-bottom:1.2rem;">'
        'Add or remove dishes from your menu. Uncheck a dish to hide it from charts '
        'and the shopping list. You can also add new dishes and set up their ingredients '
        'on the Shopping List page.</p>',
        unsafe_allow_html=True,
    )

    # ── KPI row ───────────────────────────────────────────────────────────────
    n_total  = len(catalog)
    n_active = len(active_prods)
    n_custom = sum(1 for v in catalog.values() if v["source"] == "custom")

    k1, k2, k3 = st.columns(3)
    for col, html in zip([k1, k2, k3], [
        _kpi("Total Dishes",   str(n_total),  "on your menu", "📋"),
        _kpi("Active Dishes",  str(n_active), "included in calculations", "✅"),
        _kpi("Added by You",   str(n_custom), "not in your sales file", "✏️"),
    ]):
        col.markdown(html, unsafe_allow_html=True)

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    # ── Product list (editable Active toggle) ─────────────────────────────────
    _sec("Dish List")
    st.caption(
        "Check or uncheck **Active** to show or hide a dish across the app. "
        "Dishes from your sales file cannot be deleted (their history is used for predictions), "
        "but you can deactivate them."
    )

    # Check if recipe is defined for each product (looks in session state)
    def _has_recipe(name: str) -> bool:
        if name in RECIPES:
            return True
        rkey = f"recipe_{name}_{data_hash}"
        if rkey in st.session_state:
            rdf = st.session_state[rkey]
            valid = rdf.dropna(subset=["Ingredient"])
            valid = valid[valid["Ingredient"].astype(str).str.strip().ne("")]
            return len(valid) > 0
        return False

    catalog_rows = []
    for name, meta in sorted(catalog.items()):
        share = df[df["produit"] == name]["quantite_vendue"].sum() / max(df["quantite_vendue"].sum(), 1)
        catalog_rows.append({
            "Dish":           name,
            "Source":         "From your data" if meta["source"] == "csv" else "Added by you",
            "Sales Share":    f"{share*100:.1f}%" if meta["source"] == "csv" else "—",
            "Recipe Set Up":  "Yes" if _has_recipe(name) else "No",
            "Active":         meta["active"],
        })

    cat_df = pd.DataFrame(catalog_rows)

    edited_cat = st.data_editor(
        cat_df,
        column_config={
            "Dish":          st.column_config.TextColumn("Dish",           disabled=True),
            "Source":        st.column_config.TextColumn("Source",         disabled=True),
            "Sales Share":   st.column_config.TextColumn("Sales Share",    disabled=True,
                               help="How much of your total sales this dish represents"),
            "Recipe Set Up": st.column_config.TextColumn("Recipe Set Up",  disabled=True,
                               help="Whether ingredients have been defined on the Shopping List page"),
            "Active":        st.column_config.CheckboxColumn("Active",
                               help="Uncheck to hide this dish from charts and the shopping list"),
        },
        hide_index=True,
        use_container_width=True,
        key=f"catalog_editor_{data_hash}",
    )

    # Commit active toggles back to catalog
    for _, row in edited_cat.iterrows():
        if row["Dish"] in catalog:
            catalog[row["Dish"]]["active"] = bool(row["Active"])

    # ── Add new product ───────────────────────────────────────────────────────
    _sec("Add a New Dish")
    st.caption(
        "Add a dish that isn't in your sales file. "
        "Once added, go to **Shopping List** to set up its ingredients."
    )

    with st.form("add_product_form", clear_on_submit=True):
        add_col, btn_col = st.columns([4, 1])
        with add_col:
            new_name = st.text_input(
                "Product name",
                placeholder="e.g. French Fries, Onion Rings...",
                label_visibility="collapsed",
            )
        with btn_col:
            submitted = st.form_submit_button("Add", use_container_width=True)

        if submitted:
            name_clean = new_name.strip()
            if not name_clean:
                st.error("Please enter a dish name.")
            elif name_clean in catalog:
                st.warning(f"**{name_clean}** is already on your menu.")
            else:
                catalog[name_clean] = {"active": True, "source": "custom"}
                st.success(f"**{name_clean}** added! Go to Shopping List to set up its ingredients.")
                st.rerun()

    # ── Delete custom products ────────────────────────────────────────────────
    custom_list = [p for p, v in catalog.items() if v["source"] == "custom"]
    if custom_list:
        _sec("Remove a Dish")
        st.caption("Only dishes you added manually can be removed.")

        del_col, btn_col = st.columns([4, 1])
        with del_col:
            to_delete = st.selectbox(
                "Select product to delete",
                options=["— select —"] + sorted(custom_list),
                label_visibility="collapsed",
            )
        with btn_col:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            if st.button("Delete", type="primary", use_container_width=True):
                if to_delete != "— select —":
                    del catalog[to_delete]
                    # Also clean up its recipe from session state if present
                    rkey = f"recipe_{to_delete}_{data_hash}"
                    if rkey in st.session_state:
                        del st.session_state[rkey]
                    st.success(f"**{to_delete}** removed from catalog.")
                    st.rerun()

    # ── Info callout ──────────────────────────────────────────────────────────
    st.divider()
    st.info(
        "**Dishes from your sales file** can be deactivated but not deleted — "
        "their sales history is needed to make accurate predictions.  \n"
        "**Dishes you add manually** can be freely added and removed. "
        "Set up their ingredients on the Shopping List page."
    )


# ═════════════════════════════════════════════════════════════════════════════
# Page 5 — Shopping List
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Shopping List":

    st.markdown(
        '<h1 style="color:#1e3a5f;font-weight:800;margin-bottom:.2rem;">Shopping List</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#64748b;font-size:.88rem;margin-bottom:1.2rem;">'
        'Everything you need to buy for the next 7 days, automatically calculated '
        'from your dish ingredients and sales predictions. All quantities and units '
        'are fully editable.</p>',
        unsafe_allow_html=True,
    )

    # ── Forecast next 7 days ──────────────────────────────────────────────────
    with st.spinner("Calculating your shopping needs..."):
        model = get_model(data_hash, daily)
        pred  = run_forecast(data_hash, 7, model, daily)

    next7       = pred[pred["ds"] >= fc_start].head(7).copy()
    next7_dates = next7["ds"].tolist()

    product_shares = (
        df.groupby("produit")["quantite_vendue"]
        .sum()
        .div(df["quantite_vendue"].sum())
    )
    # Use active products from catalog (includes custom products with no CSV data)
    products_list = active_prods if active_prods else sorted(df["produit"].unique())

    # ── Section 1: Dish Ingredients (one tab per product, fully editable) ──────
    _sec("Dish Ingredients")
    st.caption(
        "For each dish, list the ingredients needed to make one serving. "
        "You can edit names, quantities and units freely. Use **+** to add an ingredient, "
        "the trash icon to remove one."
    )

    recipe_tabs = st.tabs(products_list)
    edited_recipes: dict[str, pd.DataFrame] = {}

    for tab, product in zip(recipe_tabs, products_list):
        _rkey_p = f"recipe_{product}_{data_hash}"
        if _rkey_p not in st.session_state:
            if product in RECIPES:
                rows = [
                    {"Ingredient": ing, "Amount per serving": float(qty), "Unit": unit}
                    for ing, (qty, unit) in RECIPES[product].items()
                ]
            else:
                rows = [{"Ingredient": "", "Amount per serving": 0.0, "Unit": "g"}]
            st.session_state[_rkey_p] = pd.DataFrame(rows)

        with tab:
            edited = st.data_editor(
                st.session_state[_rkey_p],
                column_config={
                    "Ingredient": st.column_config.TextColumn(
                        "Ingredient",
                        help="What you need to buy (e.g. Ground beef, Burger bun)",
                    ),
                    "Amount per serving": st.column_config.NumberColumn(
                        "Amount per serving",
                        min_value=0,
                        format="%.1f",
                        help="How much of this ingredient goes into one serving of this dish",
                    ),
                    "Unit": st.column_config.SelectboxColumn(
                        "Unit",
                        options=UNIT_OPTIONS,
                        help="Unit of measurement — e.g. g for meat, piece for buns",
                    ),
                },
                num_rows="dynamic",
                hide_index=True,
                use_container_width=True,
                key=f"recipe_editor_{product}_{data_hash}",
            )
            st.session_state[_rkey_p] = edited
            edited_recipes[product] = edited

    # Build ingredient universe from all edited recipes
    # (shared ingredients like "Burger bun" accumulate across products)
    ingredient_unit: dict[str, str] = {}
    for product, rdf in edited_recipes.items():
        valid = rdf.dropna(subset=["Ingredient"])
        valid = valid[valid["Ingredient"].astype(str).str.strip().ne("")]
        for _, row in valid.iterrows():
            ingredient_unit.setdefault(str(row["Ingredient"]), str(row["Unit"]))
    all_ingredients = list(ingredient_unit.keys())

    if not all_ingredients:
        st.warning("No ingredients defined yet. Fill in the Dish Ingredients section above.")
        st.stop()

    # ── Section 2: Current Stock (editable per ingredient) ────────────────────
    _sec("What You Already Have")
    st.caption(
        "Enter how much of each ingredient you currently have in stock. "
        "Only what you're missing will appear in your shopping order."
    )

    _skey = f"stock_{data_hash}"
    if _skey not in st.session_state:
        st.session_state[_skey] = {}
    for ing in all_ingredients:
        st.session_state[_skey].setdefault(ing, 0)

    stock_df = pd.DataFrame([
        {"Ingredient":     ing,
         "Unit":           ingredient_unit[ing],
         "Current Stock":  int(st.session_state[_skey].get(ing, 0))}
        for ing in all_ingredients
    ])

    edited_stock = st.data_editor(
        stock_df,
        column_config={
            "Ingredient":    st.column_config.TextColumn("Ingredient", disabled=True),
            "Unit":          st.column_config.TextColumn("Unit", disabled=True),
            "Current Stock": st.column_config.NumberColumn(
                "Current Stock", min_value=0, step=1,
                help="Quantity available right now",
            ),
        },
        hide_index=True,
        use_container_width=True,
        key=f"stock_editor_{data_hash}",
    )
    for _, row in edited_stock.iterrows():
        st.session_state[_skey][row["Ingredient"]] = int(row["Current Stock"])

    current_stocks: dict[str, int] = {
        ing: int(st.session_state[_skey].get(ing, 0)) for ing in all_ingredients
    }

    # ── Daily ingredient consumption from edited recipes ──────────────────────
    ingredient_daily: dict[str, list[float]] = {ing: [] for ing in all_ingredients}
    for d_idx in range(len(next7)):
        yhat_total = float(next7.iloc[d_idx]["yhat"])
        day: dict[str, float] = {ing: 0.0 for ing in all_ingredients}
        for product, rdf in edited_recipes.items():
            share    = float(product_shares.get(product, 0))
            prod_qty = yhat_total * share
            valid    = rdf.dropna(subset=["Ingredient"])
            valid    = valid[valid["Ingredient"].astype(str).str.strip().ne("")]
            for _, row in valid.iterrows():
                ing = str(row["Ingredient"])
                if ing in day:
                    day[ing] += prod_qty * float(row["Amount per serving"])
        for ing in all_ingredients:
            ingredient_daily[ing].append(day[ing])

    ingredient_total  = {ing: sum(ingredient_daily[ing]) for ing in all_ingredients}
    ingredient_needed = {ing: round(ingredient_total[ing] * BUFFER_RATE) for ing in all_ingredients}
    to_order_map      = {ing: max(0, ingredient_needed[ing] - current_stocks[ing])
                         for ing in all_ingredients}

    # ── KPI row ───────────────────────────────────────────────────────────────
    n_to_buy = sum(1 for v in to_order_map.values() if v > 0)
    n_ok     = len(all_ingredients) - n_to_buy

    soonest_dt: pd.Timestamp | None = None
    for ing in all_ingredients:
        s = compute_stockout_date(current_stocks[ing], ingredient_daily[ing], next7_dates)
        if not s.startswith("OK"):
            d = pd.Timestamp(s)
            if soonest_dt is None or d < soonest_dt:
                soonest_dt = d
    soonest_str = soonest_dt.strftime("%b %d") if soonest_dt else "None"

    k1, k2, k3 = st.columns(3)
    for col, html in zip([k1, k2, k3], [
        _kpi("Ingredients to buy", str(n_to_buy),  f"out of {len(all_ingredients)}", "🛒"),
        _kpi("Already in stock",   str(n_ok),       "enough for the next 7 days", "✅"),
        _kpi("First shortage",     soonest_str,     "earliest estimated date", "⚠️"),
    ]):
        col.markdown(html, unsafe_allow_html=True)

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ── Section 3: Daily Needs Breakdown ─────────────────────────────────────
    _sec("Daily Needs — Next 7 Days")

    day_labels = [pd.Timestamp(d).strftime("%a %m/%d") for d in next7_dates]
    needs_rows = []
    for ing in all_ingredients:
        row_d: dict = {"Ingredient": ing, "Unit": ingredient_unit[ing]}
        for label, qty in zip(day_labels, ingredient_daily[ing]):
            row_d[label] = round(qty)
        row_d["7-day Total"]       = round(ingredient_total[ing])
        row_d["Total + 15% buffer"] = ingredient_needed[ing]
        needs_rows.append(row_d)

    needs_df = pd.DataFrame(needs_rows)
    st.dataframe(
        needs_df.style.map(lambda _: "font-weight:700",
                           subset=["7-day Total", "Total + 15% buffer"]),
        use_container_width=True,
        hide_index=True,
    )

    # ── Section 4: Shopping Order List ───────────────────────────────────────
    _sec("What to Buy")

    S_OK  = "OK"
    S_ORD = "Order"
    S_CRI = "Critical"
    S_BG  = {S_OK: "#dcfce7", S_ORD: "#fef9c3", S_CRI: "#fee2e2"}
    S_FG  = {S_OK:  "color:#166534;font-weight:700",
             S_ORD: "color:#854d0e;font-weight:700",
             S_CRI: "color:#991b1b;font-weight:700"}

    order_rows = []
    for ing in all_ingredients:
        need  = ingredient_needed[ing]
        stock = current_stocks[ing]
        order = to_order_map[ing]
        status = S_OK if order == 0 else (S_CRI if order > need * 0.5 else S_ORD)
        order_rows.append({
            "Ingredient":          ing,
            "Unit":                ingredient_unit[ing],
            "Need for 7 days (+15%)": need,
            "You Have":            stock,
            "To Buy":              order,
            "Status":              status,
        })

    order_df = pd.DataFrame(order_rows)

    def _style_order(row):
        return [f"background-color:{S_BG.get(row['Status'], '')}"] * len(row)

    st.dataframe(
        order_df.style
        .apply(_style_order, axis=1)
        .map(lambda v: S_FG.get(v, ""), subset=["Status"]),  # type: ignore[arg-type]
        use_container_width=True,
        hide_index=True,
    )

    # ── Section 5: Estimated Stockout Dates ──────────────────────────────────
    _sec("When Will You Run Out?")

    URG_BG = {"OK": "#dcfce7", "Very soon": "#fee2e2",
              "This week": "#fef9c3", "Later": "#fff7ed"}
    URG_FG = {"OK":        "color:#166534;font-weight:700",
              "Very soon": "color:#991b1b;font-weight:700",
              "This week": "color:#854d0e;font-weight:700",
              "Later":     "color:#9a3412;font-weight:700"}

    stockout_rows = []
    for ing in all_ingredients:
        s = compute_stockout_date(current_stocks[ing], ingredient_daily[ing], next7_dates)
        avg_d = round(ingredient_total[ing] / 7, 1)
        if s.startswith("OK"):
            date_str, delay, urg = "—", "—", "OK"
        else:
            delta    = (pd.Timestamp(s) - fc_start).days + 1
            date_str = s
            delay    = f"D+{delta}"
            urg      = "Very soon" if delta <= 2 else ("This week" if delta <= 4 else "Later")
        stockout_rows.append({
            "Ingredient":    ing,
            "Unit":          ingredient_unit[ing],
            "In Stock":      current_stocks[ing],
            "Used per Day":  avg_d,
            "Runs Out On":   date_str,
            "In":            delay,
            "Urgency":       urg,
            "_bg":           URG_BG.get(urg, "#fff"),
        })

    stockout_df = pd.DataFrame(stockout_rows)

    def _style_stockout(row):
        return [f"background-color:{stockout_df.at[row.name, '_bg']}"] * len(row)

    st.dataframe(
        stockout_df[["Ingredient", "Unit", "In Stock", "Used per Day",
                     "Runs Out On", "In", "Urgency"]]
        .style
        .apply(_style_stockout, axis=1)
        .map(lambda v: URG_FG.get(v, ""), subset=["Urgency"]),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.caption(
        "Need = 7-day sales prediction x ingredient amounts x +15% safety buffer  |  "
        "To Buy = Need minus what you already have  |  "
        "Shared ingredients (e.g. Burger bun) are combined across all dishes"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Page 3 — Analyse
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Analyse":

    st.markdown(
        f'<h1 style="color:#1e3a5f;font-weight:800;margin-bottom:.2rem;">{T("analyse_title")}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:#64748b;font-size:.88rem;margin-bottom:1.2rem;">{T("analyse_sub")}</p>',
        unsafe_allow_html=True,
    )

    with st.spinner(T("analyse_spinner")):
        model = get_model(data_hash, daily)
        pred  = run_forecast(data_hash, 30, model, daily)

    pred_future_an = pred[pred["ds"] >= fc_start].copy()

    # ── 1. Comparaison semaine en cours vs semaine dernière ───────────────────
    _sec(T("analyse_week"))

    this_week_start = last_date - pd.Timedelta(days=6)
    last_week_start = last_date - pd.Timedelta(days=13)
    last_week_end   = last_date - pd.Timedelta(days=7)

    this_week_df = daily[daily["ds"] >= this_week_start].copy()
    last_week_df = daily[(daily["ds"] >= last_week_start) & (daily["ds"] <= last_week_end)].copy()

    this_week_df["day_label"] = this_week_df["ds"].dt.day_name().str[:3]
    last_week_df["day_label"] = last_week_df["ds"].dt.day_name().str[:3]

    wk_total_cur  = int(this_week_df["y"].sum())
    wk_total_prev = int(last_week_df["y"].sum())
    wk_delta      = wk_total_cur - wk_total_prev
    wk_pct        = (wk_delta / max(wk_total_prev, 1)) * 100

    col_kw1, col_kw2, col_kw3 = st.columns(3)
    arrow = "▲" if wk_delta >= 0 else "▼"
    col_kw1.markdown(_kpi(T("analyse_cur_week"),  f"{wk_total_cur:,}",  T("analyse_cur_week_sub"),  "📅"), unsafe_allow_html=True)
    col_kw2.markdown(_kpi(T("analyse_prev_week"), f"{wk_total_prev:,}", T("analyse_prev_week_sub"), "📆"), unsafe_allow_html=True)
    col_kw3.markdown(_kpi(T("analyse_delta"),     f"{arrow} {abs(wk_pct):.1f}%", f"{arrow} {abs(wk_delta):,} {T('units')}", "📊"), unsafe_allow_html=True)

    # Bar chart side-by-side
    DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig_ww = go.Figure()
    fig_ww.add_trace(go.Bar(
        x=last_week_df["day_label"], y=last_week_df["y"],
        name=T("analyse_prev_week"), marker_color="#94a3b8",
    ))
    fig_ww.add_trace(go.Bar(
        x=this_week_df["day_label"], y=this_week_df["y"],
        name=T("analyse_cur_week"), marker_color="#14b8a6",
    ))
    fig_ww.update_layout(
        barmode="group", template="plotly_white",
        height=280, margin=dict(t=20, b=20),
        xaxis_title="", yaxis_title=T("analyse_units_sold"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_ww, use_container_width=True)

    # ── 2. Meilleure heure de vente ──────────────────────────────────────────
    _sec(T("analyse_hour"))
    has_hour = "heure" in df.columns
    if has_hour:
        df_h = df.copy()
        df_h["heure_int"] = pd.to_numeric(df_h["heure"], errors="coerce").dropna().astype(int)
        hourly = (
            df_h.groupby("heure_int")["quantite_vendue"]
            .mean().reset_index()
            .rename(columns={"heure_int": "Heure", "quantite_vendue": "Moy. unités"})
        )
        best_hour = int(hourly.loc[hourly["Moy. unités"].idxmax(), "Heure"])
        st.markdown(
            _kpi(T("analyse_best_hour"), f"{best_hour:02d}h00",
                 T("analyse_peak_avg").format(qty=hourly["Moy. unités"].max()), "🕐"),
            unsafe_allow_html=True,
        )
        fig_hour = px.bar(
            hourly, x="Heure", y="Moy. unités",
            color_discrete_sequence=["#14b8a6"],
            text_auto=".0f",
        )
        fig_hour.update_layout(
            template="plotly_white", height=260, margin=dict(t=20, b=20),
            xaxis=dict(tickmode="linear", dtick=1),
        )
        st.plotly_chart(fig_hour, use_container_width=True)
    else:
        st.info(T("analyse_no_hour"))

    # ── 3. Prévision par produit individuel ──────────────────────────────────
    _sec(T("analyse_prod"))
    if lite_mode:
        st.markdown(f'<div class="lite-banner">{T("analyse_prod_lite")}</div>',
                    unsafe_allow_html=True)
    else:
        pass  # continue below
    prod_options = sorted(df["produit"].unique().tolist()) if not lite_mode else []
    if not prod_options:
        st.warning(T("analyse_prod_empty"))
    else:
        sel_prod = st.selectbox(T("analyse_prod_pick"), prod_options, label_visibility="collapsed")
        prod_horizon = st.radio(
            T("analyse_horizon"),
            [7, 14, 30],
            horizontal=True,
            format_func=lambda x: T("analyse_horizon_fmt").format(n=x),
        )

        prod_daily = aggregate_by_product(df, sel_prod)
        if len(prod_daily) < 30:
            st.warning(T("analyse_prod_nodata").format(p=sel_prod, n=len(prod_daily)))
        else:
            with st.spinner(T("analyse_prod_spinner").format(p=sel_prod)):
                prod_model = get_product_model(data_hash, sel_prod, prod_daily)
                prod_pred  = run_product_forecast(data_hash, sel_prod, prod_horizon, prod_model, prod_daily)

            prod_future = prod_pred[prod_pred["ds"] >= fc_start].copy()

            # KPIs
            avg_prod = float(prod_daily[prod_daily["y"] > 0]["y"].mean())
            next7_total = int(prod_future.head(7)["yhat"].clip(lower=0).sum().round())
            col_p1, col_p2 = st.columns(2)
            col_p1.markdown(_kpi(T("analyse_prod_hist"), f"{avg_prod:.1f}", T("analyse_prod_units"), "📈"), unsafe_allow_html=True)
            col_p2.markdown(_kpi(T("analyse_prod_next7"), f"{next7_total:,}", T("analyse_prod_total"), "🔮"), unsafe_allow_html=True)

            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=pd.concat([prod_pred["ds"], prod_pred["ds"][::-1]]),
                y=pd.concat([prod_pred["yhat_upper"], prod_pred["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(20,184,166,.13)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip", name=T("analyse_interval"),
            ))
            fig_p.add_trace(go.Scatter(
                x=prod_pred["ds"], y=prod_pred["yhat"],
                mode="lines", line=dict(color="#14b8a6", width=2, dash="dot"),
                name=T("analyse_pred_label"),
            ))
            fig_p.add_trace(go.Scatter(
                x=prod_daily["ds"], y=prod_daily["y"],
                mode="lines", line=dict(color="#1e3a5f", width=1.2),
                name=T("analyse_hist_label"),
            ))
            fig_p.add_vline(
                x=fc_start.timestamp() * 1000,
                line=dict(color="crimson", dash="dash", width=1.5),
                annotation_text=T("analyse_fc_start"),
                annotation_position="top right",
            )
            fig_p.update_layout(
                template="plotly_white", hovermode="x unified",
                xaxis_title="", yaxis_title=T("units"),
                height=380, margin=dict(t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_p, use_container_width=True)

    # ── 4. Calendrier des fêtes marocaines ───────────────────────────────────
    _sec(T("analyse_calendar"))
    import datetime as _dt
    current_year = _dt.date.today().year
    cal_years    = [current_year, current_year + 1]

    holiday_rows = []
    for yr in cal_years:
        for month, day, name in MAROC_FIXED_HOLIDAYS:
            holiday_rows.append({
                "Date": f"{day:02d}/{month:02d}/{yr}",
                "Fête": name,
                "Type": T("analyse_cal_nat"),
            })
        for date_str, name in ISLAMIC_HOLIDAYS.get(yr, []):
            holiday_rows.append({
                "Date": pd.Timestamp(date_str).strftime("%d/%m/%Y"),
                "Fête": name,
                "Type": T("analyse_cal_isl"),
            })
        if yr in RAMADAN_DATES:
            s, e = RAMADAN_DATES[yr]
            holiday_rows.append({
                "Date": f"{pd.Timestamp(s).strftime('%d/%m/%Y')} → {pd.Timestamp(e).strftime('%d/%m/%Y')}",
                "Fête": T("analyse_cal_ram"),
                "Type": T("analyse_cal_ram"),
            })

    cal_df = pd.DataFrame(holiday_rows)

    def _style_cal(row):
        _cn, _ci, _cr = T("analyse_cal_nat"), T("analyse_cal_isl"), T("analyse_cal_ram")
        colors = {_cn: "#dbeafe", _ci: "#fef3c7", _cr: "#fce7f3"}
        return [f"background-color:{colors.get(row['Type'], '#fff')}"] * len(row)

    st.dataframe(
        cal_df.style.apply(_style_cal, axis=1),
        use_container_width=True, hide_index=True,
    )

    if ramadan_mode:
        st.info(T("analyse_ram_msg"))


# ═════════════════════════════════════════════════════════════════════════════
# Page — Finances
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Finances":

    st.markdown(
        f'<h1 style="color:#1e3a5f;font-weight:800;margin-bottom:.2rem;">{T("fin_title")}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:#64748b;font-size:.88rem;margin-bottom:1.2rem;">{T("fin_sub")}</p>',
        unsafe_allow_html=True,
    )

    price_map = st.session_state[_prices_key]
    cost_map  = st.session_state[_costs_key]

    # ── Section 1: Configuration des prix & coûts ─────────────────────────────
    with st.expander(T("fin_config"), expanded=True):
        st.caption(T("fin_config_caption"))
        if "prix_unitaire" in df.columns:
            st.success(T("fin_auto_price"))

        _cp = T("fin_col_price")
        _cc = T("fin_col_cost")
        fin_rows = [
            {
                T("fin_col_prod"): p,
                _cp: price_map.get(p, 0.0),
                _cc: cost_map.get(p, 0.0),
            }
            for p in _all_prods
        ]
        edited_fin = st.data_editor(
            pd.DataFrame(fin_rows),
            column_config={
                T("fin_col_prod"): st.column_config.TextColumn(T("fin_col_prod"), disabled=True),
                _cp: st.column_config.NumberColumn(_cp, min_value=0.0, format="%.2f", help=T("fin_col_price_help")),
                _cc: st.column_config.NumberColumn(_cc, min_value=0.0, format="%.2f", help=T("fin_col_cost_help")),
            },
            hide_index=True,
            use_container_width=True,
            key=f"fin_editor_{data_hash}",
        )
        for _, row in edited_fin.iterrows():
            price_map[row[T("fin_col_prod")]] = float(row[_cp])
            cost_map[row[T("fin_col_prod")]]  = float(row[_cc])
        st.session_state[_prices_key] = price_map
        st.session_state[_costs_key]  = cost_map

    # Guard: warn if all prices are zero
    prices_set = any(v > 0 for v in price_map.values())
    if not prices_set:
        st.warning(T("fin_warn_no_price"))

    # ── Build financial dataframe ─────────────────────────────────────────────
    df_fin = build_fin_df(df, price_map, cost_map)

    daily_fin = (
        df_fin.groupby("date")
        .agg(recette=("recette", "sum"), cout_mp=("cout_mp", "sum"),
             marge_brute=("marge_brute", "sum"), quantite=("quantite_vendue", "sum"))
        .reset_index()
    )
    daily_fin["date"] = pd.to_datetime(daily_fin["date"])

    last_day      = daily_fin["date"].max()
    last_day_row  = daily_fin[daily_fin["date"] == last_day]
    ca_jour       = float(last_day_row["recette"].sum())
    marge_jour    = float(last_day_row["marge_brute"].sum())

    week_start    = last_day - pd.Timedelta(days=6)
    ca_semaine    = float(daily_fin[daily_fin["date"] >= week_start]["recette"].sum())
    marge_semaine = float(daily_fin[daily_fin["date"] >= week_start]["marge_brute"].sum())

    month_mask    = (daily_fin["date"].dt.year  == last_day.year) & \
                   (daily_fin["date"].dt.month == last_day.month)
    ca_mois       = float(daily_fin[month_mask]["recette"].sum())
    cout_mois     = float(daily_fin[month_mask]["cout_mp"].sum())
    marge_mois    = float(daily_fin[month_mask]["marge_brute"].sum())
    marge_pct     = (marge_mois / ca_mois * 100) if ca_mois > 0 else 0.0

    # ── Section 2: KPI financiers ─────────────────────────────────────────────
    _sec(T("fin_ca_section"))
    k1, k2, k3, k4 = st.columns(4)
    for col, html in zip([k1, k2, k3, k4], [
        _fin_kpi(T("fin_kpi_day"),    _mad(ca_jour),    T("fin_margin_sub").format(m=_mad(marge_jour)), "💶"),
        _fin_kpi(T("fin_kpi_week"),   _mad(ca_semaine), T("fin_margin_sub").format(m=_mad(marge_semaine)), "📅"),
        _fin_kpi(T("fin_kpi_month"),  _mad(ca_mois),    last_day.strftime("%B %Y"), "📆"),
        _fin_kpi(T("fin_kpi_margin"), _mad(marge_mois), T("fin_margin_pct").format(pct=marge_pct), "📊"),
    ]):
        col.markdown(html, unsafe_allow_html=True)

    # ── Section 3: Objectif de CA journalier ─────────────────────────────────
    _sec(T("fin_target_section"))
    col_tgt, col_bar = st.columns([1, 2])
    with col_tgt:
        target = st.number_input(
            T("fin_target_input"),
            min_value=0.0,
            value=float(st.session_state[_target_key]),
            step=100.0,
            format="%.0f",
            label_visibility="collapsed",
        )
        st.session_state[_target_key] = target

    with col_bar:
        if target > 0:
            pct_day = ca_jour / target * 100
            bar_label = T("fin_target_above").format(pct=pct_day, ca=_mad(ca_jour), target=_mad(target))
            st.markdown(_progress_bar(pct_day, bar_label), unsafe_allow_html=True)
            days_above = int((daily_fin["recette"] >= target).sum())
            total_days = len(daily_fin)
            st.caption(T("fin_target_days").format(n=days_above, total=total_days))
        else:
            st.caption(T("fin_target_empty"))

    # Revenue trend with target line
    fig_ca = go.Figure()
    fig_ca.add_trace(go.Scatter(
        x=daily_fin["date"], y=daily_fin["recette"],
        mode="lines", fill="tozeroy",
        fillcolor="rgba(34,197,94,.10)",
        line=dict(color="#22c55e", width=1.8),
        name=T("fin_ca_series"),
    ))
    fig_ca.add_trace(go.Scatter(
        x=daily_fin["date"], y=daily_fin["cout_mp"],
        mode="lines", line=dict(color="#f97316", width=1.4, dash="dot"),
        name=T("fin_cost_series"),
    ))
    if target > 0:
        fig_ca.add_hline(
            y=target, line=dict(color="#6366f1", dash="dash", width=1.5),
            annotation_text=T("fin_target_line").format(t=_mad(target)),
            annotation_position="top left",
        )
    fig_ca.update_layout(
        template="plotly_white", hovermode="x unified",
        xaxis_title="", yaxis_title="MAD",
        height=300, margin=dict(t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_ca, use_container_width=True)

    # ── Section 4: Produit le plus rentable vs le plus vendu ─────────────────
    _sec(T("fin_prod_section"))
    prod_fin = (
        df_fin.groupby("produit")
        .agg(
            total_vendu  =("quantite_vendue", "sum"),
            total_recette=("recette",         "sum"),
            total_cout   =("cout_mp",         "sum"),
            total_marge  =("marge_brute",     "sum"),
        )
        .reset_index()
    )
    prod_fin["marge_pct"] = (
        prod_fin["total_marge"] / prod_fin["total_recette"].replace(0, np.nan) * 100
    ).fillna(0)
    prod_fin["marge_unit"] = (
        prod_fin["total_marge"] / prod_fin["total_vendu"].replace(0, np.nan)
    ).fillna(0)

    if prices_set and not prod_fin.empty:
        best_seller  = prod_fin.loc[prod_fin["total_vendu"].idxmax(), "produit"]
        best_margin  = prod_fin.loc[prod_fin["total_marge"].idxmax(), "produit"]

        bm1, bm2 = st.columns(2)
        bm1.markdown(_fin_kpi(T("fin_best_seller"), best_seller,
                              f"{int(prod_fin.loc[prod_fin['produit']==best_seller,'total_vendu'].iloc[0]):,} {T('units')}", "🏆"),
                     unsafe_allow_html=True)
        bm2.markdown(_fin_kpi(T("fin_best_margin"), best_margin,
                              _mad(prod_fin.loc[prod_fin['produit']==best_margin,'total_marge'].iloc[0]), "💰"),
                     unsafe_allow_html=True)

        # Bubble chart: volume (x) vs marge unitaire (y) vs CA total (bubble size)
        fig_bub = px.scatter(
            prod_fin,
            x="total_vendu", y="marge_unit",
            size="total_recette", color="produit",
            text="produit",
            labels={
                "total_vendu":   T("fin_bubble_x"),
                "marge_unit":    T("fin_bubble_y"),
                "total_recette": T("fin_bubble_size"),
            },
            color_discrete_sequence=["#14b8a6","#1e3a5f","#0ea5e9","#6366f1","#f59e0b","#22c55e"],
            size_max=60,
        )
        fig_bub.update_traces(textposition="top center", textfont_size=11)
        fig_bub.update_layout(
            template="plotly_white", height=340, margin=dict(t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_bub, use_container_width=True)

        # Detailed table
        disp = prod_fin.copy()
        disp["total_recette"] = disp["total_recette"].map(_mad)
        disp["total_cout"]    = disp["total_cout"].map(_mad)
        disp["total_marge"]   = disp["total_marge"].map(_mad)
        disp["marge_pct"]     = disp["marge_pct"].map(lambda x: f"{x:.1f}%")
        disp["marge_unit"]    = disp["marge_unit"].map(lambda x: f"{x:.2f} MAD")
        disp.columns = Tlist("fin_table_cols") or disp.columns.tolist()
        st.dataframe(disp.reset_index(drop=True), use_container_width=True, hide_index=True)
    else:
        st.info(T("fin_no_price_info"))

    # ── Section 5: Recettes vs dépenses matières premières par mois ──────────
    _sec(T("fin_monthly_section"))
    monthly_fin = (
        daily_fin.copy()
        .assign(mois=lambda d: d["date"].dt.to_period("M"))
        .groupby("mois")
        .agg(recette=("recette","sum"), cout_mp=("cout_mp","sum"), marge=("marge_brute","sum"))
        .reset_index()
    )
    monthly_fin["mois_label"] = monthly_fin["mois"].dt.strftime("%b %Y")

    if prices_set and not monthly_fin.empty:
        fig_mo = go.Figure()
        fig_mo.add_trace(go.Bar(
            x=monthly_fin["mois_label"], y=monthly_fin["recette"],
            name=T("fin_rev_series"), marker_color="#22c55e",
        ))
        fig_mo.add_trace(go.Bar(
            x=monthly_fin["mois_label"], y=monthly_fin["cout_mp"],
            name=T("fin_mp_series"), marker_color="#f97316",
        ))
        fig_mo.add_trace(go.Scatter(
            x=monthly_fin["mois_label"], y=monthly_fin["marge"],
            mode="lines+markers", name=T("fin_margin_series"),
            line=dict(color="#1e3a5f", width=2),
            marker=dict(size=7),
        ))
        fig_mo.update_layout(
            barmode="group", template="plotly_white",
            height=340, margin=dict(t=20, b=20),
            xaxis_title="", yaxis_title="MAD",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_mo, use_container_width=True)

        # ── Section 6: Estimation du bénéfice mensuel ────────────────────────
        _sec(T("fin_profit_section"))
        def _style_marge(val: str) -> str:
            try:
                num = float(val.replace("%",""))
                if num >= 40:
                    return "color:#166534;font-weight:700"
                if num >= 20:
                    return "color:#854d0e;font-weight:700"
                return "color:#991b1b;font-weight:700"
            except Exception:
                return ""

        _fmc = Tlist("fin_month_cols")
        _fmcols = _fmc if len(_fmc) == 5 else ["Mois","CA (MAD)","MP (MAD)","Marge (MAD)","Marge %"]
        _fm_disp = pd.DataFrame({
            _fmcols[0]: monthly_fin["mois_label"],
            _fmcols[1]: monthly_fin["recette"].map(_mad),
            _fmcols[2]: monthly_fin["cout_mp"].map(_mad),
            _fmcols[3]: monthly_fin["marge"].map(_mad),
            _fmcols[4]: (monthly_fin["marge"] / monthly_fin["recette"].replace(0, np.nan) * 100).fillna(0).map(lambda x: f"{x:.1f}%"),
        })

        st.dataframe(
            _fm_disp
            .reset_index(drop=True)
            .style.map(_style_marge, subset=[_fmcols[4]]),
            use_container_width=True,
            hide_index=True,
        )

        # Grand total row
        total_ca    = daily_fin["recette"].sum()
        total_cout  = daily_fin["cout_mp"].sum()
        total_marge = daily_fin["marge_brute"].sum()
        total_pct   = (total_marge / total_ca * 100) if total_ca > 0 else 0
        st.markdown(
            f'<div class="sum-box">'
            f'<span class="sum-big">{_mad(total_marge)}</span>'
            f'<span class="sum-sub">{T("fin_total_label")} &nbsp;|&nbsp; '
            f'{T("fin_total_sub").format(ca=_mad(total_ca), cout=_mad(total_cout), pct=total_pct)}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(T("fin_no_config_info"))

    st.divider()
    st.caption(T("fin_footer"))


# ═════════════════════════════════════════════════════════════════════════════
# Page — Tableau de bord (bilingue FR / AR)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Tableau de bord":

    _lang   = st.session_state.get("lang", "fr")
    is_ar   = _lang == "ar"
    dir_tag = 'dir="rtl"' if is_ar else ''
    wrap_cl = "rtl ar-font" if is_ar else ""

    # ── Titre ─────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="{wrap_cl}">'
        f'<h1 style="color:#1e3a5f;font-weight:800;margin-bottom:.2rem;">{T("dash_title")}</h1>'
        f'<p style="color:#64748b;font-size:.88rem;margin-bottom:1.2rem;">'
        f'{last_date.strftime("%A %d %B %Y")}</p></div>',
        unsafe_allow_html=True,
    )

    # ── Données du dernier jour ───────────────────────────────────────────────
    last_day_row  = daily[daily["ds"] == last_date]
    last_day_qty  = int(last_day_row["y"].sum()) if not last_day_row.empty else 0
    prev_date     = last_date - pd.Timedelta(days=1)
    prev_day_row  = daily[daily["ds"] == prev_date]
    prev_day_qty  = int(prev_day_row["y"].sum()) if not prev_day_row.empty else 0

    delta_vs_prev = last_day_qty - prev_day_qty
    delta_sign    = "+" if delta_vs_prev >= 0 else ""
    ratio_today   = last_day_qty / max(avg_open, 1)
    ratio_pct     = (ratio_today - 1) * 100
    ratio_sign    = "+" if ratio_pct >= 0 else ""

    # Best product of the last day
    df_last       = df[df["date"] == last_date]
    if not df_last.empty:
        best_prod_today = df_last.groupby("produit")["quantite_vendue"].sum().idxmax()
    else:
        best_prod_today = "—"

    # ── 1. Résumé du jour ─────────────────────────────────────────────────────
    _sec(T("sec_today"))
    k1, k2, k3, k4 = st.columns(4)
    for col, html in zip([k1, k2, k3, k4], [
        _kpi(T("kpi_sold"),
             f"{last_day_qty:,}",
             f"{last_date.strftime('%d %b %Y')}", "📦"),
        _kpi(T("kpi_yesterday"),
             f"{delta_sign}{delta_vs_prev:,}",
             T("kpi_yesterday_sub").format(qty=f"{prev_day_qty:,}", u=T("units")), "↕️"),
        _kpi(T("kpi_avg"),
             f"{ratio_sign}{ratio_pct:.1f}%",
             T("kpi_avg_sub").format(avg=avg_open, u=T("units")), "📊"),
        _kpi(T("kpi_best"),
             best_prod_today,
             T("kpi_best"), "🏆"),
    ]):
        col.markdown(f'<div class="{wrap_cl}">{html}</div>', unsafe_allow_html=True)

    # ── 2. Indicateur de performance ─────────────────────────────────────────
    _sec(T("sec_perf"))
    if ratio_today >= 1.15:
        p_key, p_icon, p_color, p_bg = "perf_good", "⭐", "#166534", "#dcfce7"
    elif ratio_today >= 0.85:
        p_key, p_icon, p_color, p_bg = "perf_avg",  "📊", "#854d0e", "#fef3c7"
    else:
        p_key, p_icon, p_color, p_bg = "perf_bad",  "⚠️", "#991b1b", "#fee2e2"

    c_perf, c_stats = st.columns([3, 2])
    with c_perf:
        st.markdown(
            f'<div class="perf-banner {wrap_cl}" style="background:{p_bg};color:{p_color};">'
            f'  <div class="perf-icon">{p_icon}</div>'
            f'  <div>'
            f'    <div class="perf-label">{T(p_key)}</div>'
            f'    <div class="perf-sub">{T(p_key + "_sub")}</div>'
            f'    <div class="perf-pct">'
            f'      {ratio_sign}{ratio_pct:.1f}% {T("perf_vs")}'
            f'    </div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with c_stats:
        record_day = int(daily["y"].max())
        open_days  = int((daily["y"] > 0).sum())
        for html in [
            _kpi(T("kpi_avg_day"),       f"{avg_open:.1f}", T("units"), "📈"),
            _kpi(T("kpi_best_day_ever"), f"{record_day:,}", T("units"), "🏅"),
            _kpi(T("kpi_open_days"),     str(open_days),    f"/ {len(daily)} j", "📅"),
        ]:
            st.markdown(f'<div class="{wrap_cl}" style="margin-bottom:10px;">{html}</div>',
                        unsafe_allow_html=True)

    # ── 3. Graphique des ventes de la semaine ─────────────────────────────────
    _sec(T("sec_week"))
    last7 = daily.tail(7).copy()
    last7["day_lbl"] = last7["ds"].dt.day_name().map(day_label)
    last7["color"]   = last7["y"].apply(
        lambda v: "#22c55e" if v >= avg_open * 1.15
        else ("#ef4444" if v < avg_open * 0.85 else "#f59e0b")
    )
    fig_wk = go.Figure()
    fig_wk.add_trace(go.Bar(
        x=last7["day_lbl"], y=last7["y"],
        marker_color=last7["color"].tolist(),
        text=last7["y"].round(0).astype(int),
        textposition="outside",
        name=T("kpi_sold"),
    ))
    fig_wk.add_hline(
        y=avg_open, line=dict(color="#94a3b8", dash="dash", width=1.5),
        annotation_text=f"  avg {avg_open:.0f}" if _lang == "en" else f"  moy. {avg_open:.0f}",
        annotation_position="right",
    )
    fig_wk.update_layout(
        template="plotly_white", height=280,
        margin=dict(t=20, b=10), showlegend=False,
        xaxis_title="", yaxis_title=T("units"),
        yaxis=dict(range=[0, max(last7["y"].max() * 1.2, avg_open * 1.4)]),
    )
    st.plotly_chart(fig_wk, use_container_width=True)

    # ── 4. Classement des produits ────────────────────────────────────────────
    _sec(T("sec_ranking"))
    prod_totals = (
        df.groupby("produit")["quantite_vendue"].sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    grand_total = prod_totals["quantite_vendue"].sum()

    # Trend: last 30d vs previous 30d
    cutoff_30 = last_date - pd.Timedelta(days=30)
    cutoff_60 = last_date - pd.Timedelta(days=60)
    recent_30 = df[df["date"] > cutoff_30].groupby("produit")["quantite_vendue"].sum()
    prev_30   = df[(df["date"] > cutoff_60) & (df["date"] <= cutoff_30)
                   ].groupby("produit")["quantite_vendue"].sum()

    MEDALS = ["🥇", "🥈", "🥉"]

    rank_html = '<div style="margin-top:.4rem;">'
    for i, row in prod_totals.iterrows():
        medal = MEDALS[i] if i < 3 else f"<span style='color:#94a3b8;font-weight:700;font-size:.9rem;'>{i+1}</span>"
        share = row["quantite_vendue"] / max(grand_total, 1) * 100
        r30 = float(recent_30.get(row["produit"], 0))
        p30 = float(prev_30.get(row["produit"],   0))
        trend = "↗" if r30 > p30 * 1.05 else ("↘" if r30 < p30 * 0.95 else "→")
        trend_col = "#22c55e" if trend == "↗" else ("#ef4444" if trend == "↘" else "#94a3b8")
        rank_html += (
            f'<div class="rank-row {wrap_cl}">'
            f'  <div class="rank-medal">{medal}</div>'
            f'  <div class="rank-name">{row["produit"]}</div>'
            f'  <div class="rank-qty">{int(row["quantite_vendue"]):,} {T("units")}</div>'
            f'  <div class="rank-share">{share:.1f}%</div>'
            f'  <div class="rank-trend" style="color:{trend_col};font-weight:700;">{trend}</div>'
            f'</div>'
        )
    rank_html += "</div>"
    st.markdown(rank_html, unsafe_allow_html=True)
    st.caption(T("rank_caption"))

    # ── 5. Historique 30 derniers jours ──────────────────────────────────────
    _sec(T("sec_history"))
    hist30 = daily.tail(30).copy()
    hist30["day_name"] = hist30["ds"].dt.day_name().map(day_label)
    hist30["vs_avg"]   = ((hist30["y"] / max(avg_open, 1) - 1) * 100).round(1)
    hist30["perf"]     = hist30["y"].apply(
        lambda v: T("perf_good") if v >= avg_open * 1.15
        else (T("perf_bad") if v < avg_open * 0.85 else T("perf_avg"))
    )

    hist_disp = hist30[["ds", "day_name", "y", "vs_avg", "perf"]].copy()
    hist_disp["ds"]     = hist_disp["ds"].dt.strftime("%Y-%m-%d")
    hist_disp["y"]      = hist_disp["y"].astype(int)
    hist_disp["vs_avg"] = hist_disp["vs_avg"].map(lambda x: f"{'+' if x>=0 else ''}{x:.1f}%")
    hist_disp.columns   = [
        T("hist_date"), T("hist_day"), T("kpi_sold"), T("hist_vs_avg"), T("hist_perf")
    ]

    PERF_BG = {
        T("perf_good"): "#dcfce7",
        T("perf_avg"):  "#fef3c7",
        T("perf_bad"):  "#fee2e2",
    }
    PERF_FG = {
        T("perf_good"): "color:#166534;font-weight:700",
        T("perf_avg"):  "color:#854d0e;font-weight:700",
        T("perf_bad"):  "color:#991b1b;font-weight:700",
    }

    perf_col = T("hist_perf")

    def _style_hist(row):
        return [f"background-color:{PERF_BG.get(row[perf_col], '#fff')}"] * len(row)

    st.dataframe(
        hist_disp.reset_index(drop=True)
        .style
        .apply(_style_hist, axis=1)
        .map(lambda v: PERF_FG.get(v, ""), subset=[perf_col]),
        use_container_width=True,
        hide_index=True,
    )

    # Mini area chart of 30-day history
    fig_h30 = go.Figure()
    fig_h30.add_trace(go.Scatter(
        x=hist30["ds"], y=hist30["y"],
        mode="lines+markers", fill="tozeroy",
        fillcolor="rgba(20,184,166,.09)",
        line=dict(color="#14b8a6", width=1.6),
        marker=dict(
            color=hist30["y"].apply(
                lambda v: "#22c55e" if v >= avg_open * 1.15
                else ("#ef4444" if v < avg_open * 0.85 else "#f59e0b")
            ).tolist(),
            size=6,
        ),
        name=T("kpi_sold"),
    ))
    fig_h30.add_hline(
        y=avg_open, line=dict(color="#94a3b8", dash="dash", width=1.2),
        annotation_text=f"  moy.", annotation_position="right",
    )
    fig_h30.update_layout(
        template="plotly_white", height=220, showlegend=False,
        margin=dict(t=10, b=10), xaxis_title="", yaxis_title=T("units"),
    )
    st.plotly_chart(fig_h30, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Page 5 — About
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Conseils":
    import datetime as _dtc

    _mlang  = st.session_state.get("lang", "fr")
    rtl_cls = ' rtl ar-font' if _mlang == "ar" else ""

    st.markdown(
        f'<div class="conseil-hero{rtl_cls}">'
        f'<div class="conseil-hero-title">🎯 {T("conseils_hero_title")}</div>'
        f'<div class="conseil-hero-sub">{T("conseils_hero_sub")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    _today_c   = pd.Timestamp(_dtc.date.today())
    _last_date = daily["ds"].max()
    _ref_date  = _last_date   # use last known date as "today" for the bakery demo

    # ── Pre-compute per-product daily aggregates ──────────────────────────────
    _prod_daily = (
        df.groupby(["date", "produit"])["quantite_vendue"]
        .sum().reset_index()
    )
    _prod_daily["dow"] = _prod_daily["date"].dt.day_name()  # English day name

    # Overall per-product mean (all data)
    _prod_global_avg = (
        _prod_daily.groupby("produit")["quantite_vendue"].mean()
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — Produit du jour
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="conseil-section">'
        f'<div class="conseil-section-title">{T("conseils_pdj")}</div>',
        unsafe_allow_html=True,
    )

    _dow_today  = _ref_date.day_name()   # e.g. "Monday"

    # Score = 0.4 × (dow_ratio) + 0.4 × (trend_ratio) + 0.2 × (margin_ratio)
    _price_map = st.session_state.get(f"prices_{data_hash}", {})
    _cost_map  = st.session_state.get(f"costs_{data_hash}",  {})

    _scores: dict[str, float] = {}
    for _p in active_csv:
        # 1. Day-of-week performance ratio
        _dow_data = _prod_daily[
            (_prod_daily["produit"] == _p) & (_prod_daily["dow"] == _dow_today)
        ]["quantite_vendue"]
        _dow_avg   = float(_dow_data.mean()) if len(_dow_data) > 0 else 0.0
        _global_p  = float(_prod_global_avg.get(_p, 1))
        _dow_ratio = _dow_avg / max(_global_p, 1)

        # 2. 30-day trend ratio (last 30 d vs previous 30 d)
        _cut1 = _last_date - pd.Timedelta(days=30)
        _cut0 = _last_date - pd.Timedelta(days=60)
        _last30 = float(
            _prod_daily[(_prod_daily["produit"] == _p) &
                        (_prod_daily["date"] > _cut1)]["quantite_vendue"].mean() or 0
        )
        _prev30 = float(
            _prod_daily[(_prod_daily["produit"] == _p) &
                        (_prod_daily["date"] > _cut0) &
                        (_prod_daily["date"] <= _cut1)]["quantite_vendue"].mean() or 1
        )
        _trend_ratio = _last30 / max(_prev30, 1)

        # 3. Margin ratio (normalised over all products)
        _px = _price_map.get(_p, 0.0)
        _cx = _cost_map.get(_p, 0.0)
        _margin = max(_px - _cx, 0.0)
        _max_margin = max(
            [max(_price_map.get(q, 0) - _cost_map.get(q, 0), 0) for q in active_csv],
            default=1,
        )
        _margin_ratio = _margin / max(_max_margin, 1)

        _scores[_p] = round(0.4 * _dow_ratio + 0.4 * _trend_ratio + 0.2 * _margin_ratio, 4)

    _pdj = max(_scores, key=lambda p: _scores[p]) if _scores else None

    if _pdj:
        _dow_day_fr = day_label(_dow_today)

        _pdj_dow_qty = float(
            _prod_daily[(_prod_daily["produit"] == _pdj) &
                        (_prod_daily["dow"] == _dow_today)]["quantite_vendue"].mean() or 0
        )
        _pdj_score_pct = int(_scores[_pdj] * 100)

        _margin_info = ""
        if _price_map.get(_pdj, 0) > 0 and _cost_map.get(_pdj, 0) > 0:
            _m = _price_map[_pdj] - _cost_map[_pdj]
            _margin_info = T("conseils_pdj_margin").format(m=_m)

        st.markdown(
            f'<div class="produit-du-jour">'
            f'<div style="font-size:2.6rem">🏆</div>'
            f'<div class="pdj-name">{_pdj}</div>'
            f'<div class="pdj-score">'
            f'{T("conseils_pdj_score").format(s=_pdj_score_pct, d=_dow_day_fr, qty=_pdj_dow_qty, margin=_margin_info)}'
            f'</div>'
            f'<div class="pdj-badge">{T("conseils_pdj_badge")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Mini ranking (top 5)
        _top5 = sorted(_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        _cols_pdj = st.columns(len(_top5))
        for _i, (_p, _s) in enumerate(_top5):
            _medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][_i]
            _cols_pdj[_i].markdown(
                f'<div style="text-align:center;background:#f8fafc;border-radius:12px;'
                f'padding:10px 6px;box-shadow:0 1px 4px rgba(0,0,0,.05);">'
                f'<div style="font-size:1.4rem">{_medal}</div>'
                f'<div style="font-weight:700;color:#1e3a5f;font-size:.82rem;'
                f'margin-top:4px;">{_p}</div>'
                f'<div style="font-size:.75rem;color:#94a3b8;">{int(_s*100)}/100</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info(T("conseils_pdj_empty"))
    st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — Prépare plus ce [jour]
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="conseil-section">'
        f'<div class="conseil-section-title">{T("conseils_prep")}</div>',
        unsafe_allow_html=True,
    )

    _next7_days = [_ref_date + pd.Timedelta(days=i+1) for i in range(7)]

    for _d in _next7_days:
        _dn = _d.day_name()
        _d_fr = day_label(_dn)
        _date_str = _d.strftime("%d/%m/%Y")

        # For each product: ratio of this-DOW avg vs overall avg
        _prep_items = []
        for _p in active_csv:
            _dow_vals = _prod_daily[
                (_prod_daily["produit"] == _p) & (_prod_daily["dow"] == _dn)
            ]["quantite_vendue"]
            if len(_dow_vals) < 2:
                continue
            _avg_dow = float(_dow_vals.mean())
            _avg_all = float(_prod_global_avg.get(_p, 1))
            _ratio = _avg_dow / max(_avg_all, 1)
            if _ratio >= 1.10:   # at least 10% above average for this product
                _prep_items.append((_p, _avg_dow, _ratio))

        _is_weekend = _dn in WEEKEND_DAYS
        _hol_name   = _get_holiday_name(_d)
        _is_ram     = is_ramadan_period(_d)

        _tags = []
        if _is_weekend: _tags.append("Week-end")
        if _hol_name:   _tags.append(_hol_name)
        if _is_ram:     _tags.append("Ramadan")

        _tag_html = "".join(
            f'<span style="background:#e0f2fe;color:#0369a1;border-radius:12px;'
            f'padding:2px 10px;font-size:.72rem;font-weight:700;margin-left:6px;">'
            f'{t}</span>' for t in _tags
        )

        with st.expander(f"**{_d_fr}** {_date_str}{' 🔥' if _tags else ''}", expanded=_is_weekend or bool(_hol_name)):
            if _tag_html:
                st.markdown(_tag_html, unsafe_allow_html=True)
                st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

            _prep_items_sorted = sorted(_prep_items, key=lambda x: x[2], reverse=True)
            if _prep_items_sorted:
                for _p, _qty, _ratio in _prep_items_sorted[:6]:
                    _qty_buffered = int(_qty * BUFFER_RATE)
                    _ratio_pct = int((_ratio - 1) * 100)
                    st.markdown(
                        f'<div class="prep-row">'
                        f'<span style="font-size:1.2rem">🥐</span>'
                        f'<span class="prep-prod">{_p}</span>'
                        f'<span class="prep-qty">{_qty_buffered} {T("units")}</span>'
                        f'<span class="prep-ratio">+{_ratio_pct}% {T("hist_vs_avg")}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f'<p style="color:#94a3b8;font-size:.85rem;margin:4px 0;">{T("conseils_prep_normal")}</p>',
                    unsafe_allow_html=True,
                )
    st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — Promotions : produits en baisse
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="conseil-section">'
        f'<div class="conseil-section-title">{T("conseils_promo")}</div>',
        unsafe_allow_html=True,
    )

    _promo_thresh = st.slider(
        T("conseils_promo_thresh"),
        min_value=5, max_value=40, value=15, step=5,
        help=T("conseils_promo_help")
    )

    _cut7  = _last_date - pd.Timedelta(days=7)
    _cut30 = _last_date - pd.Timedelta(days=30)

    _declining = []
    for _p in active_csv:
        _last7_avg = float(
            _prod_daily[(_prod_daily["produit"] == _p) &
                        (_prod_daily["date"] > _cut7)]["quantite_vendue"].mean() or 0
        )
        _last30_avg = float(
            _prod_daily[(_prod_daily["produit"] == _p) &
                        (_prod_daily["date"] > _cut30) &
                        (_prod_daily["date"] <= _cut7)]["quantite_vendue"].mean() or 0
        )
        if _last30_avg > 0:
            _drop_pct = (_last30_avg - _last7_avg) / _last30_avg * 100
            if _drop_pct >= _promo_thresh:
                _declining.append((_p, _drop_pct, _last7_avg, _last30_avg))

    if _declining:
        _declining.sort(key=lambda x: x[1], reverse=True)
        for _p, _drop, _l7, _l30 in _declining:
            _px = _price_map.get(_p, 0.0)
            # Discount: 10% base + 1% per extra 5% drop, capped at 30%
            _disc_pct = min(10 + int((_drop - _promo_thresh) / 5), 30)
            _promo_px = round(_px * (1 - _disc_pct / 100), 1) if _px > 0 else None

            _price_str = (
                f'<span class="promo-price">{_promo_px:.1f} MAD</span>'
                f'<span class="promo-pct">-{_disc_pct}%</span>'
            ) if _promo_px else (
                f'<span class="promo-pct" style="color:#94a3b8;">{T("conseils_promo_config")}</span>'
            )

            st.markdown(
                f'<div class="promo-row">'
                f'<span style="font-size:1.3rem">📉</span>'
                f'<span class="promo-prod">{_p} <span style="font-size:.78rem;'
                f'color:#92400e;font-weight:400;">—{_drop:.0f}% {"in 7d" if _mlang == "en" else "في 7أ" if _mlang == "ar" else "en 7j"}</span></span>'
                f'{_price_str}'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<p style="font-size:.78rem;color:#94a3b8;margin-top:8px;">'
            f'{T("conseils_promo_note")}</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;'
            f'padding:12px 16px;color:#166534;font-size:.88rem;font-weight:600;">'
            f'{T("conseils_promo_ok")}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — Alertes : produits dormants
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="conseil-section">'
        f'<div class="conseil-section-title">{T("conseils_dormant")}</div>',
        unsafe_allow_html=True,
    )

    _dormant_days = st.slider(
        T("conseils_dormant_slider"), min_value=3, max_value=30, value=7,
        help=T("conseils_dormant_help")
    )

    _last_sold = _prod_daily.groupby("produit")["date"].max()
    _dormant = []
    for _p in active_csv:
        if _p in _last_sold.index:
            _days_since = (_last_date - _last_sold[_p]).days
            if _days_since >= _dormant_days:
                _dormant.append((_p, _days_since, _last_sold[_p]))
        else:
            _dormant.append((_p, 999, None))

    if _dormant:
        _dormant.sort(key=lambda x: x[1], reverse=True)
        for _p, _ds, _ls in _dormant:
            if _ls is not None:
                _detail_str = T("conseils_dormant_since").format(
                    d=_ls.strftime("%d/%m/%Y"), n=_ds
                )
            else:
                _detail_str = T("conseils_dormant_never").format(n=_dormant_days)
            _severity = "🔴" if _ds >= 14 else "🟠"
            st.markdown(
                f'<div class="alert-row">'
                f'<span class="alert-icon">{_severity}</span>'
                f'<div class="alert-text">'
                f'<div class="alert-prod">{_p}</div>'
                f'<div class="alert-detail">{_detail_str}</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;'
            f'padding:12px 16px;color:#166534;font-size:.88rem;font-weight:600;">'
            f'{T("conseils_dormant_ok")}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5 — Idées saisonnières
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="conseil-section">'
        f'<div class="conseil-section-title">{T("conseils_seasonal")}</div>',
        unsafe_allow_html=True,
    )

    # Hardcoded Moroccan seasonal suggestion matrix
    _SEASONAL: dict[str, list[tuple[str, str]]] = {
        "ramadan": [
            ("Chebakia", "Biscuit sésame-miel, indispensable pour l'Iftar"),
            ("Briwat", "Feuilletés amande ou viande, très demandés ce mois"),
            ("Harira", "Soupe traditionnelle de rupture du jeûne"),
            ("Kaab ghzal", "Cornes de gazelle aux amandes, best-seller Ramadan"),
            ("Sellou / Sfouf", "Pâte énergétique, ventes x5 en Ramadan"),
        ],
        1:  [   # Janvier
            ("Msemen garni", "Version fourrée pour le froid hivernal"),
            ("Brioche cannelle", "Réconfortant, idéal pour les matins froids"),
        ],
        2:  [
            ("Saint-Valentin cupcakes", "Opportunité commerciale : décor cœur/rouge"),
            ("Mille-feuille rose", "Spécial février, couleurs romantiques"),
        ],
        3:  [
            ("Tarte citron meringuée", "Citron de saison, printanier"),
            ("Macaron café", "Tendance café en mars"),
        ],
        4:  [
            ("Tarte aux abricots", "Abricots marocains de printemps"),
            ("Pain à la tomate séchée", "Saveurs méditerranéennes printanières"),
        ],
        5:  [
            ("Strawberry shortcake", "Fraises de saison en plein pic"),
            ("Salade de fruits en verrine", "Fraîcheur pour la chaleur montante"),
        ],
        6:  [
            ("Glaces maison", "Forte demande dès le début de l'été"),
            ("Pain baguette fines herbes", "Barbecues et pique-niques d'été"),
        ],
        7:  [
            ("Granola artisanal", "Vacances — clientèle touristique"),
            ("Cake marbré", "Emporter facilement en voyage"),
        ],
        8:  [
            ("Sandwich baguette premium", "Haute saison touristique"),
            ("Tarte tatin pomme", "Précurseur automne, différenciation"),
        ],
        9:  [
            ("Pain aux noix", "Noix de saison, rentrée"),
            ("Financier amande", "Rentrée scolaire — goûter premium"),
        ],
        10: [
            ("Pain potiron", "Tendance Halloween, différenciation"),
            ("Tarte poire-chocolat", "Poires de saison"),
        ],
        11: [
            ("Pain de mie brioché", "Confort hivernal"),
            ("Bûche aux marrons", "Anticipation Noël — avance concurrence"),
        ],
        12: [
            ("Bûche de Noël", "Clientèle internationale et modernité"),
            ("Pain d'épices", "Ventes boostées fêtes de fin d'année"),
            ("Bredele / sablés décorés", "Clientèle expat, cadeaux festifs"),
        ],
    }

    _cur_month    = _ref_date.month
    _is_ram_now   = is_ramadan_period(_ref_date)
    _upcoming_ram = any(
        pd.Timestamp(s) <= _ref_date + pd.Timedelta(days=30)
        for s, _ in RAMADAN_DATES.values()
        if pd.Timestamp(s) >= _ref_date
    )

    _season_blocks: list[tuple[str, str, list]] = []  # (emoji, title, ideas)

    _month_names_all = {
        "fr": {1:"Janvier",2:"Février",3:"Mars",4:"Avril",5:"Mai",6:"Juin",
               7:"Juillet",8:"Août",9:"Septembre",10:"Octobre",11:"Novembre",12:"Décembre"},
        "ar": {1:"يناير",2:"فبراير",3:"مارس",4:"أبريل",5:"مايو",6:"يونيو",
               7:"يوليو",8:"أغسطس",9:"سبتمبر",10:"أكتوبر",11:"نوفمبر",12:"ديسمبر"},
        "en": {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
               7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"},
    }
    _month_name = _month_names_all.get(_mlang, _month_names_all["fr"])

    # Ramadan block (current or within 30 days)
    if _is_ram_now:
        _season_blocks.append(("🌙", T("conseils_ram_current"), _SEASONAL["ramadan"]))
    elif _upcoming_ram:
        _season_blocks.append(("🌙", T("conseils_ram_soon"), _SEASONAL["ramadan"]))

    # Current month
    _cur_ideas = _SEASONAL.get(_cur_month, [])
    if _cur_ideas:
        _season_blocks.append(("📅", T("conseils_month_now").format(m=_month_name[_cur_month]), _cur_ideas))

    # Next month
    _next_month = (_cur_month % 12) + 1
    _next_ideas = _SEASONAL.get(_next_month, [])
    if _next_ideas:
        _season_blocks.append(("⏭️", T("conseils_month_next").format(m=_month_name[_next_month]), _next_ideas))

    if _season_blocks:
        _n_cols = min(len(_season_blocks), 3)
        _scols  = st.columns(_n_cols)
        for _i, (_ico, _title, _ideas) in enumerate(_season_blocks):
            _col = _scols[_i % _n_cols]
            _ideas_html = "".join(
                f'<div class="season-idea">'
                f'<span class="season-idea-name">• {name}</span>'
                f'</div>'
                f'<div style="font-size:.76rem;color:#64748b;margin-bottom:6px;padding-left:10px;">{why}</div>'
                for name, why in _ideas
            )
            _col.markdown(
                f'<div class="season-card">'
                f'<div class="season-card-title">{_ico} {_title}</div>'
                f'{_ideas_html}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info(T("conseils_seasonal_empty"))

    st.markdown(
        f'<p style="font-size:.75rem;color:#94a3b8;margin-top:10px;">'
        f'{T("conseils_seasonal_note")}</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "About":

    # Hero
    st.markdown("""
    <div class="hero">
      <h1>FoodCast</h1>
      <div class="tag">Predict &nbsp;&middot;&nbsp; Decide &nbsp;&middot;&nbsp; Grow</div>
      <div class="desc">
        AI-powered sales forecasting designed for small food businesses.
        Upload your sales history, get accurate demand forecasts, and make
        smarter stocking decisions &mdash; in minutes, not spreadsheets.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # What & Who
    col_what, col_who = st.columns(2)
    with col_what:
        st.markdown("""
        <div style="background:#fff;border-radius:14px;padding:24px;
                    box-shadow:0 2px 12px rgba(0,0,0,.06);height:100%;">
          <div style="font-size:1.05rem;font-weight:700;color:#1e3a5f;margin-bottom:10px;">
            What is FoodCast?
          </div>
          <p style="color:#475569;font-size:.88rem;line-height:1.65;">
            FoodCast is a smart dashboard that turns your daily sales data
            into useful predictions. It automatically spots patterns in your
            sales history and tells you exactly what to prepare and buy
            &mdash; no technical skills needed.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with col_who:
        st.markdown("""
        <div style="background:#fff;border-radius:14px;padding:24px;
                    box-shadow:0 2px 12px rgba(0,0,0,.06);height:100%;">
          <div style="font-size:1.05rem;font-weight:700;color:#1e3a5f;margin-bottom:10px;">
            Who is it for?
          </div>
          <p style="color:#475569;font-size:.88rem;line-height:1.65;">
            Designed for <b>small restaurant owners</b>, <b>food truck operators</b>,
            <b>catering managers</b>, and any food business that tracks daily product
            sales and wants to reduce waste, avoid stockouts, and plan staffing more
            efficiently.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # How it works
    st.markdown(
        '<div class="sec-head">How it works</div>',
        unsafe_allow_html=True,
    )
    s1, s2, s3 = st.columns(3)
    for col, num, title, desc in [
        (s1, "1", "Upload your data",
         "Drag and drop a CSV file with your daily sales history. "
         "The only columns needed are <code>date</code>, <code>produit</code>, "
         "and <code>quantite_vendue</code>. No account or login required."),
        (s2, "2", "See your patterns",
         "FoodCast automatically spots your busiest days, seasonal peaks, "
         "and best-selling dishes. Everything is shown as simple charts "
         "and summaries on the Overview page."),
        (s3, "3", "Prepare with confidence",
         "The Recommendations page turns predictions into a clear daily "
         "plan with a built-in safety buffer, color-coded risk levels, and "
         "alerts for busy days &mdash; so you never run short."),
    ]:
        col.markdown(
            f'<div class="step-card">'
            f'<div class="step-num">{num}</div>'
            f'<div class="step-title">{title}</div>'
            f'<div class="step-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # CSV format & download
    st.markdown('<div class="sec-head">Your Sales File Format</div>', unsafe_allow_html=True)
    fmt_col, dl_col = st.columns([3, 1])
    with fmt_col:
        st.markdown("""
        Your file must be a CSV with these three columns (exact names):

        | Column | What it means | Example |
        |---|---|---|
        | `date` | The date of the sale | `2024-01-15` |
        | `produit` | The dish or product name | `Pizza Margherita` |
        | `quantite_vendue` | How many units sold | `28` |

        One row per dish per day. Days with no sales (closed days) are handled automatically.
        """)
        st.markdown(
            '<span class="info-pill">Minimum 30 days</span>'
            '<span class="info-pill">Any number of products</span>'
            '<span class="info-pill">UTF-8 encoding</span>',
            unsafe_allow_html=True,
        )
    with dl_col:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.download_button(
            label="Download sample CSV",
            data=make_sample_csv(),
            file_name="sample_ventes.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Tech stack
    st.markdown('<div class="sec-head">Built With</div>', unsafe_allow_html=True)
    st.markdown(
        '<span class="info-pill">Python 3.12</span>'
        '<span class="info-pill">Streamlit</span>'
        '<span class="info-pill">AI forecasting (Prophet)</span>'
        '<span class="info-pill">Interactive charts (Plotly)</span>'
        '<span class="info-pill">Pandas</span>'
        '<span class="info-pill">NumPy</span>',
        unsafe_allow_html=True,
    )
