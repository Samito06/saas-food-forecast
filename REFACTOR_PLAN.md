# REFACTOR_PLAN — FoodCast UX/UI Refonte

> Statut : **En attente de validation** — aucun code Python n'a été touché.  
> Dernière mise à jour : 2026-04-29

---

## 1. Mapping ancien onglet → nouveau

| Ancienne page (key sidebar) | Lignes actuelles | Nouveau logement | Sous-tab |
|---|---|---|---|
| Tableau de bord | 3203–3429 | **Aujourd'hui** | Tab principal |
| Conseils | 3435–3892 | **Aujourd'hui** | Tab interne : `st.tabs` |
| Saisie (formulaire) | 1679–1819 | **Aujourd'hui** | `st.dialog` "Ajouter des ventes" |
| Saisie (historique + export) | 1764–1818 | **Aujourd'hui** | Expander replié sous la page |
| Forecast | 1952–2111 | **Prévisions** | Tab interne : Prévision globale |
| Recommendations | 2117–2246 | **Prévisions** | Tab interne : Stock à préparer |
| Shopping List | 2403–2696 | **Prévisions** | Tab interne : Liste de courses |
| Analyse (prévision par produit) | 2792–2858 | **Prévisions** | Tab interne : Par produit |
| Finances | 2907–3198 | **Finances** | Sections verticales, pas de tab |
| Overview | 1821–1946 | **Analyses** | Tab interne : Vue d'ensemble |
| Analyse (semaine vs semaine) | 2720–2760 | **Analyses** | Tab interne : Tendances |
| Analyse (heure de pointe) | 2762–2790 | **Analyses** | Tab interne : Tendances (suite) |
| Analyse (calendrier fêtes) | 2860–2902 | **Analyses** | Tab interne : Tendances (suite) |
| Products | 2252–2397 | **Analyses** | Tab interne : Catalogue |
| About | 3894–4017 | Sidebar footer | `st.expander` (non compté comme onglet) |

---

## 2. Structure de navigation cible

```
Sidebar
├── 🥐 FoodCast / Predict · Decide · Grow
├── [sélecteur langue : fr / ar / en]
├── [upload CSV]
├── ─────────────────
├── ● Aujourd'hui          ← onglet 1 (défaut)
├── ○ Prévisions           ← onglet 2
├── ○ Finances             ← onglet 3
├── ○ Analyses             ← onglet 4
├── ─────────────────
├── [Ramadan mode toggle]
├── [Lite mode toggle]
└── v1.3 | About ▸         ← st.expander footer
```

### Onglet 1 — Aujourd'hui

```
[+ Ajouter des ventes]     ← bouton primary en haut à droite → st.dialog

Résumé du jour             ← _sec()  (4 KPIs : unités, vs hier, vs moy, meilleur produit)
Indicateur de performance  ← perf banner coloré
Semaine en cours           ← bar chart 7j (déjà dans Dashboard)

─── st.tabs ───────────────────────────────────────────
Tab A : Classement         ← ranking produits + historique 30j (de Dashboard)
Tab B : Conseils           ← PDJ · Prépare plus · Promos · Dormants · Saisonnier
───────────────────────────────────────────────────────

[Historique des saisies]   ← st.expander replié
```

### Onglet 2 — Prévisions

```
─── st.tabs ───────────────────────────────────────────
Tab A : Prévision globale  ← Forecast (horizon, badge MAE, chart, alertes, tableaux)
Tab B : Stock à préparer   ← Recommendations (résumé box, alertes, tableau coloré)
Tab C : Par produit        ← Analyse > prévision produit individuel (sélecteur + chart)
Tab D : Liste de courses   ← Shopping List (recettes, stock, besoins, à commander, ruptures)
───────────────────────────────────────────────────────
```

### Onglet 3 — Finances

```
[config prix/coûts — expander expanded par défaut]

KPIs CA (4 colonnes)
Objectif journalier + barre de progression
Graphique CA + coûts MP

[Rentabilité produits]     ← st.expander (bubble chart + tableau)
[P&L mensuel]              ← st.expander
```

### Onglet 4 — Analyses

```
─── st.tabs ───────────────────────────────────────────
Tab A : Vue d'ensemble     ← Overview (santé données, répartition, tendance filtrable)
Tab B : Tendances          ← Semaine vs semaine · Heure de pointe · Calendrier fêtes
Tab C : Catalogue          ← Products (actif/inactif, ajout, suppression)
───────────────────────────────────────────────────────
```

### Onboarding (aucun CSV chargé, page Aujourd'hui)

```
┌──────────────────────────────────────────────────────┐
│  🥐                                                   │
│  Bienvenue sur FoodCast                               │
│  Importez vos ventes pour voir vos prévisions         │
│                                                       │
│  [📂 Charger un CSV]   [▶ Utiliser les données démo] │
└──────────────────────────────────────────────────────┘
```

Si "Utiliser les données démo" est cliqué → charge `data/boulangerie.csv` sans upload réel
(via `st.session_state["use_demo"] = True` + `st.rerun()`).

---

## 3. Arborescence cible des fichiers

```
app/
├── main.py              ← routeur léger : sidebar + dispatch vers pages/
├── constants.py         ← TRANS, RECIPES, MAROC_FIXED_HOLIDAYS,
│                           ISLAMIC_HOLIDAYS, RAMADAN_DATES,
│                           REQUIRED_COLS, WEEKEND_DAYS, BUFFER_RATE,
│                           PEAK_THRESH, UNIT_OPTIONS
├── i18n.py              ← T(), Tlist(), Tdays(), day_label(), EN_DAYS_IDX
├── ui.py                ← _kpi(), _sec(), _fin_kpi(), _progress_bar(),
│                           _mad(), load_css()
├── pages/
│   ├── today.py         ← page Aujourd'hui (dashboard + conseils + dialog saisie)
│   ├── forecast.py      ← page Prévisions (4 sous-tabs)
│   ├── finance.py       ← page Finances
│   └── analytics.py     ← page Analyses (3 sous-tabs)
└── logic/
    ├── data_loading.py  ← load_and_validate(), aggregate_daily(),
    │                       aggregate_by_product(), check_data_health(),
    │                       make_sample_csv()
    ├── forecasting.py   ← get_model(), run_forecast(), get_product_model(),
    │                       run_product_forecast(), compute_mae(),
    │                       get_moroccan_holidays(), get_upcoming_peaks(),
    │                       is_ramadan_period(), _get_holiday_name()
    ├── finance.py       ← build_fin_df()
    ├── insights.py      ← score produit du jour, détection promos,
    │                       détection dormants, _risk_level()
    └── inventory.py     ← compute_stockout_date()
```

**Règles d'import :**
- `pages/*.py` importe depuis `app.constants`, `app.i18n`, `app.ui`, `app.logic.*`
- `main.py` importe depuis `app.pages.*` uniquement
- Aucune import circulaire

---

## 4. Nouvelles clés TRANS à ajouter (i18n des 5 pages anglaises)

### Onglet principal nav

| Clé | FR | AR | EN |
|---|---|---|---|
| `nav_today` | Aujourd'hui | اليوم | Today |
| `nav_forecasts` | Prévisions | التوقعات | Forecasts |
| `nav_analytics` | Analyses | التحليلات | Analytics |

> `nav_finances` existe déjà dans TRANS.

### Overview (actuellement 100% anglais hardcodé)

| Clé | FR | AR | EN |
|---|---|---|---|
| `overview_title` | Vue d'ensemble | نظرة عامة | Overview |
| `overview_sub` | {n} lignes · {start} → {end} | {n} صف · {start} → {end} | {n} rows · {start} → {end} |
| `overview_kpi_total` | Total vendu | إجمالي المبيعات | Total sold |
| `overview_kpi_best` | Meilleur produit | أفضل منتج | Best product |
| `overview_kpi_avg` | Moy. journalière | المعدل اليومي | Daily avg |
| `overview_kpi_busiest` | Jour le + chargé | أكثر الأيام ازدحاماً | Busiest day |
| `overview_trend` | Tendance des ventes | اتجاه المبيعات | Sales trend |
| `overview_filter` | Filtrer la période | تصفية الفترة | Filter period |
| `overview_breakdown` | Répartition produits | توزيع المنتجات | Product breakdown |
| `overview_health` | Santé des données | صحة البيانات | Data health |
| `overview_health_ok` | OK | موافق | OK |
| `overview_health_warn` | Avertissement | تحذير | Warning |
| `overview_health_err` | Erreur | خطأ | Error |

### Recommendations (page Stock à préparer)

| Clé | FR | AR | EN |
|---|---|---|---|
| `reco_title` | Stock à préparer | المخزون للتحضير | Stock to prepare |
| `reco_sub` | 7 prochains jours · buffer +15% | 7 أيام القادمة · هامش +15% | Next 7 days · +15% buffer |
| `reco_summary` | {total} unités à préparer cette semaine | {total} وحدة للتحضير هذا الأسبوع | {total} units to prepare this week |
| `reco_summary_sub` | prévision : {pred} + 15% buffer | توقع : {pred} + 15% هامش | forecast: {pred} + 15% buffer |
| `reco_plan` | Plan journalier | الخطة اليومية | Daily plan |
| `reco_col_date` | Date | التاريخ | Date |
| `reco_col_day` | Jour | اليوم | Day |
| `reco_col_expected` | Ventes prévues | المبيعات المتوقعة | Expected sales |
| `reco_col_stock` | À préparer | للتحضير | To prepare |
| `reco_col_peak` | Jour de pointe | يوم ذروة | Peak day |
| `reco_col_risk` | Risque | المخاطرة | Risk |
| `reco_risk_high` | Élevé | عالٍ | High |
| `reco_risk_med` | Moyen | متوسط | Medium |
| `reco_risk_low` | Faible | منخفض | Low |
| `reco_footer` | Moyenne journalière : {avg} · buffer : +15% · Élevé = +30% vs moyenne | المعدل اليومي : {avg} · هامش : +15% | Daily avg: {avg} · buffer: +15% · High = +30% vs avg |

### Products (catalogue)

| Clé | FR | AR | EN |
|---|---|---|---|
| `products_title` | Mes produits | منتجاتي | My products |
| `products_sub` | Activez/désactivez les produits dans les calculs. | فعّل/عطّل المنتجات في الحسابات. | Activate or deactivate products in calculations. |
| `products_kpi_total` | Total produits | إجمالي المنتجات | Total products |
| `products_kpi_active` | Produits actifs | المنتجات النشطة | Active products |
| `products_kpi_custom` | Ajoutés par vous | أضفتها أنت | Added by you |
| `products_list_title` | Liste des produits | قائمة المنتجات | Product list |
| `products_list_caption` | Cochez/décochez Actif pour inclure ou exclure un produit. Les produits de votre CSV ne peuvent pas être supprimés. | تأشير/إلغاء تأشير "نشط" لتضمين أو استبعاد منتج. | Check/uncheck Active to include or exclude a product. CSV products cannot be deleted. |
| `products_col_dish` | Produit | المنتج | Product |
| `products_col_source` | Source | المصدر | Source |
| `products_col_share` | Part des ventes | حصة المبيعات | Sales share |
| `products_col_recipe` | Recette | الوصفة | Recipe |
| `products_col_active` | Actif | نشط | Active |
| `products_source_csv` | Depuis vos données | من بياناتك | From your data |
| `products_source_custom` | Ajouté par vous | أضفته أنت | Added by you |
| `products_add_title` | Ajouter un produit | إضافة منتج | Add a product |
| `products_add_caption` | Ajoutez un produit absent de votre CSV. Définissez ses ingrédients dans Liste de courses. | أضف منتجاً غير موجود في CSV. | Add a product not in your CSV. Set up ingredients in Shopping List. |
| `products_add_placeholder` | ex. Frites, Nuggets… | مثال: بطاطا مقلية... | e.g. French Fries, Nuggets… |
| `products_add_btn` | Ajouter | إضافة | Add |
| `products_add_exists` | **{name}** est déjà dans votre catalogue. | **{name}** موجود بالفعل. | **{name}** is already in your catalogue. |
| `products_add_success` | **{name}** ajouté. | تمت إضافة **{name}**. | **{name}** added. |
| `products_del_title` | Supprimer un produit | حذف منتج | Remove a product |
| `products_del_caption` | Seuls les produits que vous avez ajoutés peuvent être supprimés. | يمكن حذف المنتجات المضافة يدوياً فقط. | Only manually added products can be removed. |
| `products_del_btn` | Supprimer | حذف | Delete |
| `products_info` | Les produits de votre CSV peuvent être désactivés mais pas supprimés — leur historique est nécessaire aux prévisions. | منتجات CSV يمكن تعطيلها فقط لا حذفها. | CSV products can be deactivated but not deleted — history needed for forecasts. |

### Shopping List

| Clé | FR | AR | EN |
|---|---|---|---|
| `shop_title` | Liste de courses | قائمة التسوق | Shopping list |
| `shop_sub` | Ingrédients pour les 7 prochains jours, calculés depuis vos recettes et prévisions. | المكونات لـ7 أيام القادمة بناءً على وصفاتك وتوقعاتك. | Ingredients for the next 7 days, from your recipes and forecasts. |
| `shop_ingredients_title` | Ingrédients par plat | المكونات لكل طبق | Ingredients per dish |
| `shop_ingredients_caption` | Quantités pour une portion. Modifiez librement. | الكميات لحصة واحدة. قابلة للتعديل. | Quantities for one serving. Freely editable. |
| `shop_stock_title` | Ce que vous avez déjà | ما لديك بالفعل | What you already have |
| `shop_stock_caption` | Entrez votre stock actuel. Seul ce qui manque apparaîtra dans la commande. | أدخل مخزونك الحالي. فقط الناقص سيظهر في الطلب. | Enter your current stock. Only shortfalls will appear in the order. |
| `shop_needs_title` | Besoins journaliers — 7 prochains jours | الاحتياجات اليومية — 7 أيام القادمة | Daily needs — next 7 days |
| `shop_order_title` | Quoi acheter | ماذا تشتري | What to buy |
| `shop_stockout_title` | Quand serez-vous en rupture ? | متى ستنفد المخزون؟ | When will you run out? |
| `shop_kpi_tobuy` | Ingrédients à acheter | مكونات للشراء | Ingredients to buy |
| `shop_kpi_ok` | Déjà en stock | في المخزون بالفعل | Already in stock |
| `shop_kpi_first` | Première rupture | أول نقص | First shortage |
| `shop_col_ing` | Ingrédient | المكون | Ingredient |
| `shop_col_unit` | Unité | الوحدة | Unit |
| `shop_col_need` | Besoin 7j (+15%) | احتياج 7أ (+15%) | Need 7d (+15%) |
| `shop_col_have` | En stock | في المخزون | You have |
| `shop_col_buy` | À acheter | للشراء | To buy |
| `shop_col_status` | Statut | الحالة | Status |
| `shop_status_ok` | OK | موافق | OK |
| `shop_status_order` | Commander | اطلب | Order |
| `shop_status_critical` | Critique | حرج | Critical |
| `shop_col_instock` | En stock | في المخزون | In stock |
| `shop_col_perday` | Consommé/jour | مستهلك/يوم | Used per day |
| `shop_col_runout` | Date de rupture | تاريخ النفاد | Runs out on |
| `shop_col_urgency` | Urgence | الأولوية | Urgency |
| `shop_urg_ok` | OK | موافق | OK |
| `shop_urg_soon` | Très bientôt | قريباً جداً | Very soon |
| `shop_urg_week` | Cette semaine | هذا الأسبوع | This week |
| `shop_urg_later` | Plus tard | لاحقاً | Later |
| `shop_caption` | Besoin = prévision × ingrédients × +15% buffer · À acheter = Besoin − Stock | الحاجة = توقع × مكونات × +15% هامش | Need = forecast × ingredients × +15% buffer · To buy = Need − Stock |
| `shop_no_ing` | Aucun ingrédient défini. Remplissez le tableau ci-dessus. | لا مكونات محددة. امل الجدول أعلاه. | No ingredients defined. Fill in the table above. |

### About (sidebar footer)

| Clé | FR | AR | EN |
|---|---|---|---|
| `about_title` | À propos | حول | About |
| `about_what_title` | Qu'est-ce que FoodCast ? | ما هو FoodCast؟ | What is FoodCast? |
| `about_what_body` | Dashboard de prévision des ventes pour petites enseignes alimentaires. Importez votre historique CSV, obtenez des prévisions Prophet, et décidez quoi préparer et acheter — sans compétences techniques. | لوحة تحكم لتوقعات المبيعات للمطاعم الصغيرة. | Sales forecasting dashboard for small food businesses. Upload your CSV history, get Prophet forecasts, decide what to prepare and buy — no technical skills needed. |
| `about_format_title` | Format de votre fichier CSV | تنسيق ملف CSV | Your CSV file format |
| `about_dl_sample` | Télécharger un CSV exemple | تحميل CSV تجريبي | Download sample CSV |
| `about_stack_title` | Construit avec | بُني بـ | Built with |
| `about_version` | v{v} &nbsp;·&nbsp; FoodCast © 2025 | v{v} &nbsp;·&nbsp; FoodCast © 2025 | v{v} &nbsp;·&nbsp; FoodCast © 2025 |

### Onboarding (état vide, aucun CSV chargé)

| Clé | FR | AR | EN |
|---|---|---|---|
| `onboard_title` | Bienvenue sur FoodCast | مرحباً بك في FoodCast | Welcome to FoodCast |
| `onboard_sub` | Importez vos ventes pour voir vos prévisions et conseils en temps réel. | استورد مبيعاتك لرؤية التوقعات والنصائح. | Import your sales to see forecasts and insights in real time. |
| `onboard_load_btn` | Charger un CSV | تحميل CSV | Load a CSV |
| `onboard_demo_btn` | Utiliser les données démo | استخدام البيانات التجريبية | Use demo data |
| `onboard_demo_hint` | Boulangerie Atlas · 27 mois · 12 produits | مخبزة أطلس · 27 شهراً · 12 منتجاً | Boulangerie Atlas · 27 months · 12 products |

### Dialog saisie rapide

| Clé | FR | AR | EN |
|---|---|---|---|
| `dialog_add_title` | Ajouter des ventes | إضافة مبيعات | Add sales |
| `dialog_add_sub` | Les saisies sont conservées pendant cette session. Exportez le CSV pour les garder. | تُحفظ الإدخالات خلال الجلسة. صدّر CSV للاحتفاظ بها. | Entries are kept during this session. Export CSV to keep them. |
| `dialog_btn_open` | + Ajouter des ventes | + إضافة مبيعات | + Add sales |

---

## 5. Règles d'extraction des modules

### `app/constants.py`
Exporte : `TRANS`, `RECIPES`, `MAROC_FIXED_HOLIDAYS`, `ISLAMIC_HOLIDAYS`,
`RAMADAN_DATES`, `REQUIRED_COLS`, `WEEKEND_DAYS`, `BUFFER_RATE`, `PEAK_THRESH`,
`UNIT_OPTIONS`, `EN_DAYS_IDX`

### `app/i18n.py`
Importe `TRANS`, `EN_DAYS_IDX` depuis `constants`.  
Exporte : `T(key)`, `Tlist(key)`, `Tdays()`, `day_label(en_name)`

### `app/ui.py`
Importe `T` depuis `i18n`.  
Exporte : `_kpi()`, `_sec()`, `_fin_kpi()`, `_progress_bar()`, `_mad()`, `load_css()`  
`load_css()` remplace le bloc `st.markdown("""<style>...""")` hardcodé.

### `app/logic/data_loading.py`
Importe `REQUIRED_COLS`, `DEFAULT_CSV` depuis `constants`.  
Décorateurs `@st.cache_data` conservés.

### `app/logic/forecasting.py`
Importe `MAROC_FIXED_HOLIDAYS`, `ISLAMIC_HOLIDAYS`, `RAMADAN_DATES`, `WEEKEND_DAYS`,
`PEAK_THRESH` depuis `constants`.  
Décorateurs `@st.cache_data` / `@st.cache_resource` conservés.  
**Attention** : `@st.cache_resource` n'est pas importable sans Streamlit — ce module
sera importé uniquement dans un contexte Streamlit, pas en test unitaire pur.

### `app/logic/finance.py`
Importe `build_fin_df()`, `_mad()` — logique pure Pandas, pas de Streamlit.

### `app/logic/insights.py`
Contient le scoring PDJ, la détection de promos, la détection dormants, `_risk_level()`.
Pas de Streamlit. Testable unitairement.

### `app/logic/inventory.py`
Contient `compute_stockout_date()`. Pas de Streamlit. Testable unitairement.

### `app/pages/today.py`
Fonction signature : `render(df, daily, data_hash, avg_open, last_date, fc_start,
catalog, active_prods, active_csv, price_map, cost_map, ramadan_mode, lite_mode)`

### `app/pages/forecast.py`
Même pattern de signature.

### `app/pages/finance.py`
Même pattern de signature.

### `app/pages/analytics.py`
Même pattern de signature.

### `app/main.py` (routeur léger, ~120 lignes cible)
```python
# sidebar setup (langue, upload, nav radio 4 onglets, toggles, footer about)
# data loading gate (onboarding si aucun CSV)
# shared state setup (catalog, prices, costs, target)
# dispatch :
if   page == "today"    : pages.today.render(...)
elif page == "forecast" : pages.forecast.render(...)
elif page == "finance"  : pages.finance.render(...)
elif page == "analytics": pages.analytics.render(...)
```

---

## 6. Risques identifiés

| # | Risque | Probabilité | Mitigation |
|---|--------|-------------|------------|
| R1 | `@st.cache_resource` keyed sur `data_hash` — après extraction en module, le cache est partagé entre les renders si la clé est identique. Pas de régression fonctionnelle attendue, mais à vérifier. | Faible | Tester avec re-upload CSV après extraction |
| R2 | `st.session_state` accédé via clés composées (`f"prices_{data_hash}"`) — si `data_hash` change de calcul entre main et un module, les clés divergent. | Moyen | `data_hash` calculé une seule fois dans `main.py` et passé en argument à tous les modules |
| R3 | `st.dialog` (saisie rapide) — nécessite Streamlit ≥ 1.31. Streamlit 1.56 est installé → OK. | Nul | Vérifié |
| R4 | Import circulaire si `ui.py` importe `i18n.py` qui importe `constants.py` qui importe... | Faible | Graphe d'import strict : `constants` ← `i18n` ← `ui` ← `pages` ← `main` |
| R5 | Les notebooks `01_EDA.ipynb` et `02_forecast.ipynb` importent depuis `src/` (pas `app/`) — non affectés par la refonte de `app/`. | Nul | |
| R6 | `ventes_maroc_2024.csv` orphelin — non touché par cette refonte. | Nul | |

---

## 7. Étapes de livraison

| Étape | Description | Commit message cible |
|---|---|---|
| **1** | Ce fichier `REFACTOR_PLAN.md` | `docs: refactor plan — navigation 4 onglets + extraction modules` |
| **2** | Extraction modules sans changer UX | `refactor: extract constants, i18n, ui, logic — no UX change` |
| **3** | Refonte navigation (4 onglets, sous-tabs, onboarding, dialog saisie) | `feat: navigation 4-tab + st.dialog saisie + onboarding state` |
| **4** | i18n complète des 5 pages anglaises | `feat: i18n — overview, reco, products, shopping list, about translated` |
| **5** | Polish — code mort, états vides, espacements, RTL cohérent | `fix: polish — dead code, empty states, RTL consistency` |

---

## 8. Ce qui ne change PAS

- Format CSV d'entrée (`date`, `produit`, `quantite_vendue` + optionnels `heure`, `prix_unitaire`)
- Toute la logique Prophet (paramètres, jours fériés, MAE holdout)
- Tous les calculs financiers (marges, CA, objectif)
- Toute la logique de scoring insights
- Le CSS injecté existant (ajusté si besoin, jamais remplacé par un framework)
- Les datasets `data/` (inchangés)

---

*Validez ce plan avant toute modification de code Python.*
