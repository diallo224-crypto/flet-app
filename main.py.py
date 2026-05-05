#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🇬🇳 GUINÉE AGRIMETRICS
Application mobile de suivi et prédiction des prix agricoles
Pour le peuple de Guinée - Toutes les 8 régions
Lancement: python main.py
Compilation APK: flet build apk
"""

import flet as ft
import asyncio
import io
import base64
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================
# DONNÉES DES 8 RÉGIONS DE GUINÉE
# ============================================

REGIONS = {
    "Conakry": {
        "base_prix": {"riz local": 12000, "riz importé": 15000, "huile de palme": 20000, "oignon": 8000, "tomate": 6000, "poisson frais": 25000},
        "climat": "tropical humide", "marches": ["Madina", "Matam", "Matoto", "Kaloum"]
    },
    "Kindia": {
        "base_prix": {"riz local": 10000, "mangue": 5000, "ananas": 3000, "orange": 2000, "fonio": 11000, "huile de palme": 18000},
        "climat": "tropical humide", "marches": ["Kindia centre", "Coyah", "Dubréka"]
    },
    "Boké": {
        "base_prix": {"riz local": 10000, "arachide": 8000, "mil": 8500, "niébé": 7000, "huile de palme": 16000},
        "climat": "tropical sec", "marches": ["Boké centre", "Kamsar", "Sangarédi"]
    },
    "Mamou": {
        "base_prix": {"pomme de terre": 8000, "oignon": 6000, "ail": 15000, "tomate": 4000, "chou": 5000},
        "climat": "tropical d'altitude", "marches": ["Mamou centre", "Dalaba"]
    },
    "Faranah": {
        "base_prix": {"riz local": 9000, "arachide": 7000, "mil": 8000, "sésame": 12000, "maïs": 6000},
        "climat": "savane", "marches": ["Faranah centre", "Kissidougou"]
    },
    "Kankan": {
        "base_prix": {"riz local": 9500, "mil": 7000, "sorgho": 6000, "arachide": 8000, "coton": 15000, "maïs": 6500},
        "climat": "soudanien", "marches": ["Kankan centre", "Siguiri or"]
    },
    "Labé": {
        "base_prix": {"pomme de terre": 7000, "oignon": 5000, "orange": 3000, "fonio": 12000, "tomate": 4500, "ail": 14000},
        "climat": "tropical d'altitude", "marches": ["Labé centre", "Tougué"]
    },
    "N'Zérékoré": {
        "base_prix": {"riz local": 8000, "café": 30000, "cacao": 25000, "huile de palme": 15000, "banane": 4000, "mangue": 6000},
        "climat": "équatorial", "marches": ["N'Zérékoré centre", "Guéckédou", "Macenta"]
    }
}

PRODUITS = [
    "riz local", "riz importé", "fonio", "mil", "sorgho", "maïs",
    "manioc frais", "patate douce", "igname",
    "arachide", "niébé", "soja", "sésame",
    "oignon", "ail", "tomate", "piment", "aubergine", "gombo",
    "mangue", "banane", "ananas", "orange", "citron", "papaye",
    "huile de palme", "huile d'arachide", "beurre de karité",
    "viande de bœuf", "poulet local", "poisson frais",
    "lait", "œuf", "café", "cacao"
]

# ============================================
# GÉNÉRATEUR DE DONNÉES RÉALISTES
# ============================================

class DataEngine:
    """Moteur de données complet pour la Guinée"""
    
    def __init__(self):
        np.random.seed(42)
        self.prix_df = None
        self.ml_ready = False
        self._generate_all_data()
        self._train_models()
    
    def _get_saison(self, mois):
        if mois in [6,7,8,9,10]: return "pluies"
        elif mois in [11,12,1,2]: return "sèche froide"
        else: return "sèche chaude"
    
    def _generate_all_data(self):
        """Génère 5 ans de données de prix"""
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='W')
        data = []
        
        for region, info in REGIONS.items():
            for produit in PRODUITS:
                base = info['base_prix'].get(produit, np.random.uniform(5000, 30000))
                
                t = np.linspace(0, 4*np.pi, len(dates))
                if info['climat'] in ['tropical humide', 'équatorial']:
                    saison = 0.12 * np.sin(t) + 0.08 * np.sin(2*t)
                elif info['climat'] in ['tropical sec', 'soudanien', 'savane']:
                    saison = 0.22 * np.sin(t) + 0.12 * np.sin(2*t)
                else:
                    saison = 0.10 * np.sin(t) + 0.05 * np.sin(2*t)
                
                tendance = np.linspace(0, np.random.uniform(-0.2, 0.25), len(dates))
                bruit = np.random.normal(0, 0.04, len(dates))
                prix = base * (1 + saison + tendance + bruit)
                prix = np.maximum(prix, 500).astype(int)
                
                for i, date in enumerate(dates):
                    data.append({
                        'date': date, 'region': region, 'produit': produit,
                        'prix': prix[i], 'saison': self._get_saison(date.month),
                        'mois': date.month, 'annee': date.year
                    })
        
        self.prix_df = pd.DataFrame(data)
        print(f"✅ {len(self.prix_df):,} données générées")
    
    def _train_models(self):
        """Entraîne l'IA"""
        df = self.prix_df.copy()
        df['mois_sin'] = np.sin(2*np.pi*df['mois']/12)
        df['mois_cos'] = np.cos(2*np.pi*df['mois']/12)
        
        for lag in [1,4,8,12]:
            df[f'prix_lag{lag}'] = df.groupby(['region','produit'])['prix'].shift(lag)
        
        for w in [4,8]:
            df[f'prix_roll{w}'] = df.groupby(['region','produit'])['prix'].transform(lambda x: x.rolling(w,1).mean())
        
        saison_d = pd.get_dummies(df['saison'], prefix='s')
        region_d = pd.get_dummies(df['region'], prefix='r')
        df = pd.concat([df, saison_d, region_d], axis=1)
        df = df.dropna()
        
        self.feature_cols = [c for c in df.columns if c not in 
            ['date','region','produit','saison','prix','mois','annee'] and df[c].dtype in ['float64','int64']]
        
        X = df[self.feature_cols].values
        y = df['prix'].values
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_s = self.scaler_X.fit_transform(X)
        y_s = self.scaler_y.fit_transform(y.reshape(-1,1))
        
        X_tr, X_te, y_tr, y_te = train_test_split(X_s, y_s, test_size=0.2, shuffle=False)
        
        models = {
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42),
            'GradientBoost': GradientBoostingRegressor(n_estimators=80, max_depth=5, random_state=42)
        }
        
        best_r2 = -999
        self.best_model = None
        
        for name, model in models.items():
            model.fit(X_tr, y_tr.ravel())
            r2 = r2_score(y_te, model.predict(X_te))
            if r2 > best_r2:
                best_r2 = r2
                self.best_model = model
                self.model_name = name
        
        self.ml_ready = True
        print(f"🤖 IA prête: {self.model_name} (R²={best_r2:.3f})")
    
    def get_stats(self):
        """Statistiques globales"""
        df = self.prix_df
        return {
            'prix_moyen': float(df['prix'].mean()),
            'prix_min': float(df['prix'].min()),
            'prix_max': float(df['prix'].max()),
            'nb_produits': df['produit'].nunique(),
            'nb_regions': df['region'].nunique(),
            'nb_donnees': len(df),
            'volatilite': float(df['prix'].std() / df['prix'].mean())
        }
    
    def get_price_evolution(self, region=None, produit=None):
        """Données pour graphique d'évolution"""
        df = self.prix_df.copy()
        if region and region != "Toutes":
            df = df[df['region'] == region]
        if produit:
            df = df[df['produit'] == produit]
        return df
    
    def predict(self, region, produit, periods=12):
        """Prédit les prix futurs"""
        df = self.prix_df[(self.prix_df['region']==region) & (self.prix_df['produit']==produit)].tail(24).copy()
        
        if len(df) < 8:
            return None
        
        df['mois_sin'] = np.sin(2*np.pi*df['mois']/12)
        df['mois_cos'] = np.cos(2*np.pi*df['mois']/12)
        for lag in [1,4,8,12]:
            df[f'prix_lag{lag}'] = df['prix'].shift(lag)
        for w in [4,8]:
            df[f'prix_roll{w}'] = df['prix'].rolling(w,1).mean()
        
        last = df.iloc[-1:].copy()
        future_dates = pd.date_range(df['date'].max()+timedelta(weeks=1), periods=periods, freq='W')
        predictions = []
        
        for d in future_dates:
            row = last.copy()
            row['date'] = d
            row['mois'] = d.month
            row['mois_sin'] = np.sin(2*np.pi*d.month/12)
            row['mois_cos'] = np.cos(2*np.pi*d.month/12)
            
            X = row[self.feature_cols].fillna(0).values
            prix_pred = self.scaler_y.inverse_transform(
                self.best_model.predict(self.scaler_X.transform(X))[0].reshape(-1,1)
            )[0][0]
            row['prix'] = max(500, int(prix_pred))
            predictions.append(row)
            last = row
        
        return pd.concat(predictions)
    
    def recommend(self, region, saison, top=5):
        """Recommandations agricoles"""
        df = self.prix_df[(self.prix_df['region']==region) & (self.prix_df['saison']==saison)]
        if len(df) == 0:
            return []
        
        stats = df.groupby('produit').agg(
            prix_moyen=('prix','mean'),
            stabilite=('prix','std')
        ).reset_index()
        stats['stabilite'] = stats['stabilite'].fillna(0)
        stats['score'] = stats['prix_moyen'] * 0.7 - stats['stabilite'] * 0.3
        return stats.nlargest(top, 'score').to_dict('records')

# ============================================
# APPLICATION FLET
# ============================================

class App:
    """Application mobile Guinée Agrimetrics"""
    
    def __init__(self):
        print("🔄 Démarrage de Guinée Agrimetrics...")
        self.engine = DataEngine()
        self.stats = self.engine.get_stats()
        print("✅ Application prête!")
    
    def main(self, page: ft.Page):
        # Configuration
        page.title = "Guinée Agrimetrics"
        page.theme = ft.Theme(color_scheme_seed=ft.colors.GREEN, use_material3=True)
        page.theme_mode = ft.ThemeMode.LIGHT
        page.padding = 0
        page.bgcolor = ft.colors.SURFACE_VARIANT
        
        # Barre du haut
        page.appbar = ft.AppBar(
            leading=ft.Icon(ft.icons.ECO, color=ft.colors.WHITE),
            title=ft.Text("Guinée Agrimetrics", color=ft.colors.WHITE, weight=ft.FontWeight.BOLD),
            center_title=True,
            bgcolor=ft.colors.GREEN_800,
        )
        
        # Conteneur principal
        self.body = ft.Container(expand=True, padding=15)
        
        # Barre de navigation
        page.navigation_bar = ft.NavigationBar(
            selected_index=0,
            on_change=lambda e: self._navigate(e, page),
            destinations=[
                ft.NavigationDestination(icon=ft.icons.HOME_OUTLINED, selected_icon=ft.icons.HOME, label="Accueil"),
                ft.NavigationDestination(icon=ft.icons.SHOW_CHART_OUTLINED, selected_icon=ft.icons.SHOW_CHART, label="Prix"),
                ft.NavigationDestination(icon=ft.icons.PSYCHOLOGY_OUTLINED, selected_icon=ft.icons.PSYCHOLOGY, label="IA"),
                ft.NavigationDestination(icon=ft.icons.LIGHTBULB_OUTLINED, selected_icon=ft.icons.LIGHTBULB, label="Conseils"),
                ft.NavigationDestination(icon=ft.icons.INFO_OUTLINED, selected_icon=ft.icons.INFO, label="Infos"),
            ]
        )
        
        page.add(self.body)
        self._show_home(page)
    
    def _navigate(self, e, page):
        pages = [self._show_home, self._show_prices, self._show_predict, 
                self._show_recommend, self._show_info]
        pages[e.control.selected_index](page)
    
    # ==================== PAGE ACCUEIL ====================
    
    def _show_home(self, page):
        s = self.stats
        
        cards = ft.GridView(
            expand=True, runs_count=2, max_extent=200,
            child_aspect_ratio=0.9, spacing=10, run_spacing=10,
            controls=[
                self._card("💰 Prix Moyen", f"{s['prix_moyen']:,.0f} GNF", ft.icons.ATTACH_MONEY, ft.colors.GREEN),
                self._card("📦 Produits", str(s['nb_produits']), ft.icons.INVENTORY_2, ft.colors.BLUE),
                self._card("📍 Régions", "8 régions", ft.icons.MAP, ft.colors.ORANGE),
                self._card("📊 Volatilité", f"{s['volatilite']:.1%}", ft.icons.TRENDING_UP, ft.colors.PURPLE),
            ]
        )
        
        self.body.content = ft.Column([
            ft.Text("Tableau de Bord", size=24, weight=ft.FontWeight.BOLD),
            ft.Text(f"{s['nb_donnees']:,} données • 2020-2024 • IA active", size=13, color=ft.colors.GREY_600),
            ft.Divider(height=20),
            cards,
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Row([ft.Icon(ft.icons.PSYCHOLOGY, color=ft.colors.GREEN), 
                               ft.Text("Intelligence Artificielle", weight=ft.FontWeight.BOLD)]),
                        ft.Text(f"Modèle: {self.engine.model_name}", size=13),
                        ft.Text("Prédit les prix jusqu'à 12 semaines", size=12, color=ft.colors.GREY_600),
                    ]),
                    padding=15
                )
            ),
        ], scroll=ft.ScrollMode.AUTO, spacing=10)
        page.update()
    
    def _card(self, title, value, icon, color):
        return ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Icon(icon, size=40, color=color),
                    ft.Text(title, size=13, color=ft.colors.GREY_700),
                    ft.Text(value, size=22, weight=ft.FontWeight.BOLD, color=color),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
                padding=15, alignment=ft.alignment.center
            ), elevation=4
        )
    
    # ==================== PAGE PRIX ====================
    
    def _show_prices(self, page):
        self.dd_region = ft.Dropdown(
            label="Région", width=170,
            options=[ft.dropdown.Option("Toutes")] + [ft.dropdown.Option(r) for r in REGIONS.keys()],
            value="Toutes"
        )
        self.dd_produit = ft.Dropdown(
            label="Produit", width=170,
            options=[ft.dropdown.Option(p) for p in PRODUITS[:20]],
            value="riz local"
        )
        self.img_price = ft.Image(visible=False, border_radius=10)
        
        self.body.content = ft.Column([
            ft.Text("📈 Évolution des Prix", size=24, weight=ft.FontWeight.BOLD),
            ft.Row([self.dd_region, self.dd_produit], spacing=10, wrap=True),
            ft.ElevatedButton("🔍 Afficher", on_click=lambda _: self._update_price(page)),
            ft.Divider(),
            self.img_price
        ], scroll=ft.ScrollMode.AUTO, spacing=10)
        page.update()
    
    def _update_price(self, page):
        df = self.engine.get_price_evolution(self.dd_region.value, self.dd_produit.value)
        
        fig, ax = plt.subplots(figsize=(8,4.5))
        for name, grp in df.groupby(['region','produit']):
            grp = grp.sort_values('date')
            ax.plot(grp['date'], grp['prix'], label=f"{name[0]}-{name[1]}", alpha=0.7, linewidth=1.2)
        
        ax.set_title(f"Prix - {self.dd_produit.value}", fontweight='bold')
        ax.set_ylabel("GNF"); ax.legend(fontsize=6, loc='upper left')
        ax.grid(alpha=0.3); plt.xticks(rotation=45); plt.tight_layout()
        
        buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=100); buf.seek(0)
        self.img_price.src_base64 = base64.b64encode(buf.read()).decode()
        self.img_price.visible = True; plt.close()
        page.update()
    
    # ==================== PAGE PRÉDICTIONS ====================
    
    def _show_predict(self, page):
        self.pd_region = ft.Dropdown(label="Région", width=170,
            options=[ft.dropdown.Option(r) for r in REGIONS.keys()], value="Conakry")
        self.pd_produit = ft.Dropdown(label="Produit", width=170,
            options=[ft.dropdown.Option(p) for p in PRODUITS[:20]], value="riz local")
        self.img_pred = ft.Image(visible=False, border_radius=10)
        self.txt_pred = ft.Text("", size=14, weight=ft.FontWeight.BOLD)
        
        self.body.content = ft.Column([
            ft.Text("🔮 Prédictions IA", size=24, weight=ft.FontWeight.BOLD),
            ft.Text("L'intelligence artificielle prédit les prix à venir", size=13, color=ft.colors.GREY_600),
            ft.Row([self.pd_region, self.pd_produit], spacing=10, wrap=True),
            ft.ElevatedButton("🤖 Lancer", on_click=lambda _: self._update_pred(page)),
            ft.Divider(),
            self.txt_pred,
            self.img_pred
        ], scroll=ft.ScrollMode.AUTO, spacing=10)
        page.update()
    
    def _update_pred(self, page):
        preds = self.engine.predict(self.pd_region.value, self.pd_produit.value)
        
        if preds is None:
            self.txt_pred.value = "❌ Données insuffisantes pour cette prédiction"
            self.img_pred.visible = False
            page.update()
            return
        
        fig, ax = plt.subplots(figsize=(8,4.5))
        hist = self.engine.prix_df[
            (self.engine.prix_df['region']==self.pd_region.value) & 
            (self.engine.prix_df['produit']==self.pd_produit.value)
        ].tail(30)
        
        ax.plot(hist['date'], hist['prix'], 'g-', label='Historique', linewidth=2)
        ax.plot(preds['date'], preds['prix'], 'r--o', label='Prédiction IA', linewidth=2)
        ax.set_title(f"Prédiction - {self.pd_produit.value} ({self.pd_region.value})", fontweight='bold')
        ax.set_ylabel("GNF"); ax.legend(); ax.grid(alpha=0.3)
        plt.xticks(rotation=45); plt.tight_layout()
        
        buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=100); buf.seek(0)
        self.img_pred.src_base64 = base64.b64encode(buf.read()).decode()
        self.img_pred.visible = True; plt.close()
        
        dernier = preds.iloc[-1]
        self.txt_pred.value = f"📊 Prix prédit au {dernier['date'].strftime('%d/%m/%Y')}: {dernier['prix']:,.0f} GNF"
        page.update()
    
    # ==================== PAGE CONSEILS ====================
    
    def _show_recommend(self, page):
        self.rc_region = ft.Dropdown(label="Région", width=170,
            options=[ft.dropdown.Option(r) for r in REGIONS.keys()], value="Kankan")
        self.rc_saison = ft.Dropdown(label="Saison", width=170,
            options=[
                ft.dropdown.Option("pluies", "🌧️ Pluies"),
                ft.dropdown.Option("sèche froide", "❄️ Sèche froide"),
                ft.dropdown.Option("sèche chaude", "☀️ Sèche chaude")
            ], value="pluies")
        self.rc_list = ft.Column(spacing=8)
        
        self.body.content = ft.Column([
            ft.Text("💡 Conseils Agricoles", size=24, weight=ft.FontWeight.BOLD),
            ft.Text("Cultures recommandées selon région et saison", size=13, color=ft.colors.GREY_600),
            ft.Row([self.rc_region, self.rc_saison], spacing=10, wrap=True),
            ft.ElevatedButton("🌱 Recommander", on_click=lambda _: self._update_reco(page)),
            ft.Divider(),
            self.rc_list
        ], scroll=ft.ScrollMode.AUTO, spacing=10)
        page.update()
    
    def _update_reco(self, page):
        recs = self.engine.recommend(self.rc_region.value, self.rc_saison.value)
        self.rc_list.controls.clear()
        
        if not recs:
            self.rc_list.controls.append(ft.Text("Aucune recommandation disponible"))
        else:
            emojis = ["🥇","🥈","🥉","4️⃣","5️⃣"]
            for i, r in enumerate(recs[:5]):
                self.rc_list.controls.append(
                    ft.Card(
                        content=ft.Container(
                            content=ft.Row([
                                ft.Text(emojis[i], size=24),
                                ft.Column([
                                    ft.Text(r['produit'].capitalize(), weight=ft.FontWeight.BOLD, size=15),
                                    ft.Text(f"{r['prix_moyen']:,.0f} GNF • Score: {r['score']:.0f}", size=12)
                                ], expand=True),
                                ft.Icon(ft.icons.ECO, color=ft.colors.GREEN, size=30)
                            ]), padding=12
                        ), elevation=2
                    )
                )
        page.update()
    
    # ==================== PAGE INFOS ====================
    
    def _show_info(self, page):
        self.body.content = ft.Column([
            ft.Text("ℹ️ Informations", size=24, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            ft.Card(content=ft.Container(content=ft.Column([
                ft.Text("🇬🇳 Guinée Agrimetrics v1.0", weight=ft.FontWeight.BOLD, size=16),
                ft.Text("Système de suivi et prédiction des prix agricoles", size=13),
                ft.Text(""), ft.Text("Régions couvertes:", weight=ft.FontWeight.BOLD),
                *[ft.Text(f"  • {r}") for r in REGIONS.keys()],
                ft.Text(""), ft.Text(f"Produits suivis: {len(PRODUITS)}", size=13),
                ft.Text("Données: 2020-2024 (5 ans)", size=13),
                ft.Text("IA: 3 modèles comparés", size=13),
                ft.Text(""), ft.Text("Développé pour les agriculteurs guinéens ❤️", size=13, color=ft.colors.GREEN_700),
            ]), padding=15)),
            ft.Card(content=ft.Container(content=ft.Column([
                ft.Text("🔧 Mode sombre", weight=ft.FontWeight.BOLD),
                ft.Switch(label="Activer", value=False, on_change=lambda e: self._toggle_dark(e, page)),
            ]), padding=15)),
        ], scroll=ft.ScrollMode.AUTO, spacing=10)
        page.update()
    
    def _toggle_dark(self, e, page):
        page.theme_mode = ft.ThemeMode.DARK if e.control.value else ft.ThemeMode.LIGHT
        page.update()

# ============================================
# LANCEMENT
# ============================================

async def main():
    app = App()
    await ft.app_async(target=app.main, name="Guinée Agrimetrics")

if __name__ == "__main__":
    asyncio.run(main())