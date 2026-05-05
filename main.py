import flet as ft

def main(page: ft.Page):
    page.title = "Salut la Guinée"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    
    texte_resultat = ft.Text("", size=20, weight=ft.FontWeight.BOLD)
    
    def bouton_clique(e):
        texte_resultat.value = "224 FOREVER MA GUELLE 🔥🇬🇳"
        page.update()

    page.add(
        ft.Image(
            src="https://flagcdn.com/w320/gn.png", 
            width=200,
            border_radius=10
        ),
        ft.Text("SALUT LA GUINÉE 🇬🇳", size=30, weight=ft.FontWeight.BOLD),
        ft.Text("Par la Coopérative de Conakry", size=16),
        ft.ElevatedButton(
            "Je suis fier d'être Guinéen", 
            width=300,
            on_click=bouton_clique
        ),
        texte_resultat
    )

ft.app(target=main, view=ft.AppView.FLET_APP)
