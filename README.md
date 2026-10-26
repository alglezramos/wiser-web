# Wiser · Slotting de almacén (Streamlit en el navegador con stlite)

Página estática: no necesita servidor. Todo el cálculo corre en el navegador del usuario,
así que los datos del almacén no salen de su computador.

## Archivos
- `index.html`    : monta la app con stlite 1.12.0 (carga stlite y Pyodide desde jsDelivr).
- `wiser_app.py`  : la app completa (núcleo de optimización + interfaz Streamlit).
- `skus.csv`, `slots.csv` : archivos de ejemplo para la opción "Subir CSV".
- `.nojekyll`     : evita que GitHub Pages procese la carpeta con Jekyll.

## Publicar en GitHub Pages
1. Crea un repositorio y sube estos archivos a la raíz de la rama `main`.
2. Settings → Pages → Source: "Deploy from a branch" → Branch: `main` / carpeta `/ (root)`.
3. Espera 1–2 minutos. La app queda en `https://<usuario>.github.io/<repo>/`.

## Probar en local antes de subir
    python -m http.server 8000
y abre http://localhost:8000 (no abras index.html con doble clic: el navegador bloquea
la descarga de wiser_app.py desde file://).

## Notas
- La primera carga descarga Python (Pyodide), numpy, pandas, scipy y pyarrow; es más lenta.
  Las siguientes usan la caché del navegador.
- Requiere un navegador moderno (Chrome, Edge, Safari reciente, Firefox 114+).
- Si cambias `wiser_app.py`, el navegador puede mostrar la versión en caché: recarga forzada (Ctrl+Shift+R).
- Para fijar versiones y no depender de que jsDelivr cambie nada, el HTML ya apunta a `@stlite/browser@1.12.0`.
