# Proyecto ML Integral — Platzi

## Objetivo
Construir una plataforma de Machine Learning aplicada a una empresa edtech (Platzi).
El proyecto sirve como aprendizaje práctico de ML de principio a fin.

## Meta final
Una plataforma visual hecha con **Streamlit** que permita usar los modelos de ML de forma interactiva — pensada para ser mostrada como portfolio.

---

## Módulos del proyecto

| # | Módulo | Objetivo | Estado |
|---|--------|----------|--------|
| 1 | Churn Prediction | Detectar estudiantes que van a cancelar | ✅ Completo |
| 2 | Course Recommender | Sugerir cursos personalizados | ✅ Completo |
| 3 | User Segmentation | Clustering de perfiles de estudiantes | 🔲 Pendiente |
| 4 | Conversion Prediction | Leads con alta probabilidad de pagar | 🔲 Pendiente |
| 5 | Engagement Analysis | Patrones de actividad y riesgo de abandono | 🔲 Pendiente |

---

## Stack tecnológico

- **Lenguaje:** Python
- **Datos:** pandas, numpy, faker
- **ML:** scikit-learn, imbalanced-learn
- **Visualización:** matplotlib, seaborn, plotly
- **Plataforma/UI:** Streamlit
- **Entorno:** VS Code + Git Bash + venv

---

## Estructura del proyecto

```
ProyectoMLClientes/
├── CONTEXTO.md
├── .gitignore
├── venv/                        ← no se sube a git
├── data/
│   ├── raw/                     ← se sube a git (necesario para probar el modelo)
│   │   └── platzi_churn.csv     ← 5,000 estudiantes × 58 columnas
│   └── processed/
├── notebooks/
│   ├── 01_churn_eda.ipynb       ← análisis exploratorio completo
│   └── 02_churn_modelo.ipynb    ← entrenamiento del Random Forest
├── src/
│   ├── churn/
│   │   ├── __init__.py
│   │   └── generate_data.py     ← NO se sube a git (.gitignore)
│   ├── recommender/
│   ├── segmentation/
│   ├── conversion/
│   └── engagement/
├── models/
│   └── churn_model.pkl          ← no se sube a git
├── app/
│   ├── main.py                  ← pantalla de inicio de la plataforma
│   └── pages/
│       └── 01_Churn.py          ← dashboard de churn prediction
└── reports/                     ← gráficos exportados por los notebooks
```

---

## Pipeline general de ML (se repite en cada módulo)

```
1. Generar datos        → src/<modulo>/generate_data.py
2. Explorar datos       → notebooks/0X_<modulo>_eda.ipynb
3. Entrenar modelo      → notebooks/0X_<modulo>_modelo.ipynb
4. Visualizar           → app/pages/0X_<Modulo>.py
```

---

## Decisiones tomadas

| Decisión | Por qué |
|----------|---------|
| Usar Streamlit para la UI | Solo Python, rápido de construir, visual profesional |
| Empezar por Churn | Es el más impactante y establece el pipeline base |
| No subir generate_data.py | El proyecto se presenta con datos reales |
| Subir data/raw/ | El CSV se sube para que el modelo se pueda probar |
| Dashboard batch (no formulario manual) | Más realista: predice sobre toda la base, no de a uno |
| Random Forest con class_weight='balanced' | Compensa el desbalance 3:1 sin modificar los datos |

---

## Dataset — platzi_churn.csv

- **5,000 estudiantes × 58 columnas**
- Planes reales de Platzi: Gratuito, Basic, Expert, Expert Duo, Expert Family
- 17 Escuelas reales de Platzi
- Platzi Rank calculado con fórmula oficial (ver clase=1PR, comentar=2PR, examen=200PR, ruta=500PR)
- Niveles Rewards: Estándar / Pro / Legend
- Churn rate: ~38% (emergente por plan, no forzado)
- Churn por plan: Basic 59% | Gratuito 40% | Expert 34% | Expert Duo 23% | Expert Family 19%

---

## Correr la plataforma

```bash
cd ~/OneDrive/Desktop/01PROYECTOS/ProyectoMLClientes
source venv/Scripts/activate
streamlit run app/main.py
```

**IMPORTANTE:** Para que la plataforma funcione, primero hay que correr los notebooks:
1. `01_churn_eda.ipynb` — genera los gráficos en reports/
2. `02_churn_modelo.ipynb` — genera `models/churn_model.pkl`

---

## Progreso por sesión

### Sesión 1 — 2026-05-07
- [x] Definición del proyecto: 5 módulos ML para Platzi
- [x] Decisión de usar Streamlit como plataforma visual
- [x] Definición del stack tecnológico
- [x] Creación de este archivo de contexto

### Sesión 2 — 2026-05-09
- [x] Configuración del entorno virtual (venv + librerías instaladas)
- [x] Creación de estructura de carpetas completa
- [x] Creación de .gitignore
- [x] Investigación real de Platzi (planes, escuelas, Platzi Rank, Rewards)
- [x] Script de generación de datos con realismo extremo (58 features)
- [x] Análisis profundo del dataset (bugs corregidos, features validadas)
- [x] Notebook 01 — EDA completo (12 secciones, 9 gráficos)
- [x] Notebook 02 — Modelo completo (Random Forest, métricas, feature importance)
- [x] Plataforma Streamlit — main.py + dashboard de Churn funcional
- [x] Git push al repositorio
- [x] data/raw/ habilitado para subir a git (necesario para probar el modelo)
- [x] Dashboard Streamlit rediseñado: predicción batch sobre toda la base, no formulario manual

---

### Sesión 3 — 2026-05-19

- [x] Módulo 2: Course Recommender completo
  - [x] `src/recommender/generate_data.py` — 80 cursos × 45,593 interacciones
  - [x] `data/raw/platzi_courses.csv` — catálogo con school, difficulty, rating, etc.
  - [x] `data/raw/platzi_interactions.csv` — 5,000 estudiantes × 9 cursos promedio
  - [x] `notebooks/03_recommender_eda.ipynb` — EDA completo (7 gráficos)
  - [x] `notebooks/04_recommender_modelo.ipynb` — SVD (k=20 factores latentes)
  - [x] `models/recommender_top10.csv` — recomendaciones precomputadas
  - [x] `models/recommender_model.pkl` — artefactos del modelo
  - [x] `app/pages/02_Recommender.py` — dashboard con filtros y gráficos

---

## Próximos pasos — Sesión 4

1. Correr `notebooks/03_recommender_eda.ipynb` → genera gráficos en reports/
2. Correr `notebooks/04_recommender_modelo.ipynb` → genera modelos en models/
3. Arrancar **Módulo 3: User Segmentation** (clustering de perfiles)
   - Algoritmo: K-Means + PCA para visualización
   - Dataset: usar platzi_churn.csv (ya existente)
   - Dashboard: mapa de segmentos con características de cada cluster
