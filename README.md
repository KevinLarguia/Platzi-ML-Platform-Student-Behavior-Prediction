# Platzi ML Platform

Plataforma de Machine Learning aplicada a una empresa edtech real.  
5 módulos que resuelven problemas de negocio concretos sobre el comportamiento de estudiantes.

---

## Módulos

| # | Módulo | Descripción | Estado |
|---|--------|-------------|--------|
| 1 | **Churn Prediction** | Detecta estudiantes en riesgo de cancelar antes de que ocurra | ✅ Completo |
| 2 | **Course Recommender** | Sugiere el próximo curso según historial y perfil del estudiante | ✅ Completo |
| 3 | User Segmentation | Agrupa estudiantes por perfil de comportamiento | 🔲 Pendiente |
| 4 | Conversion Prediction | Identifica usuarios gratuitos con alta probabilidad de pagar | 🔲 Pendiente |
| 5 | Engagement Analysis | Detecta caídas de actividad antes de que se conviertan en churn | 🔲 Pendiente |

---

## Módulo 1 — Churn Prediction

### El problema

Cada vez que un estudiante cancela su suscripción es ingreso perdido. El costo de adquirir un usuario nuevo es 5–7 veces mayor que retener uno existente. El módulo 1 predice qué estudiantes van a abandonar la plataforma antes de que lo hagan, para que el equipo de retención pueda actuar a tiempo.

### Cómo funciona

El modelo analiza 39 señales de comportamiento por estudiante:

- **Actividad:** días desde el último login, sesiones del último mes, horas vistas, tendencia semanal
- **Aprendizaje:** tasa de completado de cursos, certificados obtenidos, rutas completadas
- **Gamificación:** Platzi Rank, nivel Rewards, rachas activas
- **Suscripción:** plan, tipo de pago, intentos de pago fallidos, días para renovación
- **Comunidad:** participación en foros, LiveClasses asistidas, tutoriales publicados
- **Soporte:** tickets abiertos, NPS Score

El resultado es una probabilidad de churn de 0% a 100% por cada estudiante, clasificada en tres niveles de riesgo.

### Dashboard

El dashboard corre sobre toda la base de estudiantes en tiempo real:

- Métricas globales de riesgo
- Distribución de probabilidad de abandono
- Riesgo segmentado por Plan, País y Escuela
- Tabla interactiva con filtros por nivel de riesgo
- Exportación de listas para campañas de retención

### Stack

```
Python · pandas · scikit-learn · Streamlit · Plotly
Algoritmo: Random Forest (200 árboles, class_weight='balanced')
```

### Estructura

```
├── data/raw/platzi_churn.csv       # Dataset de estudiantes
├── notebooks/
│   ├── 01_churn_eda.ipynb          # Análisis exploratorio
│   └── 02_churn_modelo.ipynb       # Entrenamiento y evaluación
├── app/
│   ├── main.py                     # Plataforma principal
│   └── pages/01_Churn.py           # Dashboard de Churn
└── models/                         # Modelo entrenado (.pkl)
```

---

## Módulo 2 — Course Recommender

### El problema

Platzi tiene más de 1,000 cursos activos. Un estudiante que no encuentra rápido el siguiente paso relevante para su carrera se desactiva — y un estudiante desactivado es churn en potencia. El módulo 2 resuelve el problema de descubrimiento de contenido: dado el historial y el perfil de un estudiante, ¿qué curso debería tomar a continuación?

### Cómo funciona

El sistema usa **filtrado colaborativo con SVD** (Singular Value Decomposition). La lógica central es: estudiantes con patrones de consumo similares tienen preferencias similares. En vez de analizar el contenido de los cursos, el modelo aprende 20 factores latentes que capturan el "tipo de aprendiz" de cada usuario.

**Pipeline:**

1. Se construye una matriz usuario-curso con dos señales:
   - Rating explícito (1–5 estrellas) cuando el estudiante calificó el curso
   - Progreso normalizado como feedback implícito cuando no hay rating
2. La matriz se descompone con TruncatedSVD: `R ≈ U × Σ × Vᵀ`
3. Los factores latentes de usuario (`U`) y curso (`Vᵀ`) permiten predecir el rating de cualquier par estudiante-curso no visto
4. Las recomendaciones son los N cursos con mayor score predicho que el estudiante aún no tomó

**Métricas de evaluación:**

| Métrica | Valor |
|---------|-------|
| RMSE | 0.79 |
| MAE | 0.60 |
| Precision@10 | 9.8% (~9x mejor que recomendar por popularidad) |
| Varianza explicada | 35.4% |

> La evaluación se hace sobre el 20% de ratings explícitos reservados como test set.

### Datos

- **83 cursos** distribuidos en 15 escuelas de Platzi
- **4,891 estudiantes** con historial de interacciones
- **44,340 interacciones** (ratings + feedback de progreso)
- Sparsidad de la matriz: 88.9% — típica en plataformas de e-learning

### Dashboard

El dashboard permite explorar recomendaciones por estudiante en tiempo real:

- Selector de estudiante con perfil completo (plan, escuela, Platzi Rank, nivel Rewards)
- Historial de cursos tomados con progreso y rating
- Top-N cursos recomendados con score SVD predicho
- Filtros por escuela y dificultad
- Gráfico de scores con comparación visual
- Sección de información del modelo (varianza explicada, RMSE vs baselines)

### Stack

```
Python · pandas · numpy · scikit-learn · Streamlit · Plotly
Algoritmo: TruncatedSVD (k=20 factores latentes)
Feedback: híbrido — ratings explícitos + progreso implícito
```

### Estructura

```
├── data/raw/
│   ├── platzi_courses.csv          # Catálogo de 83 cursos
│   └── platzi_interactions.csv     # 44,340 interacciones estudiante-curso
├── notebooks/
│   ├── 03_recommender_eda.ipynb    # Análisis exploratorio (sparsidad, ratings, popularidad)
│   └── 04_recommender_modelo.ipynb # SVD, evaluación vs baselines, recomendaciones
├── app/pages/02_Recommender.py     # Dashboard interactivo
└── models/
    ├── recommender_model.pkl       # Modelo SVD + factores latentes
    └── recommender_top10.csv       # Top-10 precomputado por estudiante
```

---

## Instalación y uso

```bash
# 1. Clonar el repositorio
git clone https://github.com/KevinLarguia/ProyectoMLPlatzi.git
cd ProyectoMLPlatzi

# 2. Crear entorno virtual e instalar dependencias
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit joblib notebook

# 3. Generar los modelos (ejecutar notebooks en orden)
# notebooks/01_churn_eda.ipynb
# notebooks/02_churn_modelo.ipynb
# notebooks/03_recommender_eda.ipynb
# notebooks/04_recommender_modelo.ipynb

# 4. Levantar la plataforma
streamlit run app/main.py
```

---

## Próximamente

- Módulo 3: segmentación de usuarios con K-Means y PCA
- Módulo 4: predicción de conversión de gratuito a pago
- Módulo 5: análisis de engagement con alertas tempranas
