# Platzi ML Platform

Plataforma de Machine Learning aplicada a una empresa edtech real.  
5 módulos que resuelven problemas de negocio concretos sobre el comportamiento de estudiantes.

---

## Módulos

| # | Módulo | Descripción | Estado |
|---|--------|-------------|--------|
| 1 | **Churn Prediction** | Detecta estudiantes en riesgo de cancelar antes de que ocurra | ✅ Completo |
| 2 | Course Recommender | Sugiere el próximo curso según historial y perfil | 🔄 En progreso |
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
<img width="1917" height="915" alt="capturadash" src="https://github.com/user-attachments/assets/5d361484-3f68-4bfe-ac6c-cec8168eaea9" />
El dashboard corre sobre toda la base de estudiantes en tiempo real:

- Métricas globales de riesgo
- Distribución de probabilidad de abandono
- Riesgo segmentado por Plan, País y Escuela
- Tabla interactiva con filtros por nivel de riesgo
- Exportación de listas para campañas de retención


### Stack Tecnologico

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

### Instalación y uso

```bash
# 1. Clonar el repositorio
git clone https://github.com/KevinLarguia/ProyectoMLPlatzi.git
cd ProyectoMLPlatzi

# 2. Crear entorno virtual e instalar dependencias
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit joblib notebook

# 3. Entrenar el modelo
# Abrir y ejecutar en orden:
# notebooks/01_churn_eda.ipynb
# notebooks/02_churn_modelo.ipynb

# 4. Levantar la plataforma
streamlit run app/main.py
```
---

## Próximamente

- Módulo 2: sistema de recomendación de cursos personalizado (En progreso)
- Módulo 3: segmentación de usuarios con clustering
- Módulo 4: predicción de conversión de gratuito a pago
- Módulo 5: análisis de engagement con alertas tempranas
