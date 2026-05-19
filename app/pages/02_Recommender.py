import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")

# ── CARGA DE DATOS ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    courses  = pd.read_csv("data/raw/platzi_courses.csv")
    inter    = pd.read_csv("data/raw/platzi_interactions.csv")
    students = pd.read_csv("data/raw/platzi_churn.csv")[
        ["student_id", "plan", "pais", "escuela_principal",
         "platzi_rank", "nivel_rewards", "tasa_completado", "meses_suscrito"]
    ]
    top10    = pd.read_csv("models/recommender_top10.csv")
    return courses, inter, students, top10

@st.cache_resource
def load_model():
    with open("models/recommender_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    df_courses, df_int, df_students, df_top10 = load_data()
    model = load_model()
    model_ok = True
except FileNotFoundError as e:
    model_ok = False
    missing = str(e)

# ── HEADER ─────────────────────────────────────────────────────────────────────

st.title("🎓 Course Recommender")
st.markdown("**Filtrado colaborativo con SVD** — Cursos personalizados para cada estudiante")
st.divider()

if not model_ok:
    st.error(f"Archivo no encontrado: {missing}")
    st.info("Ejecuta primero:\n1. `src/recommender/generate_data.py`\n2. `notebooks/04_recommender_modelo.ipynb`")
    st.stop()

# ── MÉTRICAS GLOBALES ──────────────────────────────────────────────────────────

metrics = model.get("metrics", {})
rmse      = metrics.get('rmse', 0)
rmse_base = metrics.get('rmse_bl_global', 0)
mejora    = (1 - rmse / rmse_base) * 100 if rmse_base > 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Cursos en catálogo",   len(df_courses))
col2.metric("Estudiantes",          f"{df_int['student_id'].nunique():,}")
col3.metric("Interacciones",        f"{len(df_int):,}")
col4.metric("RMSE del modelo",      f"{rmse:.3f}", delta=f"{-mejora:.1f}% vs baseline", delta_color="inverse")
col5.metric("Precision@10",         f"{metrics.get('precision_at_10', 0):.3f}")

st.divider()

# ── SIDEBAR ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuración")

    # Selector de estudiante
    all_students = sorted(df_int["student_id"].unique())
    student_id = st.selectbox("Selecciona un estudiante", all_students, index=0)

    st.subheader("Filtrar recomendaciones")
    schools_available = ["Todas"] + sorted(df_courses["school"].unique())
    school_filter = st.selectbox("Escuela", schools_available)

    diff_filter = st.multiselect(
        "Dificultad",
        ["Basico", "Intermedio", "Avanzado"],
        default=["Basico", "Intermedio", "Avanzado"],
    )

    n_recs = st.slider("Número de recomendaciones", min_value=3, max_value=15, value=8)

# ── PERFIL DEL ESTUDIANTE ──────────────────────────────────────────────────────

student_info = df_students[df_students["student_id"] == student_id]

col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader(f"👤 {student_id}")

    if not student_info.empty:
        s = student_info.iloc[0]
        st.markdown(f"""
| Campo | Valor |
|-------|-------|
| Plan | **{s['plan']}** |
| País | {s['pais']} |
| Escuela principal | {s['escuela_principal'].replace('Escuela de ', '')} |
| Nivel Rewards | **{s['nivel_rewards']}** |
| Platzi Rank | {int(s['platzi_rank']):,} PR |
| Meses suscrito | {int(s['meses_suscrito'])} |
| Tasa completado | {s['tasa_completado']:.0%} |
""")

    # Historial del estudiante
    hist = (df_int[df_int["student_id"] == student_id]
            .merge(df_courses[["course_id", "name", "school", "difficulty"]], on="course_id")
            .sort_values("progress", ascending=False))

    st.subheader(f"📚 Historial ({len(hist)} cursos)")

    for _, row in hist.head(10).iterrows():
        icon  = "✅" if row["completed"] else "🔄"
        stars = f"⭐ {row['rating']:.1f}" if pd.notna(row["rating"]) else ""
        st.markdown(
            f"{icon} **{row['name']}**  \n"
            f"&nbsp;&nbsp;&nbsp;&nbsp;`{row['school']}` · {row['difficulty']} · "
            f"{row['progress']}% {stars}"
        )

    if len(hist) > 10:
        st.caption(f"... y {len(hist)-10} cursos más")

# ── RECOMENDACIONES ────────────────────────────────────────────────────────────

with col_right:
    st.subheader("🚀 Cursos Recomendados")

    # Obtener recomendaciones precomputadas
    recs = df_top10[df_top10["student_id"] == student_id].copy()
    recs = recs.merge(df_courses, on="course_id")

    # Aplicar filtros
    if school_filter != "Todas":
        recs = recs[recs["school"] == school_filter]
    if diff_filter:
        recs = recs[recs["difficulty"].isin(diff_filter)]

    recs = recs.head(n_recs)

    if recs.empty:
        st.warning("Sin recomendaciones para los filtros seleccionados. Prueba cambiando los filtros.")
    else:
        # Gráfico de barras con scores
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=recs["score"],
            y=recs["name"],
            orientation="h",
            marker=dict(
                color=recs["score"],
                colorscale="Blues",
                showscale=False,
            ),
            text=[f"{s:.2f}" for s in recs["score"]],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Score predicho: %{x:.2f}<br>"
                "<extra></extra>"
            ),
        ))
        fig.update_layout(
            xaxis_title="Score predicho (escala 1-5)",
            xaxis_range=[0, 5.5],
            yaxis=dict(autorange="reversed"),
            height=max(300, n_recs * 45),
            margin=dict(l=10, r=60, t=10, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tabla detallada
        st.subheader("Detalle")
        for _, row in recs.iterrows():
            diff_color = {"Basico": "green", "Intermedio": "orange", "Avanzado": "red"}.get(
                row["difficulty"], "gray"
            )
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 1, 1])
                c1.markdown(f"**{row['name']}**  \n`{row['school']}` · :{diff_color}[{row['difficulty']}]")
                c2.metric("Score SVD", f"{row['score']:.2f}")
                c3.metric("Rating prom.", f"⭐ {row['avg_rating']:.1f}")

st.divider()

# ── ANÁLISIS DEL CATÁLOGO ──────────────────────────────────────────────────────

with st.expander("📊 Explorar Catálogo de Cursos"):
    tab1, tab2 = st.tabs(["Por Escuela", "Por Dificultad"])

    with tab1:
        school_pop = (df_int.merge(df_courses[["course_id", "school"]], on="course_id")
                      .groupby("school").size()
                      .reset_index(name="interacciones")
                      .sort_values("interacciones", ascending=False))
        fig2 = px.bar(school_pop, x="interacciones", y="school", orientation="h",
                      color="interacciones", color_continuous_scale="Blues",
                      title="Interacciones por Escuela")
        fig2.update_layout(showlegend=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        diff_counts = df_courses["difficulty"].value_counts()
        fig3 = px.pie(values=diff_counts.values, names=diff_counts.index,
                      color_discrete_sequence=["#4CAF50", "#FF9800", "#F44336"],
                      title="Distribución por Dificultad")
        st.plotly_chart(fig3, use_container_width=True)

with st.expander("🔬 Información del Modelo SVD"):
    ev = model.get("explained_variance", np.zeros(20))
    cumev = np.cumsum(ev) * 100
    fig4 = px.line(
        x=list(range(1, len(ev)+1)), y=cumev,
        markers=True,
        labels={"x": "Componentes SVD", "y": "Varianza explicada acumulada (%)"},
        title="Varianza Explicada Acumulada por Componente"
    )
    fig4.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80%")
    st.plotly_chart(fig4, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
    m2.metric("MAE",  f"{metrics.get('mae',  0):.4f}")
    st.caption("RMSE y MAE se calculan sobre el 20% de datos de test (interacciones con rating explícito).")
