import re
import numpy as np
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_DISPONIBLES = True
except ImportError:
    EMBEDDINGS_DISPONIBLES = False

st.set_page_config(
    page_title="Nexia — Motor de Conocimiento & Grafo",
    page_icon="🧠",
    layout="wide",
)

COLOR_CAT = {"Política": "#e74c3c", "Técnico": "#3498db", "Operativo": "#2ecc71"}
COLOR_ACT = "#f39c12"
COLOR_TEC = "#9b59b6"


def limpiar(nombre: str) -> str:
    return nombre.replace("_", " ")


def cargar_base_conocimiento() -> pd.DataFrame:
    datos = [
        # --- Políticas ---
        {"id": "DOC-01", "categoria": "Política", "actividad": "Diseño_Arquitectura_Software", "tecnologia": "GitHub",
         "texto": "Guía de estilo y buenas prácticas para escribir código limpio (Clean Code) en Python."},
        {"id": "DOC-02", "categoria": "Política", "actividad": "Implementación_Machine_Learning", "tecnologia": "Base_Conocimiento",
         "texto": "Lineamientos de gobernanza de datos y privacidad de la información del cliente."},
        {"id": "DOC-03", "categoria": "Política", "actividad": "Diseño_Arquitectura_Software", "tecnologia": "GitHub",
         "texto": "Reglamento interno y políticas de la empresa para el uso de GitHub y repositorios."},
        {"id": "DOC-04", "categoria": "Política", "actividad": "Resolución_Incidentes_Críticos", "tecnologia": "Entorno_Cloud",
         "texto": "Políticas de seguridad de la información, manejo de contraseñas y accesos VPN."},
        {"id": "DOC-05", "categoria": "Política", "actividad": "Diseño_Arquitectura_Software", "tecnologia": "Base_Conocimiento",
         "texto": "Reglamento de confidencialidad y acuerdos de no divulgación (NDA) para consultores."},
        {"id": "DOC-06", "categoria": "Política", "actividad": "Diseño_Arquitectura_Software", "tecnologia": "GitHub",
         "texto": "Lineamientos estándar para la revisión de código (Pull Requests) entre pares."},
        {"id": "DOC-07", "categoria": "Política", "actividad": "Resolución_Incidentes_Críticos", "tecnologia": "Base_Conocimiento",
         "texto": "Normativa oficial de la empresa sobre horarios laborales y uso de equipos físicos."},
        {"id": "DOC-08", "categoria": "Política", "actividad": "Resolución_Incidentes_Críticos", "tecnologia": "Base_Conocimiento",
         "texto": "Políticas de uso aceptable de dispositivos personales y corporativos."},

        # --- Técnicos ---
        {"id": "DOC-09", "categoria": "Técnico", "actividad": "Diseño_Arquitectura_Software", "tecnologia": "Entorno_Cloud",
         "texto": "Manual de arquitectura de software para el nuevo cliente corporativo."},
        {"id": "DOC-10", "categoria": "Técnico", "actividad": "Implementación_Machine_Learning", "tecnologia": "Base_Conocimiento",
         "texto": "Diccionario de datos para el pipeline del modelo de Machine Learning."},
        {"id": "DOC-11", "categoria": "Técnico", "actividad": "Diseño_Arquitectura_Software", "tecnologia": "GitHub",
         "texto": "Documentación técnica de la API REST para el manejo de usuarios."},
        {"id": "DOC-12", "categoria": "Técnico", "actividad": "Diseño_Arquitectura_Software", "tecnologia": "Entorno_Cloud",
         "texto": "Diagramas de flujo de datos y arquitectura de microservicios en AWS."},
        {"id": "DOC-13", "categoria": "Técnico", "actividad": "Diseño_Arquitectura_Software", "tecnologia": "Entorno_Cloud",
         "texto": "Esquema relacional de la base de datos PostgreSQL para el módulo de facturación."},
        {"id": "DOC-14", "categoria": "Técnico", "actividad": "Diseño_Arquitectura_Software", "tecnologia": "Entorno_Cloud",
         "texto": "Especificaciones de los endpoints de integración con la pasarela de pagos."},

        # --- Operativos ---
        {"id": "DOC-15", "categoria": "Operativo", "actividad": "Resolución_Incidentes_Críticos", "tecnologia": "Entorno_Cloud",
         "texto": "Reporte post-mortem del fallo en el clúster de producción de AWS."},
        {"id": "DOC-16", "categoria": "Operativo", "actividad": "Diseño_Arquitectura_Software", "tecnologia": "Contenedor",
         "texto": "Guía de despliegue de contenedores utilizando Docker y Kubernetes."},
        {"id": "DOC-17", "categoria": "Operativo", "actividad": "Resolución_Incidentes_Críticos", "tecnologia": "Entorno_Cloud",
         "texto": "Ticket de resolución de incidente crítico de latencia en la base de datos."},
        {"id": "DOC-18", "categoria": "Operativo", "actividad": "Diseño_Arquitectura_Software", "tecnologia": "Base_Conocimiento",
         "texto": "Manual de usuario paso a paso para el sistema interno de gestión de viáticos."},
        {"id": "DOC-19", "categoria": "Operativo", "actividad": "Resolución_Incidentes_Críticos", "tecnologia": "Entorno_Cloud",
         "texto": "Bitácora de mantenimiento semanal y rotación de credenciales en servidores de desarrollo."},
        {"id": "DOC-20", "categoria": "Operativo", "actividad": "Resolución_Incidentes_Críticos", "tecnologia": "Base_Conocimiento",
         "texto": "Reporte mensual de métricas de incidencias (SLA) y tiempos de respuesta de soporte."},
        {"id": "DOC-21", "categoria": "Operativo", "actividad": "Resolución_Incidentes_Críticos", "tecnologia": "Contenedor",
         "texto": "Guía rápida de soporte para reiniciar servicios y recuperar aplicaciones web caídas."},
    ]
    return pd.DataFrame(datos)


class SistemaRecomendacionNexia:
    def __init__(self, df_docs: pd.DataFrame, usar_embeddings: bool = True):
        self.df = df_docs.reset_index(drop=True).copy()
        self.usar_embeddings = usar_embeddings and EMBEDDINGS_DISPONIBLES

        corpus_enriquecido = (
            self.df["categoria"] + ": " + self.df["texto"] + " (" + self.df["tecnologia"] + ")"
        ).tolist()

        if self.usar_embeddings:
            self.modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            self.vectores_docs = self.modelo.encode(corpus_enriquecido, normalize_embeddings=True)
            self.modo = "Embeddings Semánticos (Sentence-BERT)"
        else:
            stop_words_es = [
                "de", "la", "el", "los", "las", "y", "en", "para", "del", "al",
                "un", "una", "con", "por", "sobre", "entre", "o", "a", "que", "se", "su", "sus", "es"
            ]
            self.vectorizador = TfidfVectorizer(
                stop_words=stop_words_es,
                strip_accents="unicode",
                ngram_range=(1, 2)
            )
            self.vectores_docs = self.vectorizador.fit_transform(corpus_enriquecido).toarray()
            self.modo = "TF-IDF (N-gramas)"

        self.matriz_similitud_docs = cosine_similarity(self.vectores_docs)

    def _vectorizar_textos(self, textos: list[str]) -> np.ndarray:
        if self.usar_embeddings:
            return self.modelo.encode(textos, normalize_embeddings=True)
        return self.vectorizador.transform(textos).toarray()

    def agregar_documento(self, texto: str, categoria: str, actividad: str = "General", tecnologia: str = "Base_Conocimiento"):
        nuevo_id = f"DOC-{len(self.df) + 1:02d}"
        nueva_fila = pd.DataFrame([{
            "id": nuevo_id, "categoria": categoria, "actividad": actividad,
            "tecnologia": tecnologia, "texto": texto
        }])
        self.df = pd.concat([self.df, nueva_fila], ignore_index=True)
        self.__init__(self.df, usar_embeddings=self.usar_embeddings)

    def recomendar_por_documento(self, doc_id: str, top_k: int = 3, boost_ontologia: float = 0.05) -> pd.DataFrame:
        if doc_id not in self.df["id"].values:
            raise ValueError(f"El ID '{doc_id}' no existe en la base de conocimiento.")

        idx = self.df.index[self.df["id"] == doc_id][0]
        doc_base = self.df.iloc[idx]

        scores = self.matriz_similitud_docs[idx].copy()
        misma_actividad = (self.df["actividad"] == doc_base["actividad"]).astype(float) * boost_ontologia
        misma_tecnologia = (self.df["tecnologia"] == doc_base["tecnologia"]).astype(float) * boost_ontologia
        scores_finales = scores + misma_actividad + misma_tecnologia

        resultados = self.df.copy()
        resultados["similitud_semantica"] = scores
        resultados["score_final"] = scores_finales
        resultados = resultados[resultados["id"] != doc_id]
        return resultados.sort_values("score_final", ascending=False).head(top_k)

    def recomendar_por_contexto(
        self,
        historial_consultas: list[str],
        docs_vistos: list[str] | None = None,
        top_k: int = 3,
        umbral: float = 0.15
    ) -> pd.DataFrame:
        docs_vistos = docs_vistos or []
        vectores_contexto = []
        pesos = []

        if historial_consultas:
            vecs_q = self._vectorizar_textos(historial_consultas)
            n_q = len(historial_consultas)
            for i, vec in enumerate(vecs_q):
                peso_recencia = 0.6 ** (n_q - 1 - i)
                vectores_contexto.append(vec)
                pesos.append(peso_recencia)

        for doc_id in docs_vistos:
            if doc_id in self.df["id"].values:
                idx = self.df.index[self.df["id"] == doc_id][0]
                vectores_contexto.append(self.vectores_docs[idx])
                pesos.append(0.5)

        if not vectores_contexto:
            return pd.DataFrame()

        perfil_usuario = np.average(vectores_contexto, axis=0, weights=pesos).reshape(1, -1)
        similitudes = cosine_similarity(perfil_usuario, self.vectores_docs).flatten()

        resultados = self.df.copy()
        resultados["afinidad_contexto"] = similitudes
        resultados = resultados[~resultados["id"].isin(docs_vistos)]
        resultados = resultados[resultados["afinidad_contexto"] >= umbral]
        return resultados.sort_values("afinidad_contexto", ascending=False).head(top_k)


class ChatbotNexia:
    def __init__(self, motor_recomendacion: SistemaRecomendacionNexia):
        self.motor = motor_recomendacion
        self.historial_consultas: list[str] = []

    def reiniciar_conversacion(self) -> str:
        self.historial_consultas.clear()
        return "Listo, limpié el historial de la conversación. ¿Sobre qué otro tema de Nexia quieres consultar?"

    def responder(self, mensaje_usuario: str) -> str:
        texto = mensaje_usuario.strip()
        if not texto:
            return "Escribe una pregunta o el ID de un documento (por ejemplo, `DOC-15`) para ayudarte."

        texto_lower = texto.lower()

        if any(cmd in texto_lower for cmd in ["limpiar", "reiniciar", "nuevo tema", "borrar historial"]):
            return self.reiniciar_conversacion()

        if re.fullmatch(r"(hola|buenos dias|buenos días|buenas tardes|buenas noches|que tal|qué tal|hey)[!?. ]*", texto_lower):
            return (
                "¡Hola! Soy el asistente de conocimiento de **Nexia Analytics & Software**. "
                "Puedo ayudarte a encontrar políticas internas, documentación técnica o guías operativas.\n\n"
                "Ejemplos de lo que puedes preguntarme:\n"
                "- *¿Cómo desplegar contenedores en Docker y Kubernetes?*\n"
                "- *Políticas de seguridad y contraseñas*\n"
                "- *Muéstrame el DOC-15*"
            )

        if texto_lower in ["ayuda", "help", "comandos", "que puedes hacer", "¿qué puedes hacer?"]:
            return (
                "Puedo buscar información por lenguaje natural usando contexto de la charla o buscar un documento exacto por su código (`DOC-01` al `DOC-21`). "
                "Si cambias de tema y quieres evitar que mezcle preguntas anteriores, escribe **reiniciar**."
            )

        if any(desp in texto_lower for desp in ["gracias", "muchas gracias", "adios", "adiós", "bye", "hasta luego"]):
            return "¡Con gusto! Si necesitas consultar otro documento o procedimiento, aquí sigo."

        match_doc = re.search(r"\bdoc-\d{2}\b", texto_lower)
        if match_doc:
            doc_id = match_doc.group(0).upper()
            try:
                doc_info = self.motor.df[self.motor.df["id"] == doc_id].iloc[0]
                relacionados = self.motor.recomendar_por_documento(doc_id, top_k=2)

                respuesta = (
                    f"Aquí tienes la información del **{doc_id}**:\n"
                    f"- **Categoría:** {doc_info['categoria']} | **Tecnología:** {doc_info['tecnologia']}\n"
                    f"- **Descripción:** {doc_info['texto']}\n\n"
                    f"**Documentos relacionados que te pueden servir:**\n"
                )
                for _, row in relacionados.iterrows():
                    respuesta += f"- **[{row['id']}]** ({row['categoria']}) — {row['texto']} *(Score: {row['score_final']:.2f})*\n"
                return respuesta
            except (IndexError, ValueError):
                return f"No encontré ningún documento con el identificador **{doc_id}**. Los IDs disponibles van del `DOC-01` al `DOC-21`."

        self.historial_consultas.append(texto)
        contexto_reciente = self.historial_consultas[-3:]

        resultados = self.motor.recomendar_por_contexto(
            historial_consultas=contexto_reciente,
            docs_vistos=[],
            top_k=3,
            umbral=0.15
        )

        if resultados.empty:
            self.historial_consultas.pop()
            return (
                "No encontré documentos que superen el umbral de relevancia para esa consulta. "
                "Intenta incluir términos como *AWS, Docker, GitHub, API REST, PostgreSQL, seguridad, incidente* o *viáticos*."
            )

        respuesta = "Encontré estos documentos relevantes en la base de conocimiento:\n\n"
        for _, row in resultados.iterrows():
            respuesta += (
                f"- **[{row['id']}]** `{row['categoria']}` · `{row['tecnologia']}` "
                f"(Afinidad: **{row['afinidad_contexto']:.2f}**)\n"
                f"  {row['texto']}\n"
            )
        respuesta += "\n¿Quieres profundizar en alguno o consultar un ID específico?"
        return respuesta


def construir_grafo(recomendador: SistemaRecomendacionNexia, vecinos_por_doc: int = 2) -> nx.Graph:
    df = recomendador.df
    G = nx.Graph()

    for cat, col in COLOR_CAT.items():
        G.add_node(f"cat:{cat}", label=cat, title=f"Categoría: {cat}",
                   color=col, size=45, shape="dot")
    for act in df["actividad"].unique():
        G.add_node(f"act:{act}", label=limpiar(act), title=f"Actividad: {limpiar(act)}",
                   color=COLOR_ACT, size=32, shape="triangle")
    for tec in df["tecnologia"].unique():
        G.add_node(f"tec:{tec}", label=limpiar(tec), title=f"Tecnología: {limpiar(tec)}",
                   color=COLOR_TEC, size=32, shape="square")

    for _, fila in df.iterrows():
        G.add_node(
            fila["id"], label=fila["id"], shape="dot", size=16,
            color=COLOR_CAT[fila["categoria"]],
            title=(f"{fila['id']} · {fila['categoria']}\n{fila['texto']}\n"
                   f"Actividad: {limpiar(fila['actividad'])}\n"
                   f"Tecnología: {limpiar(fila['tecnologia'])}"),
        )
        G.add_edge(fila["id"], f"cat:{fila['categoria']}", title="pertenece_a", color="#777777")
        G.add_edge(fila["id"], f"act:{fila['actividad']}", title="parte_de_actividad", color=COLOR_ACT)
        G.add_edge(fila["id"], f"tec:{fila['tecnologia']}", title="usa_tecnología", color=COLOR_TEC)

    sim = recomendador.matriz_similitud_docs
    for i in range(len(df)):
        orden = sim[i].argsort()[::-1]
        for j in [k for k in orden if k != i][:vecinos_por_doc]:
            a, b = df.iloc[i]["id"], df.iloc[j]["id"]
            if not G.has_edge(a, b):
                G.add_edge(a, b, title=f"similar_a ({float(sim[i, j]):.2f})",
                           color="#bbbbbb", dashes=True, width=1 + 2 * float(sim[i, j]))
    return G


def generar_html_grafo(G: nx.Graph, doc_foco: str | None = None, recomendador: SistemaRecomendacionNexia | None = None) -> str:
    H = G.copy()
    if doc_foco and recomendador is not None:
        recs = recomendador.recomendar_por_documento(doc_foco, top_k=3)["id"].tolist()
        for n, borde in [(doc_foco, "#ffffff")] + [(r, "#ffd700") for r in recs]:
            if n in H.nodes:
                H.nodes[n]["color"] = {"background": H.nodes[n]["color"], "border": borde}
                H.nodes[n]["borderWidth"] = 4
                H.nodes[n]["size"] = 26
        for r in recs:
            H.add_edge(doc_foco, r, color="#ff4d4d", width=4, title="recomendado")

    net = Network(height="720px", width="100%", bgcolor="#1e1e1e",
                  font_color="white", cdn_resources="in_line")
    net.from_nx(H)
    net.force_atlas_2based(gravity=-60, spring_length=110)
    return net.generate_html()


@st.cache_resource
def inicializar_motor(usar_embeddings: bool) -> SistemaRecomendacionNexia:
    df_base = cargar_base_conocimiento()
    return SistemaRecomendacionNexia(df_base, usar_embeddings=usar_embeddings)


with st.sidebar:
    st.title("⚙️ Configuración Nexia")
    usar_emb = st.toggle("Intentar usar Sentence-BERT", value=EMBEDDINGS_DISPONIBLES)
    recomendador = inicializar_motor(usar_emb)
    st.info(f"**Motor activo:** {recomendador.modo}")
    st.metric("Documentos indexados", len(recomendador.df))

if "bot" not in st.session_state or st.session_state.get("modo_bot") != recomendador.modo:
    st.session_state.bot = ChatbotNexia(recomendador)
    st.session_state.modo_bot = recomendador.modo
    st.session_state.mensajes_chat = [
        {"role": "assistant", "content": st.session_state.bot.responder("hola")}
    ]

if "docs_vistos" not in st.session_state:
    st.session_state.docs_vistos = []

st.title("🧠 Nexia Analytics & Software — Sistema de Conocimiento")
st.caption("Recomendación semántica y ontológica, asistente conversacional y grafo de conocimiento")

tab_chat, tab_rec, tab_grafo, tab_base = st.tabs([
    "💬 Chatbot Nexia",
    "🎯 Recomendación (Doc y Contexto)",
    "🕸️ Grafo de Conocimiento",
    "📚 Base de Conocimiento",
])

with tab_chat:
    st.subheader("Asistente Conversacional de Conocimiento")
    st.caption("Prueba escribir una duda técnica, preguntar por políticas o pedir un código como `DOC-15` o `reiniciar`.")

    for msg in st.session_state.mensajes_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Escribe tu consulta o ID de documento (ej. DOC-15)...")
    if prompt:
        st.session_state.mensajes_chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        respuesta = st.session_state.bot.responder(prompt)
        st.session_state.mensajes_chat.append({"role": "assistant", "content": respuesta})
        with st.chat_message("assistant"):
            st.markdown(respuesta)

with tab_rec:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Mecanismo A: Similitud entre Documentos")
        doc_seleccionado = st.selectbox(
            "Selecciona un documento base:",
            options=recomendador.df["id"].tolist(),
            format_func=lambda x: f"{x} — {recomendador.df.loc[recomendador.df['id'] == x, 'texto'].iloc[0][:55]}..."
        )
        top_k_doc = st.slider("Cantidad de recomendaciones", 1, 5, 3, key="topk_doc")

        if st.button("Recomendar relacionados", type="primary"):
            if doc_seleccionado not in st.session_state.docs_vistos:
                st.session_state.docs_vistos.append(doc_seleccionado)
            recs = recomendador.recomendar_por_documento(doc_seleccionado, top_k=top_k_doc)
            st.dataframe(
                recs[["id", "categoria", "actividad", "tecnologia", "score_final", "texto"]],
                use_container_width=True,
                hide_index=True,
            )

    with col2:
        st.subheader("Mecanismo B: Recomendación por Contexto")
        historial_texto = st.text_area(
            "Historial de consultas (una consulta por línea, de más antigua a más reciente):",
            value="cómo estructurar tablas de clientes\nmodelos predictivos y variables de entrada",
            height=110,
        )
        umbral_ctx = st.slider("Umbral mínimo de afinidad", 0.0, 0.8, 0.15, 0.05)

        if st.button("Recomendar por contexto"):
            consultas = [line.strip() for line in historial_texto.splitlines() if line.strip()]
            recs_ctx = recomendador.recomendar_por_contexto(
                historial_consultas=consultas,
                docs_vistos=st.session_state.docs_vistos,
                top_k=3,
                umbral=umbral_ctx,
            )
            if recs_ctx.empty:
                st.warning("No hubo documentos que superaran el umbral con ese contexto.")
            else:
                st.dataframe(
                    recs_ctx[["id", "categoria", "actividad", "tecnologia", "afinidad_contexto", "texto"]],
                    use_container_width=True,
                    hide_index=True,
                )

with tab_grafo:
    st.subheader("Grafo Ontológico y Semántico")
    col_g1, col_g2 = st.columns([1, 2])
    with col_g1:
        vecinos = st.slider("Vecinos semánticos por documento", 1, 4, 2)
        foco_opciones = ["Ninguno (Grafo completo)"] + recomendador.df["id"].tolist()
        doc_foco = st.selectbox("Resaltar documento y sus recomendaciones:", foco_opciones)
    with col_g2:
        st.markdown("**Leyenda:** 🔴 Política · 🔵 Técnico · 🟢 Operativo · 🟠 Actividad · 🟣 Tecnología")

    G = construir_grafo(recomendador, vecinos_por_doc=vecinos)
    html_grafo = generar_html_grafo(
        G,
        doc_foco=None if doc_foco == "Ninguno (Grafo completo)" else doc_foco,
        recomendador=recomendador,
    )
    components.html(html_grafo, height=740, scrolling=True)

with tab_base:
    st.subheader("Documentos en la Base de Conocimiento")
    st.dataframe(recomendador.df, use_container_width=True, hide_index=True)