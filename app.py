import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# =====================================================================
# HELPERS UI
# =====================================================================
def card(icon, titulo, valor, subtitulo="", color="#0D9488", bg="#F0FDF4", border="#BBF7D0"):
    return f"""
    <div style="background:{bg};border:1.5px solid {border};border-left:5px solid {color};
                border-radius:10px;padding:18px 22px;margin-bottom:4px;">
        <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;
                    letter-spacing:.08em;margin-bottom:6px;">{icon} {titulo}</div>
        <div style="font-size:28px;font-weight:800;color:{color};line-height:1.1;">{valor}</div>
        {"<div style='font-size:12px;color:#9CA3AF;margin-top:4px;'>" + subtitulo + "</div>" if subtitulo else ""}
    </div>"""

def delta_card(icon, titulo, valor, delta, subtitulo="", color="#0D9488", bg="#F0FDF4", border="#BBF7D0"):
    signo   = "▲" if delta >= 0 else "▼"
    d_color = "#059669" if delta >= 0 else "#DC2626"
    return f"""
    <div style="background:{bg};border:1.5px solid {border};border-left:5px solid {color};
                border-radius:10px;padding:18px 22px;margin-bottom:4px;">
        <div style="font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;
                    letter-spacing:.08em;margin-bottom:6px;">{icon} {titulo}</div>
        <div style="font-size:28px;font-weight:800;color:{color};line-height:1.1;">{valor}</div>
        <div style="font-size:13px;color:{d_color};font-weight:700;margin-top:4px;">
            {signo} {abs(delta):.1f}% vs año anterior</div>
        {"<div style='font-size:11px;color:#9CA3AF;margin-top:2px;'>" + subtitulo + "</div>" if subtitulo else ""}
    </div>"""

def badge_ganador(nombre, acc, diff):
    return f"""
    <div style="background:linear-gradient(135deg,#1E3A8A,#2563EB);border-radius:12px;
                padding:22px 26px;text-align:center;color:#fff;margin-bottom:4px;">
        <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                    letter-spacing:.1em;opacity:.75;margin-bottom:8px;">🏆 Modelo Óptimo</div>
        <div style="font-size:22px;font-weight:800;margin-bottom:6px;">{nombre}</div>
        <div style="font-size:13px;opacity:.85;">Accuracy: <b>{acc:.1%}</b></div>
        <div style="display:inline-block;background:rgba(255,255,255,.2);border-radius:20px;
                    padding:3px 12px;font-size:12px;margin-top:8px;">+{diff:.1%} sobre el rival</div>
    </div>"""

# =====================================================================
# CLASE POO
# =====================================================================
class AnalizadorComercial:
    def __init__(self, df):
        self.df = df.copy()
        if 'fecha' in self.df.columns:
            self.df['fecha'] = pd.to_datetime(self.df['fecha'])

    def kpis_generales(self, anio_sel=None):
        df = self.df[self.df['anio']==anio_sel] if anio_sel else self.df
        venta   = df['venta_neta'].sum()        if 'venta_neta' in df.columns else 0
        units   = df['unidades_vendidas'].sum() if 'unidades_vendidas' in df.columns else 0
        ticket  = df['venta_neta'].mean()       if 'venta_neta' in df.columns else 0
        pct     = df['cumplimiento_meta'].mean()*100 if 'cumplimiento_meta' in df.columns else None
        return venta, units, ticket, pct

    def ytd(self, anio, mes_corte):
        """YTD acumulado hasta mes_corte para un año dado."""
        mask = (self.df['anio']==anio) & (self.df['mes']<=mes_corte)
        return self.df[mask]['venta_neta'].sum()

    def media_movil_mensual(self, ventana=3):
        """Media móvil de venta neta mensual."""
        df = self.df.copy()
        df['anio_mes'] = df['anio'].astype(str)+'-'+df['mes'].astype(str).str.zfill(2)
        serie = df.groupby(['anio','mes'])['venta_neta'].sum().reset_index()
        serie = serie.sort_values(['anio','mes'])
        serie['fecha_mes'] = pd.to_datetime(
            serie['anio'].astype(str)+'-'+serie['mes'].astype(str).str.zfill(2)+'-01')
        serie[f'MM{ventana}'] = serie['venta_neta'].rolling(ventana, min_periods=1).mean()
        return serie

    def crecimiento_anual(self):
        anual = self.df.groupby('anio')['venta_neta'].sum()
        return anual

    def detectar_outliers_iqr(self, columna, factor=1.5):
        q1  = self.df[columna].quantile(0.25)
        q3  = self.df[columna].quantile(0.75)
        iqr = q3 - q1
        li, ls = q1-factor*iqr, q3+factor*iqr
        return self.df[(self.df[columna]<li)|(self.df[columna]>ls)], li, ls

    def preparar_modelo(self, target):
        df = self.df.copy()
        le = LabelEncoder()
        for col in df.select_dtypes(include=['object','str']).columns:
            df[col] = le.fit_transform(df[col].astype(str))
        if 'fecha' in df.columns:
            df = df.drop(columns=['fecha'])
        # Se excluyen variables con data leakage directo o derivado de la variable objetivo:
        # - venta_bruta y venta_neta: calculadas a partir de unidades * precio, correlación lineal directa
        # - meta_unidades: parte de la ecuación que define cumplimiento_meta (target)
        # Mantener estas variables permitiría al modelo "ver la respuesta" y obtener
        # accuracy artificialmente perfecto, lo que no representa aprendizaje real.
        EXCLUIR_LEAKAGE = ['venta_bruta', 'venta_neta', 'meta_unidades']
        excluir = [target, 'id_registro'] + EXCLUIR_LEAKAGE
        feats   = [c for c in df.columns if c not in excluir]
        return df[feats], df[target]

# =====================================================================
# CONFIG
# =====================================================================
st.set_page_config(page_title="Analizador Comercial — Embutidos",
                   page_icon="🥩", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container{padding-top:1.2rem;}
    [data-testid="stSidebar"]{background-color:#0F2044;}
    [data-testid="stSidebar"] *{color:#E2E8F0 !important;}
    [data-testid="stSidebar"] .stMarkdown h2{color:#F59E0B !important;}
    [data-testid="stSidebar"] hr{border-color:#1E3A8A;}
    .stTabs [data-baseweb="tab"]{font-weight:600;font-size:13px;}
</style>""", unsafe_allow_html=True)

# =====================================================================
# SIDEBAR
# =====================================================================
st.sidebar.markdown("## 🥩 Panel Comercial")
st.sidebar.markdown("**División Comercial — Embutidos**")
st.sidebar.markdown("---")

uploaded = st.sidebar.file_uploader("📂 Cargar dataset (CSV)", type=["csv"])

@st.cache_data
def cargar_demo():
    return pd.read_csv("ventas_comercial_embutidos_2023_2025.csv")

if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
        st.sidebar.success(f"✅ **{uploaded.name}**")
        st.sidebar.caption(f"{df_raw.shape[0]} registros · {df_raw.shape[1]} columnas")
        es_demo = False
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        df_raw = cargar_demo(); es_demo = True
else:
    df_raw = cargar_demo(); es_demo = True
    st.sidebar.info("Usando dataset de demostración 2023–2025.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filtros Globales")

anios_disp = sorted(df_raw['anio'].unique().tolist()) if 'anio' in df_raw.columns else []
anio_sel = st.sidebar.multiselect("Año", anios_disp, default=anios_disp)

df = df_raw[df_raw['anio'].isin(anio_sel)].copy() if anio_sel else df_raw.copy()

if 'region' in df.columns:
    reg_opts = ['Todas'] + sorted(df_raw['region'].unique().tolist())
    reg_sel  = st.sidebar.selectbox("Región", reg_opts)
    if reg_sel != 'Todas': df = df[df['region']==reg_sel]

if 'canal' in df.columns:
    can_opts = ['Todos'] + sorted(df_raw['canal'].unique().tolist())
    can_sel  = st.sidebar.selectbox("Canal", can_opts)
    if can_sel != 'Todos': df = df[df['canal']==can_sel]

st.sidebar.markdown("---")
st.sidebar.markdown("**Sistema:** Análisis Comercial v1.0")
st.sidebar.markdown("**Paradigmas:** POO + Funcional")
st.sidebar.markdown("**Modelos:** Logistic Regression · Random Forest")

# =====================================================================
# HEADER
# =====================================================================
st.markdown("# 🥩 Analizador Comercial Inteligente — Embutidos")
st.caption("Plataforma de análisis de ventas, crecimiento YTD, medias móviles y detección de anomalías · División Comercial")
if es_demo:
    st.info("📋 Visualizando datos de demostración 2023–2025. Sube tu CSV desde el panel lateral.", icon="ℹ️")

analizador = AnalizadorComercial(df_raw)   # siempre sobre datos completos para YTD
analizador_filt = AnalizadorComercial(df)  # datos filtrados para el resto

# =====================================================================
# PESTAÑAS
# =====================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Resumen Ejecutivo",
    "📈 Análisis de Ventas",
    "🚀 Crecimiento YTD & Tendencias",
    "🚨 Auditoría de Anomalías",
    "🤖 Modelo Predictivo"
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — RESUMEN EJECUTIVO
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Indicadores Clave de Desempeño Comercial")

    # KPIs con delta vs año anterior si hay un solo año seleccionado
    venta, units, ticket, pct_meta = analizador_filt.kpis_generales()

    if len(anio_sel)==1:
        anio_ant = anio_sel[0]-1
        venta_ant, units_ant, ticket_ant, _ = analizador.kpis_generales(anio_ant)
        d_venta  = (venta -venta_ant) /max(venta_ant,1)*100
        d_units  = (units -units_ant) /max(units_ant,1)*100
        d_ticket = (ticket-ticket_ant)/max(ticket_ant,1)*100
    else:
        d_venta = d_units = d_ticket = None

    st.markdown("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if d_venta is not None:
            st.markdown(delta_card("💰","Venta Neta Total",f"USD {venta:,.0f}",d_venta,
                "Acumulado período","#0D9488","#F0FDF4","#6EE7B7"), unsafe_allow_html=True)
        else:
            st.markdown(card("💰","Venta Neta Total",f"USD {venta:,.0f}",
                "Acumulado período","#0D9488","#F0FDF4","#6EE7B7"), unsafe_allow_html=True)
    with c2:
        if d_units is not None:
            st.markdown(delta_card("📦","Unidades Vendidas",f"{units:,}",d_units,
                "Volumen total","#2563EB","#EFF6FF","#93C5FD"), unsafe_allow_html=True)
        else:
            st.markdown(card("📦","Unidades Vendidas",f"{units:,}",
                "Volumen total","#2563EB","#EFF6FF","#93C5FD"), unsafe_allow_html=True)
    with c3:
        if d_ticket is not None:
            st.markdown(delta_card("🧾","Ticket Promedio",f"USD {ticket:,.2f}",d_ticket,
                "Venta media por transacción","#7C3AED","#F5F3FF","#C4B5FD"), unsafe_allow_html=True)
        else:
            st.markdown(card("🧾","Ticket Promedio",f"USD {ticket:,.2f}",
                "Venta media por transacción","#7C3AED","#F5F3FF","#C4B5FD"), unsafe_allow_html=True)
    with c4:
        if pct_meta is not None:
            m_col = "#059669" if pct_meta>=70 else "#D97706" if pct_meta>=50 else "#DC2626"
            m_bg  = "#F0FDF4" if pct_meta>=70 else "#FFFBEB" if pct_meta>=50 else "#FEF2F2"
            m_brd = "#6EE7B7" if pct_meta>=70 else "#FCD34D" if pct_meta>=50 else "#FCA5A5"
            st.markdown(card("🎯","Cumplimiento de Meta",f"{pct_meta:.1f}%",
                "Transacciones que alcanzaron meta",m_col,m_bg,m_brd), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        if 'canal' in df.columns:
            st.markdown("#### Venta Neta por Canal")
            df_canal = df.groupby('canal')['venta_neta'].sum().reset_index().sort_values('venta_neta',ascending=True)
            fig = px.bar(df_canal,x='venta_neta',y='canal',orientation='h',
                         color='venta_neta',color_continuous_scale='Teal',
                         labels={'venta_neta':'USD','canal':'Canal'},template='plotly_white')
            fig.update_layout(height=300,showlegend=False,coloraxis_showscale=False)
            fig.update_traces(text=df_canal['venta_neta'].apply(lambda x:f"USD {x:,.0f}"),textposition='outside')
            st.plotly_chart(fig, width='stretch')
    with col_b:
        if 'categoria' in df.columns:
            st.markdown("#### Venta Neta por Categoría")
            df_cat = df.groupby('categoria')['venta_neta'].sum().reset_index()
            fig2 = px.pie(df_cat,values='venta_neta',names='categoria',
                          color_discrete_sequence=px.colors.sequential.Teal,template='plotly_white')
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, width='stretch')

    st.markdown("#### Vista Previa del Dataset")
    st.dataframe(df.head(10), width='stretch')

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — ANÁLISIS DE VENTAS
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Análisis Temporal y por Dimensión Comercial")
    meses_n = {1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',
                7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}

    if 'anio' in df.columns and 'mes' in df.columns:
        st.markdown("#### Evolución Mensual por Año")
        df_mes = df.groupby(['anio','mes']).agg(
            venta_neta=('venta_neta','sum'),
            unidades=('unidades_vendidas','sum')).reset_index()
        df_mes['mes_nombre'] = df_mes['mes'].map(meses_n)
        df_mes['Año'] = df_mes['anio'].astype(str)

        fig_mes = px.line(df_mes, x='mes_nombre', y='venta_neta', color='Año',
                          markers=True, template='plotly_white',
                          labels={'venta_neta':'Venta Neta (USD)','mes_nombre':'Mes'},
                          color_discrete_sequence=['#94A3B8','#0D9488','#F59E0B'])
        fig_mes.update_layout(height=380, legend=dict(orientation='h',y=1.08))
        st.plotly_chart(fig_mes, width='stretch')

    col_c, col_d = st.columns(2)
    with col_c:
        if 'region' in df.columns:
            st.markdown("#### Ventas por Región")
            df_reg = df.groupby('region')['venta_neta'].sum().reset_index().sort_values('venta_neta',ascending=False)
            fig_r = px.bar(df_reg,x='region',y='venta_neta',color='region',text_auto='.2s',
                           template='plotly_white',color_discrete_sequence=px.colors.qualitative.Set2)
            fig_r.update_layout(height=340,showlegend=False)
            st.plotly_chart(fig_r, width='stretch')
    with col_d:
        if 'producto' in df.columns:
            st.markdown("#### Top 10 Productos")
            df_prod = df.groupby('producto')['venta_neta'].sum().reset_index()\
                        .sort_values('venta_neta',ascending=True).tail(10)
            fig_p = px.bar(df_prod,x='venta_neta',y='producto',orientation='h',
                           color='venta_neta',color_continuous_scale='Blues',template='plotly_white')
            fig_p.update_layout(height=340,showlegend=False,coloraxis_showscale=False)
            st.plotly_chart(fig_p, width='stretch')

    if 'canal' in df.columns and 'cumplimiento_meta' in df.columns:
        st.markdown("#### Cumplimiento de Meta por Canal")
        df_meta = df.groupby('canal')['cumplimiento_meta'].mean().reset_index()
        df_meta['pct'] = df_meta['cumplimiento_meta']*100
        fig_m = px.bar(df_meta,x='canal',y='pct',color='canal',text=df_meta['pct'].apply(lambda x:f"{x:.1f}%"),
                       template='plotly_white',color_discrete_sequence=['#059669','#D97706','#2563EB','#7C3AED'])
        fig_m.add_hline(y=70,line_dash='dash',line_color='#059669',annotation_text='Meta 70%')
        fig_m.update_layout(height=350,showlegend=False)
        st.plotly_chart(fig_m, width='stretch')

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — CRECIMIENTO YTD & TENDENCIAS
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🚀 Crecimiento YTD & Análisis de Tendencias")
    st.markdown("")

    anios_completos = sorted(df_raw['anio'].unique().tolist())
    mes_actual = df_raw[df_raw['anio']==max(anios_completos)]['mes'].max()

    # ── YTD comparativo ──────────────────────────────────────────────
    st.markdown("#### 📅 YTD — Comparativo Acumulado al Mismo Período")
    st.caption(f"Comparando acumulado enero → {meses_n[mes_actual]} para cada año disponible")

    ytd_data = []
    for a in anios_completos:
        ytd_v = analizador.ytd(a, mes_actual)
        ytd_data.append({'Año': str(a), 'YTD Venta Neta': ytd_v})
    df_ytd = pd.DataFrame(ytd_data)

    # deltas YTD
    ytd_vals = df_ytd['YTD Venta Neta'].tolist()
    cols_ytd = st.columns(len(anios_completos))
    for i, (_, row) in enumerate(df_ytd.iterrows()):
        with cols_ytd[i]:
            if i > 0:
                delta = (ytd_vals[i]-ytd_vals[i-1])/max(ytd_vals[i-1],1)*100
                st.markdown(delta_card("📅",f"YTD {row['Año']}",
                    f"USD {row['YTD Venta Neta']:,.0f}", delta,
                    f"Ene–{meses_n[mes_actual]}",
                    "#0D9488","#F0FDF4","#6EE7B7"), unsafe_allow_html=True)
            else:
                st.markdown(card("📅",f"YTD {row['Año']}",
                    f"USD {row['YTD Venta Neta']:,.0f}",
                    f"Ene–{meses_n[mes_actual]} (base)",
                    "#6B7280","#F9FAFB","#E5E7EB"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig_ytd = px.bar(df_ytd, x='Año', y='YTD Venta Neta',
                     color='Año', text=df_ytd['YTD Venta Neta'].apply(lambda x:f"USD {x:,.0f}"),
                     color_discrete_sequence=['#94A3B8','#0D9488','#F59E0B'],
                     template='plotly_white', title="YTD Acumulado por Año (mismo período)")
    fig_ytd.update_layout(height=350, showlegend=False)
    fig_ytd.update_traces(textposition='outside')
    st.plotly_chart(fig_ytd, width='stretch')

    st.markdown("---")

    # ── Crecimiento anual ─────────────────────────────────────────────
    st.markdown("#### 📊 Crecimiento Anual de Venta Neta")
    anual = analizador.crecimiento_anual().reset_index()
    anual.columns = ['Año','Venta Neta']
    anual['Año'] = anual['Año'].astype(str)
    anual['Crec %'] = anual['Venta Neta'].pct_change()*100

    fig_anual = go.Figure()
    fig_anual.add_trace(go.Bar(
        x=anual['Año'], y=anual['Venta Neta'],
        name='Venta Neta', marker_color=['#94A3B8','#0D9488','#F59E0B'],
        text=anual['Venta Neta'].apply(lambda x:f"USD {x:,.0f}"),
        textposition='outside'))
    fig_anual.add_trace(go.Scatter(
        x=anual['Año'], y=anual['Crec %'],
        name='Crecimiento %', yaxis='y2', mode='lines+markers+text',
        line=dict(color='#DC2626',width=2.5),
        text=anual['Crec %'].apply(lambda x:f"{x:.1f}%" if not np.isnan(x) else ""),
        textposition='top center'))
    fig_anual.update_layout(
        yaxis=dict(title='Venta Neta (USD)'),
        yaxis2=dict(title='Crecimiento %', overlaying='y', side='right'),
        template='plotly_white', height=380,
        legend=dict(orientation='h', y=1.1))
    st.plotly_chart(fig_anual, width='stretch')

    st.markdown("---")

    # ── Media móvil ───────────────────────────────────────────────────
    st.markdown("#### 📉 Media Móvil de Ventas Mensuales")
    ventana = st.slider("Ventana de la media móvil (meses)", 2, 6, 3)
    serie_mm = analizador.media_movil_mensual(ventana)

    fig_mm = go.Figure()
    # barras por año con colores
    colores_anio = {2023:'#CBD5E1', 2024:'#0D9488', 2025:'#F59E0B'}
    for a in sorted(serie_mm['anio'].unique()):
        sub = serie_mm[serie_mm['anio']==a]
        fig_mm.add_trace(go.Bar(
            x=sub['fecha_mes'], y=sub['venta_neta'],
            name=str(a), marker_color=colores_anio.get(a,'#94A3B8'),
            opacity=0.7))
    # línea media móvil
    fig_mm.add_trace(go.Scatter(
        x=serie_mm['fecha_mes'], y=serie_mm[f'MM{ventana}'],
        name=f'MM {ventana} meses', mode='lines',
        line=dict(color='#DC2626', width=2.5, dash='dot')))
    fig_mm.update_layout(
        title=f"Venta Mensual + Media Móvil {ventana} meses",
        xaxis_title='Mes', yaxis_title='Venta Neta (USD)',
        template='plotly_white', height=400,
        legend=dict(orientation='h', y=1.1), barmode='group')
    st.plotly_chart(fig_mm, width='stretch')

    st.markdown("---")

    # ── Estacionalidad ────────────────────────────────────────────────
    st.markdown("#### 🎄 Índice de Estacionalidad por Mes")
    df_estac = df_raw.groupby(['anio','mes'])['venta_neta'].sum().reset_index()
    prom_anual = df_raw.groupby('anio')['venta_neta'].sum() / 12
    df_estac['indice'] = df_estac.apply(
        lambda r: r['venta_neta'] / prom_anual[r['anio']], axis=1)
    df_estac_prom = df_estac.groupby('mes')['indice'].mean().reset_index()
    df_estac_prom['mes_nombre'] = df_estac_prom['mes'].map(meses_n)
    df_estac_prom['color'] = df_estac_prom['indice'].apply(
        lambda x: '#059669' if x>=1.1 else '#DC2626' if x<0.9 else '#0D9488')

    fig_estac = px.bar(df_estac_prom, x='mes_nombre', y='indice',
                       color='indice', color_continuous_scale='RdYlGn',
                       text=df_estac_prom['indice'].apply(lambda x:f"{x:.2f}x"),
                       template='plotly_white',
                       title="Índice de Estacionalidad (1.0 = promedio mensual)")
    fig_estac.add_hline(y=1.0, line_dash='dash', line_color='#6B7280',
                        annotation_text='Promedio')
    fig_estac.update_layout(height=360, coloraxis_showscale=False)
    st.plotly_chart(fig_estac, width='stretch')

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — AUDITORÍA ANOMALÍAS
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🚨 Detección de Anomalías Transaccionales por IQR")
    st.write("Identifica transacciones que se desvían significativamente del comportamiento esperado.")
    st.markdown("")

    cols_num = [c for c in df.select_dtypes(include=np.number).columns
                if c not in ['id_registro','mes','anio','cumplimiento_meta']]

    if not cols_num:
        st.warning("No hay columnas numéricas disponibles.")
    else:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            col_audit = st.selectbox("Variable de auditoría", cols_num,
                index=cols_num.index('venta_neta') if 'venta_neta' in cols_num else 0)
        with col_s2:
            factor_iqr = st.slider("Sensibilidad IQR", 0.5, 3.0, 1.5, 0.1)

        an_filt = AnalizadorComercial(df)
        outliers_df, lim_inf, lim_sup = an_filt.detectar_outliers_iqr(col_audit, factor_iqr)
        n_out   = len(outliers_df)
        pct_out = n_out/max(len(df),1)*100

        m1,m2,m3 = st.columns(3)
        with m1:
            st.markdown(card("📉","Límite Inferior IQR",f"{lim_inf:,.2f}",
                color="#2563EB",bg="#EFF6FF",border="#93C5FD"), unsafe_allow_html=True)
        with m2:
            st.markdown(card("📈","Límite Superior IQR",f"{lim_sup:,.2f}",
                color="#2563EB",bg="#EFF6FF",border="#93C5FD"), unsafe_allow_html=True)
        with m3:
            o_col="#DC2626" if pct_out>10 else "#D97706" if pct_out>5 else "#059669"
            o_bg ="#FEF2F2" if pct_out>10 else "#FFFBEB" if pct_out>5 else "#F0FDF4"
            o_brd="#FCA5A5" if pct_out>10 else "#FCD34D" if pct_out>5 else "#6EE7B7"
            st.markdown(card("⚠️","Transacciones Anómalas",f"{n_out} ({pct_out:.1f}%)",
                color=o_col,bg=o_bg,border=o_brd), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_box, col_tbl = st.columns(2)
        with col_box:
            fig_box = px.box(df, y=col_audit, color_discrete_sequence=["#0D9488"],
                             title=f"Box Plot — {col_audit}", template="plotly_white")
            fig_box.add_hline(y=lim_inf,line_dash="dash",line_color="#DC2626",
                              annotation_text=f"L.Inf:{lim_inf:,.2f}")
            fig_box.add_hline(y=lim_sup,line_dash="dash",line_color="#DC2626",
                              annotation_text=f"L.Sup:{lim_sup:,.2f}")
            fig_box.update_layout(height=380)
            st.plotly_chart(fig_box, width='stretch')
        with col_tbl:
            st.markdown(f"#### Transacciones Desviadas ({n_out})")
            if n_out > 0:
                cols_show=[c for c in ['fecha','anio','canal','categoria','producto',
                                        col_audit,'region'] if c in df.columns]
                st.dataframe(outliers_df[cols_show].reset_index(drop=True),
                             width='stretch', height=380)
            else:
                st.success("✅ No se detectaron anomalías con los parámetros actuales.")

        st.markdown("#### Dispersión de Transacciones")
        df_plot = df.copy().reset_index(drop=True)
        df_plot['_tipo'] = df_plot[col_audit].apply(
            lambda x: '🔴 Anomalía' if (x<lim_inf or x>lim_sup) else '🟢 Normal')
        fig_scat = px.scatter(df_plot, x=df_plot.index, y=col_audit, color='_tipo',
                              color_discrete_map={'🔴 Anomalía':'#DC2626','🟢 Normal':'#0D9488'},
                              template='plotly_white')
        fig_scat.add_hline(y=lim_inf,line_dash="dot",line_color="#F59E0B")
        fig_scat.add_hline(y=lim_sup,line_dash="dot",line_color="#F59E0B")
        fig_scat.update_layout(height=370)
        st.plotly_chart(fig_scat, width='stretch')

# ══════════════════════════════════════════════════════════════════════
# TAB 5 — MODELO PREDICTIVO
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🤖 Modelo Predictivo de Cumplimiento de Meta Comercial")
    st.write("Clasifica transacciones según probabilidad de alcanzar la meta.")
    st.markdown("")

    cols_bin = [c for c in df.columns if df[c].nunique()==2 and df[c].dropna().isin([0,1]).all()]
    if not cols_bin:
        st.warning("⚠️ No se detectó variable binaria (0/1).")
    else:
        col_p0,col_p1,col_p2 = st.columns(3)
        with col_p0:
            target = st.selectbox("Variable objetivo", cols_bin,
                index=cols_bin.index('cumplimiento_meta') if 'cumplimiento_meta' in cols_bin else 0)
        with col_p1:
            test_size = st.slider("Datos de prueba (%)",10,40,30)/100
        with col_p2:
            n_est = st.slider("Árboles Random Forest",10,200,100,step=10)

        an_mod = AnalizadorComercial(df)
        X, y   = an_mod.preparar_modelo(target)

        if len(y.unique())<2:
            st.error("La variable objetivo tiene solo una clase.")
        else:
            X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=test_size,random_state=42,stratify=y)
            lr = LogisticRegression(max_iter=1000,random_state=42).fit(X_tr,y_tr)
            rf = RandomForestClassifier(n_estimators=n_est,random_state=42).fit(X_tr,y_tr)
            acc_lr = accuracy_score(y_te,lr.predict(X_te))
            acc_rf = accuracy_score(y_te,rf.predict(X_te))
            ganador = "Random Forest" if acc_rf>=acc_lr else "Regresión Logística"
            acc_win = max(acc_rf,acc_lr)
            diff    = abs(acc_rf-acc_lr)

            st.markdown("#### Precisión Comparativa de Modelos")
            def colores(acc):
                if acc>=0.75: return "#059669","#F0FDF4","#6EE7B7"
                if acc>=0.60: return "#D97706","#FFFBEB","#FCD34D"
                return "#DC2626","#FEF2F2","#FCA5A5"

            c1,c2,c3 = st.columns(3)
            with c1:
                col,bg,brd=colores(acc_lr)
                st.markdown(card("📊","Regresión Logística",f"{acc_lr:.1%}","Accuracy en test set",
                    col,bg,brd), unsafe_allow_html=True)
            with c2:
                col,bg,brd=colores(acc_rf)
                st.markdown(card("🌲","Random Forest",f"{acc_rf:.1%}","Accuracy en test set",
                    col,bg,brd), unsafe_allow_html=True)
            with c3:
                st.markdown(badge_ganador(ganador,acc_win,diff), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_imp,col_comp = st.columns(2)
            with col_imp:
                st.markdown("#### Variables con Mayor Influencia")
                imp_df = pd.DataFrame({'Variable':X.columns,'Importancia':rf.feature_importances_})\
                           .sort_values('Importancia',ascending=True)
                fig_imp = px.bar(imp_df,x='Importancia',y='Variable',orientation='h',
                                 color='Importancia',color_continuous_scale='Blues',
                                 template='plotly_white')
                fig_imp.update_layout(height=380,showlegend=False,coloraxis_showscale=False)
                st.plotly_chart(fig_imp, width='stretch')
            with col_comp:
                st.markdown("#### Comparación Visual")
                fig_comp = go.Figure([
                    go.Bar(x=['Reg. Logística'],y=[acc_lr],marker_color='#0D9488',
                           text=[f"{acc_lr:.1%}"],textposition='outside'),
                    go.Bar(x=['Random Forest'],y=[acc_rf],marker_color='#1E3A8A',
                           text=[f"{acc_rf:.1%}"],textposition='outside')])
                fig_comp.update_layout(showlegend=False,
                    yaxis=dict(tickformat='.0%',range=[0,1]),
                    height=380,template='plotly_white')
                st.plotly_chart(fig_comp, width='stretch')

            st.markdown(f"#### Reporte Detallado — {ganador}")
            mejor = rf if acc_rf>=acc_lr else lr
            rep = classification_report(y_te,mejor.predict(X_te),
                target_names=['No Cumple (0)','Cumple Meta (1)'],output_dict=True)
            df_rep = pd.DataFrame(rep).transpose().round(3)
            def color_score(val):
                try:
                    v=float(val)
                    if v>=0.75: return 'background-color:#D1FAE5;color:#065F46'
                    if v>=0.50: return 'background-color:#FEF3C7;color:#92400E'
                    return 'background-color:#FEE2E2;color:#991B1B'
                except: return ''
            st.dataframe(df_rep.style.map(
                color_score,subset=['precision','recall','f1-score']),
                width='stretch')
