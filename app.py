import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# Configuração da página
st.set_page_config(
    page_title="Passos Mágicos - Prevenção de Defasagem",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Sistema Preditivo de Prevenção ao Risco de Defasagem")
st.subheader("Análise Preventiva e Inteligência de Dados - Ciclo 2024")
st.markdown("---")

# Carregamento de dados e modelo
@st.cache_data
def carregar_dados():
    return pd.read_csv('resultados_predicoes_2024.csv')

try:
    df_2024 = carregar_dados()
    modelo = joblib.load('modelo_xgboost_passos_refinado.pkl')
except Exception as e:
    st.error(f"Erro ao carregar os arquivos. Detalhes: {e}")
    st.stop()

# ==============================================================================
# ABAS DA APLICAÇÃO
# ==============================================================================
aba_simulador, aba_visao_geral, aba_busca_individual = st.tabs([
    "🔮 Simulador Preditivo", 
    "📊 Visão Geral da Instituição", 
    "🔍 Fila de Atenção"
])

# ------------------------------------------------------------------------------
# ABA 1: SIMULADOR PREDITIVO (14 VARIÁVEIS EXATAS DO MODELO)
# ------------------------------------------------------------------------------
with aba_simulador:
    st.markdown("### 🔮 Simulador de Risco Preventivo")
    st.markdown("Insira manualmente os 14 indicadores puros exigidos pelo modelo atualizado para realizar a análise.")
    
    with st.form("formulario_simulacao"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("**Desempenho e Nível**")
            ida = st.slider("IDA (Desempenho Acadêmico):", 0.0, 10.0, 7.0, step=0.1)
            ian = st.slider("IAN (Adequação de Nível):", 0.0, 10.0, 7.0, step=0.1)
            matematica = st.slider("Nota de Matemática:", 0.0, 10.0, 7.0, step=0.1)
            portugues = st.slider("Nota de Português:", 0.0, 10.0, 7.0, step=0.1)
            ingles = st.slider("Nota de Inglês:", 0.0, 10.0, 7.0, step=0.1)
            
        with c2:
            st.markdown("**Psicossocial e Engajamento**")
            ieg = st.slider("IEG (Engajamento):", 0.0, 10.0, 7.5, step=0.1)
            ips = st.slider("IPS (Psicossocial):", 0.0, 10.0, 7.2, step=0.1)
            ipv = st.slider("IPV (Ponto de Virada):", 0.0, 10.0, 6.8, step=0.1)
            ipp = st.slider("IPP (Psicopedagógico):", 0.0, 10.0, 7.0, step=0.1)
            iaa = st.slider("IAA (Autoavaliação):", 0.0, 10.0, 7.0, step=0.1)
            
        with c3:
            st.markdown("**Componentes INDE e Histórico**")
            cg = st.slider("Cg (Componente Geral):", 0.0, 10.0, 7.0, step=0.1)
            ct = st.slider("Ct (Componente Técnico):", 0.0, 10.0, 7.0, step=0.1)
            cf = st.slider("Cf (Componente Formação - Base 100):", 0.0, 100.0, 70.0, step=1.0)
            anos_passos = st.number_input("Anos de Vínculo (Passos Mágicos):", min_value=0, max_value=15, value=2)
            
        botao_simular = st.form_submit_button("Gerar Análise Preditiva")

    if botao_simular:
        # 1. Lista de colunas EXATAMENTE na ordem que o XGBoost apontou no erro
        colunas_treino = [
            'Cg', 'Cf', 'Ct', 'IAN', 'IAA', 'IEG', 'IPS', 'IDA', 'IPP', 'IPV', 
            'Matematica', 'Portugues', 'Ingles', 'Anos_de_Passos_Magicos'
        ]

        # 2. Dicionário coletando os inputs diretos dos sliders
        input_dict = {
            'Cg': cg,
            'Cf': cf,
            'Ct': ct,
            'IAN': ian,
            'IAA': iaa,
            'IEG': ieg,
            'IPS': ips,
            'IDA': ida,
            'IPP': ipp,
            'IPV': ipv,
            'Matematica': matematica,
            'Portugues': portugues,
            'Ingles': ingles,
            'Anos_de_Passos_Magicos': anos_passos
        }

        # 3. Criando o DataFrame final respeitando a ordem
        df_simulacao = pd.DataFrame([input_dict])[colunas_treino]
        
        # 4. Execução do modelo
        try:
            probabilidade = modelo.predict_proba(df_simulacao)[0][1] * 100
            
            st.markdown("---")
            st.markdown("### 📋 Diagnóstico da Simulação")
            
            col_metrica, col_texto = st.columns([1, 2])
            
            with col_metrica:
                if probabilidade >= 50.0:
                    st.error(f"Risco: {probabilidade:.1f}%")
                    st.markdown("🔴 **ALERTA DE RISCO ATIVADO**")
                else:
                    st.success(f"Risco: {probabilidade:.1f}%")
                    st.markdown("🔵 **STATUS: REGULAR**")
                    
            with col_texto:
                st.markdown("**Direcionamento Pedagógico:**")
                if probabilidade >= 50.0:
                    st.write("O perfil analisado indica um acúmulo de variáveis de alerta nos indicadores de base. É recomendada a inclusão imediata na fila de acompanhamento psicopedagógico ativo.")
                else:
                    st.write("Os indicadores comportamentais e acadêmicos estão saudáveis de acordo com a maturação e histórico. Manter a rotina de avaliação e acompanhamento padrão.")
                    
        except Exception as sim_error:
            st.error(f"Erro de processamento da matriz de decisão. Detalhes: {sim_error}")

# ------------------------------------------------------------------------------
# ABA 2: VISÃO GERAL INSTITUCIONAL
# ------------------------------------------------------------------------------
with aba_visao_geral:
    st.markdown("### Mapeamento Ativo (Base 2024)")
    
    g1, g2 = st.columns([1, 1.5])
    
    with g1:
        if 'Alerta_Risco' in df_2024.columns:
            df_volumetria = df_2024['Alerta_Risco'].value_counts().reset_index()
            df_volumetria.columns = ['Status', 'Quantidade']
            
            fig_volumetria = px.pie(
                df_volumetria, values='Quantidade', names='Status', hole=0.5,
                color='Status', color_discrete_map={'Sim': '#e74c3c', 'Não': '#34495e'},
                title='Proporção de Alunos'
            )
            fig_volumetria.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_volumetria, use_container_width=True)
        else:
            st.warning("A coluna 'Alerta_Risco' não foi encontrada na base exportada de 2024.")
            
    with g2:
        if 'Fase' in df_2024.columns and 'Alerta_Risco' in df_2024.columns:
            df_cruzamento = df_2024.groupby(['Fase', 'Alerta_Risco']).size().reset_index(name='Quantidade')
            df_totais = df_2024.groupby('Fase').size().reset_index(name='Total_Fase')
            df_cruzamento = df_cruzamento.merge(df_totais, on='Fase')
            df_cruzamento['Porcentagem'] = ((df_cruzamento['Quantidade'] / df_cruzamento['Total_Fase']) * 100).round(1)
            
            fig_fases = px.bar(
                df_cruzamento, x='Fase', y='Quantidade', color='Alerta_Risco',
                color_discrete_map={'Sim': '#e74c3c', 'Não': '#34495e'}, barmode='stack',
                title='Concentração de Risco por Fase Escolar'
            )
            fig_fases.update_traces(hovertemplate="<b>Fase:</b> %{x}<br><b>Qtd:</b> %{y}<br><b>Proporção na fase:</b> %{customdata}%", customdata=df_cruzamento['Porcentagem'])
            st.plotly_chart(fig_fases, use_container_width=True)

# ------------------------------------------------------------------------------
# ABA 3: FILA DE ATENÇÃO
# ------------------------------------------------------------------------------
with aba_busca_individual:
    st.markdown("### Fila de Prioridade Pedagógica")
    
    if 'Alerta_Risco' in df_2024.columns and 'RA' in df_2024.columns:
        filtro_status = st.selectbox("Filtrar Situação:", ["Apenas em Alerta de Risco", "Sem Alerta", "Todos"])
        
        df_filtrado = df_2024.copy()
        if filtro_status == "Apenas em Alerta de Risco":
            df_filtrado = df_filtrado[df_filtrado['Alerta_Risco'] == 'Sim']
        elif filtro_status == "Sem Alerta":
            df_filtrado = df_filtrado[df_filtrado['Alerta_Risco'] == 'Não']
            
        busca_ra = st.text_input("Buscar RA (Ex: RA-102):").strip()
        if busca_ra:
            df_filtrado = df_filtrado[df_filtrado['RA'].str.contains(busca_ra, case=False, na=False)]
            
        # Tenta exibir as colunas padrões de busca (ignora as ausentes caso o CSV tenha sido reduzido)
        colunas_exibicao = [col for col in ['RA', 'Nome Anonimizado', 'Fase', 'Pedra_Atual', 'Probabilidade_Risco (%)', 'Alerta_Risco'] if col in df_filtrado.columns]
        
        st.dataframe(
            df_filtrado[colunas_exibicao],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.dataframe(df_2024)