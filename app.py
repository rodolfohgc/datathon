import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(page_title="Passos Mágicos - Radar Preventivo", page_icon="🎓", layout="wide")

st.title("🎓 Radar Psicossocial e Comportamental")
st.subheader("Sistema de Alerta Precoce de Vulnerabilidade - Ciclo 2024")
st.markdown("---")

@st.cache_data
def carregar_dados():
    return pd.read_csv('resultados_predicoes_2024.csv')

try:
    df_2024 = carregar_dados()
    modelo = joblib.load('modelo_xgboost_passos_refinado.pkl')
except Exception as e:
    st.error(f"Erro ao carregar os arquivos. Detalhes: {e}")
    st.stop()

aba_simulador, aba_busca_individual = st.tabs(["🔮 Simulador Preventivo", "🔍 Base de Alunos (2024)"])

# ------------------------------------------------------------------------------
# ABA 1: SIMULADOR PREDITIVO (FOCO COMPORTAMENTAL)
# ------------------------------------------------------------------------------
with aba_simulador:
    st.markdown("### 🔮 Análise Individual de Risco")
    st.markdown("O modelo preditivo analisa fatores sociodemográficos e psicossociais para antecipar riscos de defasagem, sem depender de notas acadêmicas.")
    
    with st.form("formulario_simulacao"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            idade_atual = st.number_input("Idade Atual:", min_value=5, max_value=25, value=12)
            anos_passos = st.number_input("Anos de Passos Mágicos:", min_value=0, max_value=15, value=2)
            
        with c2:
            ieg = st.slider("IEG (Engajamento):", 0.0, 10.0, 7.5, step=0.1)
            ips = st.slider("IPS (Psicossocial):", 0.0, 10.0, 7.0, step=0.1)
            
        with c3:
            ipp = st.slider("IPP (Psicopedagógico):", 0.0, 10.0, 7.0, step=0.1)
            ipv = st.slider("IPV (Ponto de Virada):", 0.0, 10.0, 6.5, step=0.1)
            
        botao_simular = st.form_submit_button("Gerar Diagnóstico Preventivo")

    if botao_simular:
        # Ordem EXATA das colunas que restaram no seu X_treino
        colunas_treino = ['IEG', 'IPS', 'IPP', 'IPV', 'Anos_de_Passos_Magicos', 'Idade_Atual']
        
        input_dict = {
            'IEG': ieg,
            'IPS': ips,
            'IPP': ipp,
            'IPV': ipv,
            'Anos_de_Passos_Magicos': anos_passos,
            'Idade_Atual': idade_atual
        }

        df_simulacao = pd.DataFrame([input_dict])[colunas_treino]
        
        try:
            probabilidade = modelo.predict_proba(df_simulacao)[0][1] * 100
            
            st.markdown("---")
            st.markdown("### 📋 Diagnóstico da Simulação")
            
            col_metrica, col_texto = st.columns([1, 2])
            
            # Limiar de negócio ajustado para 35% (alta captura preventiva)
            with col_metrica:
                if probabilidade >= 35.0:
                    st.error(f"Risco Projetado: {probabilidade:.1f}%")
                    st.markdown("🔴 **ALERTA DE VULNERABILIDADE**")
                else:
                    st.success(f"Risco Projetado: {probabilidade:.1f}%")
                    st.markdown("🔵 **PERFIL ESTÁVEL**")
                    
            with col_texto:
                if probabilidade >= 35.0:
                    st.write("**Direcionamento:** O cruzamento de idade, tempo de ONG e indicadores psicossociais aponta vulnerabilidade futura. Priorizar escuta ativa e acolhimento familiar antes que reflita no desempenho acadêmico.")
                else:
                    st.write("**Direcionamento:** O aluno demonstra maturidade psicológica e engajamento compatíveis com sua fase. Manter acompanhamento padrão.")
                    
        except Exception as sim_error:
            st.error(f"Erro de processamento. Detalhes: {sim_error}")

# ------------------------------------------------------------------------------
# ABA 2: FILA DE ATENÇÃO
# ------------------------------------------------------------------------------
with aba_busca_individual:
    st.markdown("### Base de Dados Institucional")
    
    busca = st.text_input("Buscar aluno por RA ou Nome:").strip()
    
    df_exibicao = df_2024.copy()
    
    if busca:
        mask_ra = df_exibicao['RA'].astype(str).str.contains(busca, case=False, na=False) if 'RA' in df_exibicao.columns else False
        mask_nome = df_exibicao['Nome Anonimizado'].astype(str).str.contains(busca, case=False, na=False) if 'Nome Anonimizado' in df_exibicao.columns else False
        df_exibicao = df_exibicao[mask_ra | mask_nome]
        
    st.dataframe(df_exibicao, use_container_width=True, hide_index=True)