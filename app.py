import streamlit as st
import pandas as pd
import joblib

# ==============================================================================
# CONFIGURAÇÃO INICIAL
# ==============================================================================
st.set_page_config(
    page_title="Passos Mágicos - Radar Preventivo",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Radar Psicossocial e Comportamental")
st.subheader("Sistema de Alerta Precoce de Vulnerabilidade - Ciclo 2024")
st.markdown("---")

# ==============================================================================
# CARREGAMENTO DOS ARQUIVOS
# ==============================================================================
@st.cache_data
def carregar_dados():
    return pd.read_csv('resultados_predicoes_2024.csv')

try:
    df_2024 = carregar_dados()
    modelo = joblib.load('modelo_vencedor_datathon.pkl')
except Exception as e:
    st.error(f"Erro ao carregar os arquivos. Certifique-se de que 'resultados_predicoes_2024.csv' e 'modelo_vencedor_datathon.pkl' estão no repositório. Detalhes: {e}")
    st.stop()

aba_simulador, aba_busca = st.tabs(["🔮 Simulador Preventivo", "🔍 Base Institucional (2024)"])

# ==============================================================================
# ABA 1: SIMULADOR PREDITIVO
# ==============================================================================
with aba_simulador:
    st.markdown("### Análise Individual de Risco")
    st.markdown("Antecipar a probabilidade de um aluno apresentar baixo desempenho (INDE < 6.0) analisando exclusivamente seus indicadores psicossociais e de engajamento.")
    
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
            
        botao_simular = st.form_submit_button("Gerar Diagnóstico")

    if botao_simular:
        # A ordem deve ser estritamente igual à lista "features" do treinamento
        colunas_treino = ['IEG', 'IPS', 'IPP', 'IPV', 'Idade_Atual', 'Anos_de_Passos_Magicos']
        
        input_dict = {
            'IEG': ieg,
            'IPS': ips,
            'IPP': ipp,
            'IPV': ipv,
            'Idade_Atual': idade_atual,
            'Anos_de_Passos_Magicos': anos_passos
        }

        df_simulacao = pd.DataFrame([input_dict])[colunas_treino]
        
        try:
            # Captura a probabilidade da classe 1 (Em Risco)
            probabilidade = modelo.predict_proba(df_simulacao)[0][1] * 100
            
            st.markdown("---")
            st.markdown("### 📋 Diagnóstico da Simulação")
            
            col_metrica, col_texto = st.columns([1, 2])
            
            with col_metrica:
                if probabilidade >= 50.0:
                    st.error(f"Risco Projetado: {probabilidade:.1f}%")
                    st.markdown("🔴 **ALERTA DE VULNERABILIDADE**")
                else:
                    st.success(f"Risco Projetado: {probabilidade:.1f}%")
                    st.markdown("🔵 **PERFIL ESTÁVEL**")
                    
            with col_texto:
                if probabilidade >= 50.0:
                    st.write("**Direcionamento:** O modelo cruzou o perfil comportamental e de engajamento do aluno e detectou um padrão forte de vulnerabilidade. É recomendada a inclusão na fila de acompanhamento psicopedagógico ativo.")
                else:
                    st.write("**Direcionamento:** Os indicadores psicossociais e de maturidade do aluno apresentam estabilidade e resiliência frente aos desafios acadêmicos. Manter rotina padrão.")
                    
        except Exception as sim_error:
            st.error(f"Erro de processamento da simulação. Detalhes: {sim_error}")

# ==============================================================================
# ABA 2: BASE INSTITUCIONAL
# ==============================================================================
with aba_busca:
    st.markdown("### Consulta de Alunos - Ciclo Atual")
    
    busca = st.text_input("Buscar aluno por RA ou Nome:").strip()
    
    df_exibicao = df_2024.copy()
    
    if busca:
        mask_ra = df_exibicao['RA'].astype(str).str.contains(busca, case=False, na=False) if 'RA' in df_exibicao.columns else False
        mask_nome = df_exibicao['Nome Anonimizado'].astype(str).str.contains(busca, case=False, na=False) if 'Nome Anonimizado' in df_exibicao.columns else False
        df_exibicao = df_exibicao[mask_ra | mask_nome]
        
    st.dataframe(df_exibicao, use_container_width=True, hide_index=True)