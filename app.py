
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Passos Mágicos - Prevenção de Defasagem",
    page_icon="🎓",
    layout="wide"
)

# Estilização básica e Títulos
st.title("🎓 Sistema Preditivo de Prevenção ao Risco de Defasagem")
st.subheader("Análise Preventiva e Inteligência de Dados - Ciclo 2024")
st.markdown("---")

# Carregar os dados pontuados e o modelo
@st.cache_data
def carregar_dados():
    df = pd.read_csv('resultados_predicoes_2024.csv')
    return df

try:
    df_2024 = carregar_dados()
    modelo = joblib.load('modelo_xgboost_passos_refinado.pkl')
except Exception as e:
    st.error(f"Erro ao carregar os arquivos necessários. Certifique-se de que 'resultados_predicoes_2024.csv' e 'modelo_xgboost_passos_refinado.pkl' estão na raiz do repositório. Detalhes: {e}")
    st.stop()

# Mapeamento interno de Fases usado no treinamento do modelo
# Certifique-se de que as chaves correspondam exatamente às colunas categóricas tratadas no notebook
fases_mapeamento = {
    "Alfa": 0, "Quartzo": 1, "Ágata": 2, "Ametista": 3, 
    "Topázio": 4, "Opala": 5, "Turmalina": 6
}

# ==============================================================================
# ABAS DE VISUALIZAÇÃO
# ==============================================================================
aba_simulador, aba_visao_geral, aba_busca_individual = st.tabs([
    "🔮 Simulador Preditivo (Momento Futuro)", 
    "📊 Visão Geral da Instituição", 
    "🔍 Busca por Aluno & Fila de Atenção"
])

# ------------------------------------------------------------------------------
# ABA 1: SIMULADOR PREDITIVO
# ------------------------------------------------------------------------------
with aba_simulador:
    st.markdown("### 🔮 Simulador de Risco Preventivo")
    st.markdown("Insira os indicadores comportamentais e acadêmicos abaixo para prever em tempo real a probabilidade de risco de defasagem.")
    
    # Criando formulário estruturado para a entrada de dados do modelo
    with st.form("formulario_simulacao"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            idade_atual = st.number_input("Idade Atual do Aluno:", min_value=5, max_value=25, value=12)
            anos_passos = st.number_input("Anos de Vínculo na Passos Mágicos:", min_value=0, max_value=15, value=2)
            fase_selecionada = st.selectbox("Fase Atual do Aluno:", list(fases_mapeamento.keys()))
            
        with c2:
            nota_ida = st.slider("Nota IDA (Indicador de Desempenho Acadêmico):", 0.0, 10.0, 7.0, step=0.1)
            nota_ieg = st.slider("Nota IEG (Indicador de Engajamento):", 0.0, 10.0, 7.5, step=0.1)
            nota_ips = st.slider("Nota IPS (Indicador Psicossocial):", 0.0, 10.0, 7.2, step=0.1)
            
        with c3:
            nota_ipv = st.slider("Nota IPV (Indicador de Ponto de Virada):", 0.0, 10.0, 6.8, step=0.1)
            nota_ian = st.slider("Nota IAN (Indicador de Adequação de Nível):", 0.0, 10.0, 7.0, step=0.1)
            pedra_atual = st.selectbox("Pedra Conceito Atual:", ["Quartzo", "Ágata", "Ametista", "Topázio"])
            
        botao_simular = st.form_submit_button("Gerar Análise Preditiva")

    if botao_simular:
        # Montar o vetor de entrada com base nas features exatas que seu modelo XGBoost espera
        # NOTA: Ajuste a ordem e os nomes das chaves conforme as colunas do X_train do seu notebook
        dados_input = pd.DataFrame([{
            'Idade_Atual': idade_atual,
            'Anos_de_Passos_Magicos': anos_passos,
            'Fase': fases_mapeamento[fase_selecionada],
            'IDA': nota_ida,
            'IEG': nota_ieg,
            'IPS': nota_ips,
            'IPV': nota_ipv,
            'IAN': nota_ian
        }])
        
        try:
            # Predição direta usando o pipeline/modelo do XGBoost
            probabilidade = modelo.predict_proba(dados_input)[0][1] * 100
            
            # Exibição do diagnóstico baseado na calibração protetora do seu modelo
            st.markdown("---")
            st.markdown("### 📋 Diagnóstico da Simulação")
            
            col_metrica, col_texto = st.columns([1, 2])
            
            with col_metrica:
                if probabilidade >= 50.0: # Threshold padrão ou calibrado
                    st.error(f"Risco: {probabilidade:.1f}%")
                    st.markdown("🔴 **ALERTA DE RISCO ATIVADO**")
                else:
                    st.success(f"Risco: {probabilidade:.1f}%")
                    st.markdown("🔵 **STATUS: MONITORAMENTO REGULAR**")
                    
            with col_texto:
                st.markdown("**Direcionamento Pedagógico:**")
                if probabilidade >= 50.0:
                    st.write("Este perfil simulado apresenta instabilidade nos indicadores críticos de evolução. Recomenda-se a inclusão imediata na fila prioritária de acompanhamento psicopedagógico antes da consolidação do ciclo.")
                else:
                    st.write("O aluno demonstra métricas de maturação saudáveis e está respondendo bem ao tempo de instituição. Manter plano de acompanhamento padrão.")
        except Exception as sim_error:
            st.error(f"Erro ao processar as variáveis na árvore de decisão do modelo. Garanta que a ordem das colunas no dataframe de simulação corresponda perfeitamente ao formato de treino. Detalhes: {sim_error}")

# ------------------------------------------------------------------------------
# ABA 2: VISÃO GERAL INSTITUCIONAL
# ------------------------------------------------------------------------------
with aba_visao_geral:
    st.markdown("### Distribuição Macroscópica do Risco (Base Ativa 2024)")
    
    g1, g2 = st.columns([1, 1.5])
    
    with g1:
        df_volumetria = df_2024['Alerta_Risco'].value_counts().reset_index()
        df_volumetria.columns = ['Status de Risco', 'Quantidade']
        
        fig_volumetria = px.pie(
            df_volumetria, values='Quantidade', names='Status de Risco', hole=0.5,
            color='Status de Risco', color_discrete_map={'Sim': '#e74c3c', 'Não': '#34495e'},
            title='Proporção de Alunos Monitorados'
        )
        fig_volumetria.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_volumetria, use_container_width=True)
        
    with g2:
        df_cruzamento = df_2024.groupby(['Fase', 'Alerta_Risco']).size().reset_index(name='Quantidade')
        df_totais_fase = df_2024.groupby('Fase').size().reset_index(name='Total_Fase')
        df_cruzamento = df_cruzamento.merge(df_totais_fase, on='Fase')
        df_cruzamento['Porcentagem_Na_Fase'] = ((df_cruzamento['Quantidade'] / df_cruzamento['Total_Fase']) * 100).round(1)
        
        fig_fases = px.bar(
            df_cruzamento, x='Fase', y='Quantidade', color='Alerta_Risco',
            color_discrete_map={'Sim': '#e74c3c', 'Não': '#34495e'}, barmode='stack',
            title='Severidade e Concentração de Risco por Fase Escolar',
            labels={'Fase': 'Fase do Aluno', 'Quantidade': 'Número de Alunos'}
        )
        fig_fases.update_traces(
            hovertemplate="<b>Fase:</b> %{x}<br><b>Quantidade:</b> %{y} alunos<br><b>Representa:</b> %{customdata}% da fase",
            customdata=df_cruzamento['Porcentagem_Na_Fase']
        )
        st.plotly_chart(fig_fases, use_container_width=True)

# ------------------------------------------------------------------------------
# ABA 3: FILA DE ATENÇÃO INDIVIDUAL
# ------------------------------------------------------------------------------
with aba_busca_individual:
    st.markdown("### Lista de Prioridade Pedagógica")
    st.markdown("Utilize a busca ou ordenação da tabela para encontrar o histórico pontuado dos alunos vigentes.")
    
    filtro_status = st.selectbox("Filtrar Tabela por Situação:", ["Todos", "Apenas em Alerta de Risco", "Sem Alerta"])
    
    df_filtrado = df_2024.copy()
    if filtro_status == "Apenas em Alerta de Risco":
        df_filtrado = df_filtrado[df_filtrado['Alerta_Risco'] == 'Sim']
    elif filtro_status == "Sem Alerta":
        df_filtrado = df_filtrado[df_filtrado['Alerta_Risco'] == 'Não']
        
    busca_ra = st.text_input("Digite o RA do Aluno (Ex: RA-102):").strip()
    if busca_ra:
        df_filtrado = df_filtrado[df_filtrado['RA'].str.contains(busca_ra, case=False, na=False)]
        
    st.dataframe(
        df_filtrado[['RA', 'Nome Anonimizado', 'Fase', 'Pedra_Atual', 'Probabilidade_Risco (%)', 'Alerta_Risco']],
        use_container_width=True,
        hide_index=True
    )

# Rodapé institucional
st.markdown("---")
st.caption("Solução desenvolvida para o Datathon Passos Mágicos - Pós Tech Data Analytics. O modelo prioriza a sensibilidade (Recall de 72%) para mitigar falsos negativos.")
