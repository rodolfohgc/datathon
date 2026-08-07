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
# ABA 1: SIMULADOR PREDITIVO (MODELO PREVENTIVO ATUALIZADO)
# ------------------------------------------------------------------------------
with aba_simulador:
    st.markdown("### 🔮 Simulador de Risco Preventivo")
    st.markdown("Ajuste os indicadores abaixo para prever a probabilidade de risco. O modelo foi ajustado para atuar preventivamente reagindo de forma sensível às oscilações de notas e engajamento.")
    
    with st.form("formulario_simulacao"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            idade_atual = st.number_input("Idade Atual:", min_value=5, max_value=25, value=12)
            anos_passos = st.number_input("Anos de Vínculo (Passos Mágicos):", min_value=0, max_value=15, value=2)
            pedra_atual = st.selectbox("Pedra Conceito:", ["Quartzo", "Ágata", "Ametista", "Topázio"])
            genero = st.selectbox("Gênero:", ["Feminino", "Masculino"])
            
        with c2:
            nota_inde = st.slider("Nota INDE (Geral):", 0.0, 10.0, 7.0, step=0.1)
            nota_ida = st.slider("Nota IDA (Desempenho Acadêmico):", 0.0, 10.0, 7.0, step=0.1)
            nota_ieg = st.slider("Nota IEG (Engajamento):", 0.0, 10.0, 7.5, step=0.1)
            nota_ian = st.slider("Nota IAN (Adequação de Nível):", 0.0, 10.0, 7.0, step=0.1)
            
        with c3:
            nota_ips = st.slider("Nota IPS (Psicossocial):", 0.0, 10.0, 7.2, step=0.1)
            nota_ipv = st.slider("Nota IPV (Ponto de Virada):", 0.0, 10.0, 6.8, step=0.1)
            delta_inde = st.slider("Evolução INDE vs Ano Anterior (Delta):", -10.0, 10.0, 0.0, step=0.1)
            
        botao_simular = st.form_submit_button("Gerar Análise Preditiva")

    if botao_simular:
        # 1. Lista atualizada de colunas (Exatamente as 36 colunas sem a Defasagem)
        colunas_treino = [
            'INDE', 'Cg', 'Cf', 'Ct', 'IAN', 'IAA', 'IEG', 'IPS', 'IDA', 'IPP', 'IPV', 
            'Matematica', 'Portugues', 'Ingles', 'Ano Pesquisa', 'Pedra_Atual_Cod', 
            'Gênero_Masculino', 'Gênero_Menina', 'Gênero_Menino', 
            'Instituição de ensino_Concluiu O 3º Em', 'Instituição de ensino_Escola Jp Ii', 
            'Instituição de ensino_Escola Pública', 'Instituição de ensino_Nenhuma Das Opções Acima', 
            'Instituição de ensino_Privada', 'Instituição de ensino_Privada *Parcerias Com Bolsa 100%', 
            'Instituição de ensino_Privada - Pagamento Por *Empresa Parceira', 
            'Instituição de ensino_Privada - Programa De Apadrinhamento', 
            'Instituição de ensino_Pública', 'Instituição de ensino_Rede Decisão', 
            'Delta_INDE', 'Delta_IDA', 'Delta_IEG', 'Delta_IPS', 'Delta_IPV', 
            'Idade_Atual', 'Anos_de_Passos_Magicos'
        ]

        # 2. Inicialização do dicionário com todas as colunas zeradas
        input_dict = {col: 0 for col in colunas_treino}
        
        # Inserção de valores neutros padrão nas categorias que não estão na interface
        input_dict.update({
            'Ano Pesquisa': 2024,
            'Instituição de ensino_Escola Pública': 1,
            'IPP': 7, 
            'Delta_IDA': 0,
            'Delta_IEG': 0,
            'Delta_IPS': 0,
            'Delta_IPV': 0
        })

        # 3. Mapeamento das entradas da interface
        mapa_pedras = {'Quartzo': 1, 'Ágata': 2, 'Ametista': 3, 'Topázio': 4}
        
        input_dict['Idade_Atual'] = idade_atual
        input_dict['Anos_de_Passos_Magicos'] = anos_passos
        input_dict['Pedra_Atual_Cod'] = mapa_pedras[pedra_atual]
        
        # ESPELHAMENTO DE VARIÁVEIS (Força o modelo a compreender a queda das notas)
        # Sincroniza a nota IDA do slider com as matérias específicas
        input_dict['IDA'] = nota_ida
        input_dict['Matematica'] = nota_ida
        input_dict['Portugues'] = nota_ida
        input_dict['Ingles'] = nota_ida
        
        # Sincroniza a nota IEG do slider com o indicador Ct
        input_dict['IEG'] = nota_ieg
        input_dict['Ct'] = nota_ieg
        
        # Sincroniza a nota INDE com as métricas dependentes
        input_dict['INDE'] = nota_inde
        input_dict['Cg'] = nota_inde
        input_dict['Cf'] = nota_inde * 10 
        input_dict['IAA'] = nota_inde
        
        input_dict['IAN'] = nota_ian
        input_dict['IPS'] = nota_ips
        input_dict['IPV'] = nota_ipv
        input_dict['Delta_INDE'] = delta_inde
        
        if genero == "Feminino":
            input_dict['Gênero_Menina'] = 1
        else:
            input_dict['Gênero_Masculino'] = 1

        # 4. Criando o DataFrame final respeitando estritamente a ordem da modelagem
        df_simulacao = pd.DataFrame([input_dict])[colunas_treino]
        
        # 5. Execução do modelo
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
            st.error(f"Erro de processamento da matriz de decisão. Garanta que todas as colunas correspondem ao modelo atualizado (.pkl). Detalhes: {sim_error}")

# ------------------------------------------------------------------------------
# ABA 2: VISÃO GERAL INSTITUCIONAL
# ------------------------------------------------------------------------------
with aba_visao_geral:
    st.markdown("### Mapeamento Ativo (Base 2024)")
    
    g1, g2 = st.columns([1, 1.5])
    
    with g1:
        df_volumetria = df_2024['Alerta_Risco'].value_counts().reset_index()
        df_volumetria.columns = ['Status', 'Quantidade']
        
        fig_volumetria = px.pie(
            df_volumetria, values='Quantidade', names='Status', hole=0.5,
            color='Status', color_discrete_map={'Sim': '#e74c3c', 'Não': '#34495e'},
            title='Proporção de Alunos'
        )
        fig_volumetria.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_volumetria, use_container_width=True)
        
    with g2:
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
    
    filtro_status = st.selectbox("Filtrar Situação:", ["Apenas em Alerta de Risco", "Sem Alerta", "Todos"])
    
    df_filtrado = df_2024.copy()
    if filtro_status == "Apenas em Alerta de Risco":
        df_filtrado = df_filtrado[df_filtrado['Alerta_Risco'] == 'Sim']
    elif filtro_status == "Sem Alerta":
        df_filtrado = df_filtrado[df_filtrado['Alerta_Risco'] == 'Não']
        
    busca_ra = st.text_input("Buscar RA (Ex: RA-102):").strip()
    if busca_ra:
        df_filtrado = df_filtrado[df_filtrado['RA'].str.contains(busca_ra, case=False, na=False)]
        
    st.dataframe(
        df_filtrado[['RA', 'Nome Anonimizado', 'Fase', 'Pedra_Atual', 'Probabilidade_Risco (%)', 'Alerta_Risco']],
        use_container_width=True,
        hide_index=True
    )