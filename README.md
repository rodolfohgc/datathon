# Passos Mágicos: Radar Psicossocial e Comportamental 🎓

Este repositório contém a modelagem de dados e a aplicação interativa desenvolvidas para o Datathon da ONG Passos Mágicos. O projeto constitui a atividade final do curso de **Pós-Tech em Data Analytics da FIAP**.

---

## Sobre o Projeto

A solução desenvolvida atua como um sistema de alerta precoce de vulnerabilidade educacional. O objetivo principal é antecipar o risco de um aluno apresentar baixo desempenho geral (INDE < 6.0) no ciclo vigente.

Para evitar vazamento de dados (Target Leakage) e diagnósticos tardios, o modelo abandonou o uso de notas acadêmicas puras. A predição é construída exclusivamente sobre **indicadores psicossociais, sociodemográficos e de engajamento** (como IEG, IPS, IPP e IPV). Essa abordagem garante à equipe psicopedagógica uma ferramenta de prevenção real, permitindo intervenções e acolhimento antes que o aluno entre em um quadro severo de defasagem.

O motor preditivo utiliza o algoritmo **Random Forest**, escolhido por meio de validação cruzada (*Cross-Validation*) por apresentar o melhor equilíbrio técnico entre a captura efetiva de alunos vulneráveis (*Recall*) e a consistência dos alertas gerados (*Precision*).

---

## Acessos e Plataformas

* **Simulador Preditivo (Aplicação Streamlit):** [Acessar Modelo Preditivo](https://datathon-pos-tech.streamlit.app/)
* **Relatório Gerencial (Impact Lens):** [Acessar Painel Analítico](https://passos-magicos-impact-lens.lovable.app)

---

## Estrutura do Repositório

* `anlise.ipynb`: Notebook completo contendo a limpeza de dados e análise que deu origem ao relatório gerencial.
* `machine-learning.ipynb`: Notebook completo contendo a limpeza de dados, correção de vazamentos, aplicação de técnicas de balanceamento (SMOTE), *benchmark* de algoritmos e exportação do modelo campeão.
* `app.py`: Código-fonte da aplicação web, estruturada com sistema de simulação de risco, consulta à base institucional e detalhamento de *trade-offs*.
* `modelo_vencedor_datathon.pkl`: Algoritmo preditivo treinado e otimizado para deploy.
* `resultados_predicoes_2024.csv`: Base de dados consolidada e tratada referente ao ciclo de 2024 para alimentação do painel de busca.

---

## Tecnologias Utilizadas

* **Linguagem e Dados:** Python, Pandas, NumPy
* **Machine Learning:** Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Visualização e Web:** Streamlit, Plotly

---

## Integrantes do Grupo

* Fábio Aluizio Paulino
* Filipe Hideki Kwang 
* Rodolfo Henrique Gonçalo Conceição