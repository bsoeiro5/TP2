import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

st.set_page_config(layout="wide")

st.title("Student Performance Dashboard")

# Carregar dados
df = pd.read_csv('student-data.csv')  
df['passed'] = df['passed'].map({'yes': 1, 'no': 0})

# Criar versão numérica do dataframe para análise
ndf = pd.read_csv('student-data.csv')
ndf['passed'] = ndf['passed'].map({'yes': 1, 'no': 0})
ndf['school'] = ndf['school'].map({'GP': 1, 'MS': 0})
ndf['sex'] = ndf['sex'].map({'F': 1, 'M': 0})
ndf['address'] = ndf['address'].map({'R': 1, 'U': 0})
ndf['famsize'] = ndf['famsize'].map({'GT3': 1, 'LE3': 0})
ndf['Pstatus'] = ndf['Pstatus'].map({'T': 1, 'A': 0})
ndf['schoolsup'] = ndf['schoolsup'].map({'yes': 1, 'no': 0})
ndf['famsup'] = ndf['famsup'].map({'yes': 1, 'no': 0})
ndf['paid'] = ndf['paid'].map({'yes': 1, 'no': 0})
ndf['activities'] = ndf['activities'].map({'yes': 1, 'no': 0})
ndf['nursery'] = ndf['nursery'].map({'yes': 1, 'no': 0})
ndf['higher'] = ndf['higher'].map({'yes': 1, 'no': 0})
ndf['internet'] = ndf['internet'].map({'yes': 1, 'no': 0})
ndf['romantic'] = ndf['romantic'].map({'yes': 1, 'no': 0})
ndf = pd.get_dummies(ndf, columns=['Mjob', 'Fjob'], drop_first=True)
ndf = pd.get_dummies(ndf, columns=['reason'], drop_first=True)
ndf = pd.get_dummies(ndf, columns=['guardian'], drop_first=True)
ndf = ndf.replace({True: 1, False: 0})

# Treinar modelos
X = ndf.drop(['passed'], axis=1)
y = ndf['passed']

# Definir modelos
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'MLP Classifier': MLPClassifier(random_state=42, max_iter=500, early_stopping=True)
}

# Treinar todos os modelos
for name, model in models.items():
    model.fit(X, y)

# Navegação
tabs = st.tabs(["Dashboard", "Novo Aluno"])

with tabs[0]:
    with st.expander("Trabalho realizado por:"):
        st.markdown("[Àlvaro Castro (FCUP_IACD:202403514)](https://www.linkedin.com/in/%C3%A1lvaro-vieira-de-castro-1571b4332/)")
        st.write("[Bernardo Soeiro (FCUP_IACD:202403514)](https://www.linkedin.com/in/bernardo-soeiro-46347831a/)")
        st.write("[Francisco Machado (FCUP_IACD:202403514)](https://www.linkedin.com/in/franciscovilasboasmachado/)")

    st.markdown("---")

    st.title("0. Introdução")

    st.markdown("""
    <div style='text-align: justify'>
    O insucesso escolar continua a ser uma das principais preocupações no contexto educativo atual, com impactos significativos na trajetória académica e pessoal dos alunos. A deteção precoce de estudantes em risco de reprovação é essencial para permitir intervenções pedagógicas eficazes.

    Neste projeto, propomos o desenvolvimento de um <strong>Student Intervention System</strong>, baseado em técnicas de <em>Supervised Machine Learning</em>, com o objetivo de prever se um aluno irá ou não ser aprovado no exame final, utilizando dados reais recolhidos de duas escolas de ensino secundário em Portugal.

    Para atingir esse objetivo, baseamo-nos em várias etapas fundamentais da ciência de dados, nomeadamente: <strong>análise exploratória dos dados</strong>, <strong>tratamento e transformação das variáveis</strong>, <strong>treino de modelos de classificação</strong> e <strong>avaliação rigorosa dos seus desempenhos</strong>. Cada etapa é essencial para garantir que o sistema resultante seja não só eficaz, mas também transparente e replicável.
    </div>

    ### Métodos de Aprendizagem Supervisionada

    -  K-Nearest Neighbors (KNN)  
    -  Decision Tree  
    -  Random Forest  
    -  Logistic Regression  
    -  Gradient Boosting  
    -  Support Vector Machine (SVM)  
    -  Rede Neural (Neural Network)
    """, unsafe_allow_html=True)

    st.subheader("0.1. Sobre o estudo")
    st.markdown("""
    <div style='text-align: justify'>
    Tipicamente, como em muitos projetos de <strong>Data Science</strong>, todo o código implementado foi desenvolvido em <strong>Jupyter Notebook</strong>.

    <br>

    Alguns trechos de código ser-lhe-ão apresentados ao longo desta documentação. Para além do foco primário do projeto, desejávamos que qualquer dataset pudesse ser convertido num <strong>DataFrame</strong> e que, posteriormente, pudesse ser <strong>processado e polido</strong> segundo os métodos que desenhamos, a fim de alimentar um <strong>algoritmo de machine learning</strong>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.title("1. Análise Exploratória dos Dados")
    st.markdown("""
    <div style='text-align: justify'>
    Talvez o processo mais importante de todo o estudo. Um dos grandes desafios que qualquer <strong>Data Scientist</strong> enfrenta é a forma crua como os dados lhe são apresentados. Analogamente, podemos pensar que o dataset inicial é como um minério de ouro recém-extraído: contém impurezas, fragmentos de outras rochas, e precisa ser cuidadosamente limpo e polido. O nosso trabalho, à semelhança do mineiro, é extrair valor — neste caso, <strong>informação relevante</strong> para alimentar algoritmos de <em>machine learning</em>.

    <br>

    Neste projeto, utilizámos o ficheiro <code>student-data.csv</code>, que contém dados reais recolhidos de duas escolas secundárias em Portugal. Cada linha do <code>DataFrame</code> representa um estudante individual, e cada coluna representa uma característica (atributo) relevante para o seu desempenho escolar — como estatuto familiar, hábitos de estudo, consumo de álcool, tempo de deslocação, apoio extra-curricular, entre outros.

    <br>

    Contam-se <strong>395 estudantes</strong> descritos por <strong>33 atributos</strong>, que combinam variáveis <strong>categóricas</strong> (como género, tipo de escola, curso frequentado) e <strong>numéricas</strong> (como notas anteriores, número de faltas, idade). Este tipo de informação serve de base para a análise exploratória, a identificação de padrões e a criação de modelos preditivos.

    <br>

    Uma boa análise dos dados é determinante para a qualidade dos resultados. Foi-nos incumbido não só de compreender como os dados estão representados, mas também de <strong>relacionar variáveis</strong>, <strong>tratar valores em falta</strong>, e <strong>transformar os atributos</strong> para que estejam prontos a ser utilizados nos algoritmos de aprendizagem supervisionada.
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df)

    st.subheader("Guia de Análise Exploratória de Dados")

    st.markdown("""
    <div style='text-align: justify'>
    Assim, de modo a ultrapassarmos esta fase corretamente, decidimos que a nossa <strong>análise exploratória de dados</strong> (<em>data analysis</em>) se guiaria pelos seguintes aspetos principais:
    </div>

    <ul>
        <li>📊 <strong>Estatísticas descritivas básicas</strong> – para compreender a distribuição geral dos dados.</li>
        <li>🔍 <strong>Análise de variáveis</strong> – para examinar a importância, variabilidade e relação entre os atributos.</li>
        <li>🚨 <strong>Deteção de outliers</strong> – para identificar valores atípicos que possam afetar os modelos preditivos.</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("1.1. Novo Dataset")

    st.markdown("""
    Para que os algoritmos de Machine Learning consigam interpretar corretamente os dados, é necessário que todas as variáveis estejam em formato numérico. 
    Muitos modelos não conseguem lidar diretamente com variáveis categóricas (por exemplo, `sexo`, `escola`, `endereço`), pelo que transformámos essas variáveis 
    em valores binários (0 ou 1), ou aplicámos codificação one-hot em casos com múltiplas categorias.

    Este processo garante a compatibilidade com modelos supervisionados e melhora a performance geral da aprendizagem automática.
    """)

    # Exibir uma amostra do novo dataset
    st.markdown("### 🔎 Pré-visualização do novo dataset transformado:")
    st.dataframe(df.head(), use_container_width=True)
    codigo_transformacao = '''
    df = pd.read_csv('student-data.csv')  
    df['passed'] = df['passed'].map({'yes': 1, 'no': 0})
    df['school'] = df['school'].map({'GP': 1, 'MS': 0})
    df['sex'] = df['sex'].map({'F': 1, 'M': 0})
    df['address'] = df['address'].map({'R': 1, 'U': 0})
    df['famsize'] = df['famsize'].map({'GT3': 1, 'LE3': 0})
    df['Pstatus'] = df['Pstatus'].map({'T': 1, 'A': 0})
    df['schoolsup'] = df['schoolsup'].map({'yes': 1, 'no': 0})
    df['famsup'] = df['famsup'].map({'yes': 1, 'no': 0})
    df['paid'] = df['paid'].map({'yes': 1, 'no': 0})
    df['activities'] = df['activities'].map({'yes': 1, 'no': 0})
    df['nursery'] = df['nursery'].map({'yes': 1, 'no': 0})
    df['higher'] = df['higher'].map({'yes': 1, 'no': 0})
    df['internet'] = df['internet'].map({'yes': 1, 'no': 0})
    df['romantic'] = df['romantic'].map({'yes': 1, 'no': 0})
    df = pd.get_dummies(df, columns=['Mjob', 'Fjob'], drop_first=True)
    df = pd.get_dummies(df, columns=['reason'], drop_first=True)
    df = pd.get_dummies(df, columns=['guardian'], drop_first=True)
    df = df.replace({True: 1, False: 0})
    df
    '''
    st.markdown("### 💻 Código utilizado para transformar o dataset:")
    st.code(codigo_transformacao, language='python')

    st.markdown("---")
    st.subheader("1.2. Matriz de Correlação")

    # Calcular matriz de correlação
    corr_matrix = ndf.corr()

    # Mostrar heatmap com Plotly
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Viridis',
        title='Matriz de Correlação Interativa (Variáveis Numéricas + Alvo)'
    )
    st.plotly_chart(fig)
    st.markdown("---")
    st.subheader("1.3. Estatísticas Descritivas Básicas")

    st.markdown("""
    <div style='text-align: justify'>
    Um dos sistemas mais simples, mas ainda assim dos mais eficazes, é a realização de uma <strong>análise descritiva</strong> de cada atributo. 
    Entende-se por estatísticas descritivas básicas as seguintes medidas, que são aplicáveis apenas a <em>features numéricas</em>:
    </div>

    <ul>
        <li>📌 <strong>Média</strong></li>
        <li>📌 <strong>Mediana</strong></li>
        <li>📌 <strong>Desvio Padrão</strong></li>
        <li>📌 <strong>Assimetria</strong></li>
        <li>📌 <strong>Curtose</strong></li>
    </ul>

    <p>Abaixo encontram-se tabelas que apresentam essas estatísticas para cada atributo, juntamente com uma breve descrição da sua importância.</p>
    """, unsafe_allow_html=True)

    # Selecionar apenas colunas numéricas
    numeric_df = ndf.select_dtypes(include=['number'])

    # Calcular estatísticas
    mean = numeric_df.mean().rename("Média")
    median = numeric_df.median().rename("Mediana")
    std_dev = numeric_df.std().rename("Desvio Padrão")
    skewness = numeric_df.skew().rename("Assimetria")
    kurtosis = numeric_df.kurt().rename("Curtose")

    # Concatenar tudo numa única tabela
    statistics_df = pd.concat([mean, median, std_dev, skewness, kurtosis], axis=1)

    # Mostrar tabela no Streamlit
    st.dataframe(statistics_df.style.format(precision=2), use_container_width=True)

    st.markdown("""
    - **📊 Média:** Representa o valor central dos valores de cada atributo, sendo muito útil para resumir grandes quantidades de dados num único valor.
    - **📈 Mediana:** Útil para entender a tendência central dos dados, especialmente quando há valores extremos (outliers), pois não é afetada por eles como a média.
    - **📉 Desvio Padrão:** Indica o grau de dispersão dos dados. Um valor alto significa que os dados estão espalhados numa ampla gama; um valor baixo indica que estão mais concentrados em torno da média.
    - **📐 Assimetria:** Ajuda a entender a simetria da distribuição. Assimetrias podem indicar presença de outliers ou sugerir a necessidade de transformar os dados.
    - **⛰️ Curtose:** Informa sobre a forma da distribuição — se é mais "pontiaguda" ou achatada — o que pode ser relevante em análises estatísticas e modelagem.
    """)

    st.markdown("---")
    st.subheader("1.4. Ajuste dos Outliers")

    st.markdown("""
    Outliers são valores que se desviam significativamente da maioria dos dados em um conjunto. Eles podem surgir por erros de medição, entrada de dados incorreta ou até por variações naturais. Sua presença pode influenciar negativamente a performance de alguns modelos de Machine Learning, distorcendo métricas e padrões.

    A seguir, aplicamos duas abordagens para identificar possíveis outliers:
    """)

    # --- Subtítulo 1.4.1 ---
    st.subheader("1.4.1. Identificação Visual dos Outliers")

    st.markdown("""
    Nesta abordagem, utilizamos o intervalo interquartil (IQR) para detectar outliers. 
    A técnica compara cada valor com os limites inferiores e superiores baseados nos quartis.

    - **Outliers**: valores fora do intervalo \[Q1 - 1.5×IQR, Q3 + 1.5×IQR\]
    - **Outliers severos**: valores fora do intervalo \[Q1 - 3×IQR, Q3 + 3×IQR\]
    """)

    # Cálculo dos outliers com código formatado
    codigo_outliers_visual = '''
    numeric_df = df.select_dtypes(include=['number'])
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_df = df[numeric_cols]

    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = ((numeric_df < lower_bound) | (numeric_df > upper_bound))
    outliers_count = outliers.sum()
    print("Número de outliers por coluna:\\n", outliers_count)

    lower_bound_severe = Q1 - 3 * IQR
    upper_bound_severe = Q3 + 3 * IQR

    outliers_severe = ((numeric_df < lower_bound_severe) | (numeric_df > upper_bound_severe))
    outliers_severe_count = outliers_severe.sum()
    print("Número de outliers severos por coluna:\\n", outliers_severe_count)

    outliers_count.plot(kind='bar', figsize=(10, 6), title='Número de Outliers por Feature')
    plt.xlabel('Features')
    plt.ylabel('Número de Outliers Severos')
    plt.show()
    '''
    st.code(codigo_outliers_visual, language='python')

    # --- Subtítulo 1.4.2 ---
    st.subheader("1.4.2. Identificação via PCA (Redução de Dimensionalidade)")

    st.markdown("""
    Outra abordagem foi a aplicação de **PCA (Análise de Componentes Principais)** para reduzir os dados para duas dimensões.
    A distância de cada ponto à origem foi usada para identificar valores extremos, ou seja, possíveis outliers.

    Selecionamos os **2% mais distantes** como potenciais outliers.
    """)

    codigo_outliers_pca = '''
    from sklearn.decomposition import PCA
    pca_2d = PCA(n_components=2)

    X_train_processed = preprocessor.fit_transform(X_train)

    X_train_pca_2d = pca_2d.fit_transform(X_train_processed)

    distances_2d = np.sqrt(np.sum(X_train_pca_2d**2, axis=1))
    outlier_threshold_2d = np.percentile(distances_2d, 98)  
    outliers_pca_indices_train = X_train.index[distances_2d > outlier_threshold_2d]

    print(f'Threshold de distância (Percentil 98 no espaço 2D - Treino): {outlier_threshold_2d:.2f}')
    print(f'Número de outliers potenciais identificados no treino: {len(outliers_pca_indices_train)}')
    if not outliers_pca_indices_train.empty:
        print(f'Índices dos outliers potenciais no treino: {outliers_pca_indices_train.tolist()}')
    else:
        print('Nenhum outlier potencial identificado no treino.')

    fig = px.scatter(x=X_train_pca_2d[:, 0], y=X_train_pca_2d[:, 1], color=y_train.astype(str),
                    labels={'color': 'Passed'}, title='PCA (2 Componentes) com Outliers Potenciais - Treino',
                    hover_name=X_train.index)
    fig.add_trace(go.Scatter(x=X_train_pca_2d[distances_2d > outlier_threshold_2d, 0],
                            y=X_train_pca_2d[distances_2d > outlier_threshold_2d, 1],
                            mode='markers', marker=dict(color='red', size=10, symbol='x'),
                            name='Outlier Potencial', hoverinfo='text',
                            hovertext=[f'Index: {i}' for i in outliers_pca_indices_train]))
    fig.update_layout(xaxis_title='Componente Principal 1', yaxis_title='Componente Principal 2')
    fig.show()
    '''
    st.code(codigo_outliers_pca, language='python')

    # --- Decisão final ---
    st.subheader("📌 Decisão Final")

    st.markdown("""
    Após identificar possíveis outliers pelas duas abordagens apresentadas, decidimos **manter os outliers para a modelagem inicial**.

    Isso nos permite observar se esses valores realmente impactam negativamente os modelos. Se necessário, na fase de otimização e validação dos modelos, retornaremos ao tratamento destes valores.
    """)

    st.markdown("---")

    st.title("2. Algoritmos de Supervised Machine Learning")

    st.subheader("O que é Aprendizagem Supervisionada?")

    st.markdown("""
    Os algoritmos de **Aprendizagem Supervisionada** são técnicas que utilizam dados previamente rotulados para treinar modelos capazes de realizar **previsões** ou **classificações**.

    Durante esse processo de treino, o modelo aprende a associar as **variáveis de entrada (features)** com os **resultados esperados (labels)**, ajustando seus parâmetros internos para minimizar o erro entre as previsões geradas e os valores reais.
    """)


    st.subheader("🔍 Algoritmos")

    st.markdown("""
    Nesta secção, treinamos e avaliamos o desempenho inicial de diferentes modelos de classificação usando **validação cruzada** no conjunto de treino pré-processado.

    **Modelos a explorar:**
    - Regressão Logística
    - K-Nearest Neighbors (KNN)
    - Árvore de Decisão
    - Random Forest
    - Gradient Boosting
    - Support Vector Machine (SVM)
    - Rede Neuronal (MLP Classifier)
    """)

    st.markdown("---")
    st.subheader("2.1. Aplicação dos Algoritmos")

    codigo = """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42), 
        'MLP Classifier': MLPClassifier(random_state=42, max_iter=500, early_stopping=True) 
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results_cv = {}

    print("--- Avaliação Inicial com Validação Cruzada Estratificada (Treino) ---")

    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    for name, model in models.items():
        print(f'A avaliar {name}...')
        metrics_scores = {metric: cross_val_score(model, X_train_processed, y_train, cv=skf, scoring=metric) for metric in scoring_metrics}
        results_cv[name] = {metric: scores.mean() for metric, scores in metrics_scores.items()}
        print(f'{name} - Acurácia Média: {results_cv[name]["accuracy"]:.4f}')

    results_cv_df = pd.DataFrame(results_cv).T
    print('\\n--- Resultados Médios da Validação Cruzada (Treino) ---')
    display(results_cv_df.style.highlight_max(axis=0, color='lightgreen'))

    fig = px.bar(results_cv_df.reset_index().melt(id_vars='index', var_name='Métrica', value_name='Score'),
                x='index', y='Score', color='Métrica', barmode='group',
                title='Comparação de Métricas dos Modelos (Validação Cruzada no Treino)',
                labels={'index': 'Modelo'})
    fig.show()
    """

    st.code(codigo, language='python')

    st.markdown("---")
    st.subheader("2.2. Otimização e Avaliação Avançada")
    st.markdown("""
    Com base nos resultados iniciais, selecionamos alguns modelos promissores para otimização de hiperparâmetros e avaliamos o impacto do tratamento do desbalanceamento de classes usando SMOTE.

    *   **Otimização de Hiperparâmetros:** Usar GridSearchCV para encontrar os melhores parâmetros.
    *   **Tratamento de Desbalanceamento (SMOTE):** Aplicar SMOTE no conjunto de treino para lidar com a ligeira predominância da classe 'yes'.
    *   **Avaliação Final no Teste:** Avaliar os modelos otimizados (com e sem SMOTE) no conjunto de teste.
    """)
    
    st.markdown("---")
    st.title("3. Alunos com Necessidades Especiais")

    st.markdown("""
    <div style='text-align: justify'>
    A identificação precoce de alunos com necessidades educativas especiais representa um dos maiores desafios no contexto educacional contemporâneo. Tradicionalmente, esta identificação depende de avaliações psicopedagógicas individualizadas, que embora eficazes, são processos demorados e que exigem recursos significativos. Neste projeto, exploramos uma abordagem complementar baseada em técnicas de aprendizagem não-supervisionada, especificamente a clusterização K-Means, para identificar potenciais grupos de alunos que possam beneficiar de intervenções educativas personalizadas.
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Metodologia de Identificação")
    
    st.markdown("""
    <div style='text-align: justify'>
    A nossa abordagem baseia-se na premissa de que alunos com necessidades educativas especiais frequentemente apresentam padrões distintos em múltiplas variáveis académicas e comportamentais. Utilizando o algoritmo K-Means, conseguimos agrupar os alunos em clusters com características semelhantes, permitindo a identificação de grupos que possam necessitar de maior atenção pedagógica.
    
    O processo de identificação segue várias etapas fundamentais. Primeiramente, selecionamos as variáveis mais relevantes para a análise, focando em indicadores académicos e comportamentais que possam sinalizar necessidades especiais. Em seguida, aplicamos normalização aos dados para garantir que todas as variáveis contribuam equitativamente para a análise, independentemente das suas escalas originais.
    
    Após a preparação dos dados, aplicamos o algoritmo K-Means para agrupar os alunos em três clusters distintos. Esta escolha de três clusters permite-nos identificar grupos com características de alto, médio e baixo desempenho, facilitando a identificação de padrões específicos. Uma vez formados os clusters, analisamos detalhadamente as características de cada grupo, com especial atenção às taxas de aprovação.
    
    O cluster com a menor taxa de aprovação é identificado como potencial grupo de alunos com necessidades educativas especiais. Este grupo tipicamente apresenta um conjunto de características distintivas que podem incluir maior dificuldade de aprendizagem, menor suporte familiar ou escolar, ou outros fatores que contribuem para o baixo desempenho académico.
    </div>
    """, unsafe_allow_html=True)

    codigo = """
        X_special = X_train_special[available_vars].copy()
    scaler = StandardScaler()
    X_special_scaled = scaler.fit_transform(X_special)

    # Aplicar K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    X_train_special['cluster'] = kmeans.fit_predict(X_special_scaled)

    # Adicionar passed_numeric se existir
    if 'passed_numeric' in X_train.columns:
        X_train_special['passed_numeric'] = X_train['passed_numeric']
        available_vars.append('passed_numeric')
    elif 'passed_numeric' in df.columns:  # Se estiver no DataFrame original
        X_train_special['passed_numeric'] = df.loc[X_train.index, 'passed_numeric']
        available_vars.append('passed_numeric')

    # Analisar características por cluster
    cluster_stats = X_train_special.groupby('cluster')[available_vars].mean()
    print("\nCaracterísticas por cluster:")
    print(cluster_stats)

    # Identificar o cluster com menor taxa de aprovação (se passed_numeric existir)
    if 'passed_numeric' in X_train_special.columns:
        approval_by_cluster = X_train_special.groupby('cluster')['passed_numeric'].mean() * 100
        worst_cluster = approval_by_cluster.idxmin()
        print(f"\nCluster {worst_cluster} tem a menor taxa de aprovação: {approval_by_cluster[worst_cluster]:.2f}%")
        print("Este cluster pode representar alunos com necessidades especiais.")

        # Identificar alunos no cluster de risco
        special_needs_candidates = X_train_special[X_train_special['cluster'] == worst_cluster].index
        print(f"Número de alunos no cluster de risco: {len(special_needs_candidates)}")
        # Gráfico de barras para visualização do número de alunos por cluster
        cluster_counts = X_train_special['cluster'].value_counts().sort_index()
        """

    st.code(codigo, language='python')

    st.markdown("---")
    st.title("4. Conclusões sobre os Algoritmos Utilizados")
    
    st.markdown("""
    <div style='text-align: justify'>
    Após a implementação, treino e avaliação dos diversos algoritmos de aprendizagem supervisionada neste projeto, chegamos a conclusões importantes sobre o desempenho e aplicabilidade de cada um deles para o problema de previsão de aprovação de estudantes.
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("4.1. Análise Comparativa dos Modelos")
    
    st.markdown("""
    <div style='text-align: justify'>
    A análise comparativa dos modelos implementados revelou padrões interessantes quanto à sua eficácia na previsão do sucesso académico dos estudantes:
    
    **Random Forest e Decision Tree** destacaram-se pela sua capacidade de capturar relações não-lineares complexas nos dados educacionais. Estes modelos apresentaram um equilíbrio notável entre precisão e capacidade de generalização, sendo particularmente eficazes na identificação de estudantes em risco de reprovação.
    
    **KNN (K-Nearest Neighbors)** demonstrou um desempenho surpreendentemente bom, sugerindo que estudantes com perfis semelhantes tendem a ter resultados académicos semelhantes. Este modelo é especialmente útil quando existem "clusters" naturais de estudantes com características e desempenhos similares.
    
    **Regressão Logística**, apesar da sua simplicidade, mostrou-se menos eficaz para este problema específico, indicando que as relações entre as variáveis preditoras e o sucesso académico não são puramente lineares. No entanto, a sua interpretabilidade continua a ser uma vantagem significativa em contextos educacionais onde a explicabilidade é importante.
    
    **Gradient Boosting** apresentou resultados promissores após otimização, mas com tendência para overfitting quando não adequadamente regularizado. Este modelo beneficiou significativamente da otimização de hiperparâmetros.
    
    **SVM (Support Vector Machine)** mostrou um desempenho moderado, sendo mais eficaz após a normalização dos dados e otimização de parâmetros. A sua capacidade de lidar com fronteiras de decisão complexas foi valiosa para este problema.
    
    **MLP Classifier (Rede Neural)** demonstrou potencial para capturar padrões complexos nos dados, mas exigiu um ajuste cuidadoso para evitar overfitting, especialmente considerando o tamanho limitado do dataset.
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("4.2. Fatores Determinantes para o Sucesso Académico")
    
    st.markdown("""
    <div style='text-align: justify'>
    A análise da importância das features revelou os fatores mais determinantes para o sucesso académico dos estudantes:
    
    1. **Histórico de reprovações anteriores** emergiu consistentemente como o preditor mais forte do desempenho futuro, indicando a importância de intervenções precoces.
    
    2. **Tempo de estudo semanal** mostrou uma correlação positiva significativa com a aprovação, reforçando a importância de hábitos de estudo regulares.
    
    3. **Educação dos pais** (especialmente da mãe) demonstrou um impacto considerável no desempenho académico, sublinhando a influência do ambiente familiar.
    
    4. **Consumo de álcool** (especialmente aos fins-de-semana) correlacionou-se negativamente com o sucesso académico, destacando a importância de abordar questões comportamentais.
    
    5. **Assiduidade** (número de faltas) também se revelou um preditor importante, com maior absentismo associado a menor probabilidade de aprovação.
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("4.3. Limitações e Trabalho Futuro")
    
    st.markdown("""
    <div style='text-align: justify'>
    Apesar dos resultados promissores, identificámos algumas limitações importantes:
    
    - O **tamanho limitado do dataset** (395 estudantes) pode afetar a generalização para populações estudantis mais diversas
    - A **natureza estática dos dados** não captura a evolução do desempenho ao longo do tempo
    - Alguns **fatores potencialmente relevantes** (como saúde mental, motivação intrínseca, qualidade do ensino) não estão representados no dataset
    
    Para trabalho futuro, recomendamos:
    
    - **Expandir o dataset** com dados de mais escolas e contextos educativos diversos
    - Implementar um **sistema de monitorização contínua** que acompanhe a evolução do desempenho dos estudantes
    - Explorar técnicas de **aprendizagem profunda** para capturar padrões temporais em dados educacionais longitudinais
    - Integrar **dados qualitativos** sobre experiências e perceções dos estudantes
    - Desenvolver **interfaces mais interativas** para educadores, permitindo simulações de intervenções e análises de cenários
    </div>
    """, unsafe_allow_html=True)

with tabs[1]:
    st.title("Previsão para Novo Aluno")
    
    st.markdown("""
    Nesta secção, você pode inserir os dados de um novo aluno para prever se ele será aprovado ou não no exame final.
    
    Preencha todos os campos abaixo com as informações do aluno e clique em "Prever" para obter o resultado.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Dados Pessoais")
        school = st.selectbox("Escola", ["GP", "MS"], help="GP - Gabriel Pereira ou MS - Mousinho da Silveira")
        sex = st.selectbox("Sexo", ["F", "M"], help="F - Feminino ou M - Masculino")
        age = st.number_input("Idade", min_value=15, max_value=22, value=16, help="Idade do aluno")
        address = st.selectbox("Endereço", ["U", "R"], help="U - Urbano ou R - Rural")
        famsize = st.selectbox("Tamanho da Família", ["LE3", "GT3"], help="LE3 - Menor ou igual a 3 ou GT3 - Maior que 3")
        Pstatus = st.selectbox("Status dos Pais", ["T", "A"], help="T - Vivendo juntos ou A - Separados")
        
    with col2:
        st.subheader("Educação e Família")
        Medu = st.selectbox("Educação da Mãe", [0, 1, 2, 3, 4], help="0 - nenhuma, 1 - primário, 2 - 5º a 9º ano, 3 - secundário, 4 - superior")
        Fedu = st.selectbox("Educação do Pai", [0, 1, 2, 3, 4], help="0 - nenhuma, 1 - primário, 2 - 5º a 9º ano, 3 - secundário, 4 - superior")
        Mjob = st.selectbox("Profissão da Mãe", ["teacher", "health", "services", "at_home", "other"])
        Fjob = st.selectbox("Profissão do Pai", ["teacher", "health", "services", "at_home", "other"])
        reason = st.selectbox("Razão para escolher a escola", ["home", "reputation", "course", "other"])
        guardian = st.selectbox("Responsável", ["mother", "father", "other"])
        
    with col3:
        st.subheader("Estudo e Tempo Livre")
        traveltime = st.selectbox("Tempo de viagem para a escola", [1, 2, 3, 4], help="1 - <15 min., 2 - 15-30 min., 3 - 30-60 min., 4 - >60 min.")
        studytime = st.selectbox("Tempo de estudo semanal", [1, 2, 3, 4], help="1 - <2 horas, 2 - 2-5 horas, 3 - 5-10 horas, 4 - >10 horas")
        failures = st.selectbox("Número de reprovações anteriores", [0, 1, 2, 3], help="0 - nenhuma, 1 - uma, 2 - duas, 3 - três ou mais")
        
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.subheader("Suporte e Atividades")
        schoolsup = st.selectbox("Suporte educacional extra da escola", ["yes", "no"])
        famsup = st.selectbox("Suporte educacional familiar", ["yes", "no"])
        paid = st.selectbox("Aulas extras pagas", ["yes", "no"])
        activities = st.selectbox("Atividades extracurriculares", ["yes", "no"])
        nursery = st.selectbox("Frequentou jardim de infância", ["yes", "no"])
        
    with col5:
        st.subheader("Aspirações e Relacionamentos")
        higher = st.selectbox("Deseja seguir ensino superior", ["yes", "no"])
        internet = st.selectbox("Acesso à internet em casa", ["yes", "no"])
        romantic = st.selectbox("Em relacionamento romântico", ["yes", "no"])
        famrel = st.selectbox("Qualidade das relações familiares", [1, 2, 3, 4, 5], help="1 - muito má a 5 - excelente")
        
    with col6:
        st.subheader("Lazer e Saúde")
        freetime = st.selectbox("Tempo livre após a escola", [1, 2, 3, 4, 5], help="1 - muito pouco a 5 - muito")
        goout = st.selectbox("Sair com amigos", [1, 2, 3, 4, 5], help="1 - muito pouco a 5 - muito")
        Dalc = st.selectbox("Consumo de álcool durante a semana", [1, 2, 3, 4, 5], help="1 - muito baixo a 5 - muito alto")
        Walc = st.selectbox("Consumo de álcool no fim de semana", [1, 2, 3, 4, 5], help="1 - muito baixo a 5 - muito alto")
        health = st.selectbox("Estado de saúde atual", [1, 2, 3, 4, 5], help="1 - muito mau a 5 - muito bom")
        absences = st.number_input("Número de faltas", min_value=0, max_value=93, value=0)
    
    # Botão para prever
    if st.button("Prever"):
        # Criar um dataframe com os dados do novo aluno
        new_student = pd.DataFrame({
            'school': [school],
            'sex': [sex],
            'age': [age],
            'address': [address],
            'famsize': [famsize],
            'Pstatus': [Pstatus],
            'Medu': [Medu],
            'Fedu': [Fedu],
            'Mjob': [Mjob],
            'Fjob': [Fjob],
            'reason': [reason],
            'guardian': [guardian],
            'traveltime': [traveltime],
            'studytime': [studytime],
            'failures': [failures],
            'schoolsup': [schoolsup],
            'famsup': [famsup],
            'paid': [paid],
            'activities': [activities],
            'nursery': [nursery],
            'higher': [higher],
            'internet': [internet],
            'romantic': [romantic],
            'famrel': [famrel],
            'freetime': [freetime],
            'goout': [goout],
            'Dalc': [Dalc],
            'Walc': [Walc],
            'health': [health],
            'absences': [absences]
        })
        
        # Aplicar as mesmas transformações que foram aplicadas ao dataset original
        new_student_processed = new_student.copy()
        new_student_processed['school'] = new_student_processed['school'].map({'GP': 1, 'MS': 0})
        new_student_processed['sex'] = new_student_processed['sex'].map({'F': 1, 'M': 0})
        new_student_processed['address'] = new_student_processed['address'].map({'R': 1, 'U': 0})
        new_student_processed['famsize'] = new_student_processed['famsize'].map({'GT3': 1, 'LE3': 0})
        new_student_processed['Pstatus'] = new_student_processed['Pstatus'].map({'T': 1, 'A': 0})
        new_student_processed['schoolsup'] = new_student_processed['schoolsup'].map({'yes': 1, 'no': 0})
        new_student_processed['famsup'] = new_student_processed['famsup'].map({'yes': 1, 'no': 0})
        new_student_processed['paid'] = new_student_processed['paid'].map({'yes': 1, 'no': 0})
        new_student_processed['activities'] = new_student_processed['activities'].map({'yes': 1, 'no': 0})
        new_student_processed['nursery'] = new_student_processed['nursery'].map({'yes': 1, 'no': 0})
        new_student_processed['higher'] = new_student_processed['higher'].map({'yes': 1, 'no': 0})
        new_student_processed['internet'] = new_student_processed['internet'].map({'yes': 1, 'no': 0})
        new_student_processed['romantic'] = new_student_processed['romantic'].map({'yes': 1, 'no': 0})
        
        # Aplicar one-hot encoding para variáveis categóricas
        # Mjob
        mjob_cols = ['Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher']
        for col in mjob_cols:
            new_student_processed[col] = 0
        if new_student['Mjob'].iloc[0] != 'at_home':  # at_home é a categoria de referência
            col_name = f"Mjob_{new_student['Mjob'].iloc[0]}"
            if col_name in mjob_cols:
                new_student_processed[col_name] = 1
        
        # Fjob
        fjob_cols = ['Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher']
        for col in fjob_cols:
            new_student_processed[col] = 0
        if new_student['Fjob'].iloc[0] != 'at_home':  # at_home é a categoria de referência
            col_name = f"Fjob_{new_student['Fjob'].iloc[0]}"
            if col_name in fjob_cols:
                new_student_processed[col_name] = 1
        
        # reason
        reason_cols = ['reason_home', 'reason_other', 'reason_reputation']
        for col in reason_cols:
            new_student_processed[col] = 0
        if new_student['reason'].iloc[0] != 'course':  # course é a categoria de referência
            col_name = f"reason_{new_student['reason'].iloc[0]}"
            if col_name in reason_cols:
                new_student_processed[col_name] = 1
        
        # guardian
        guardian_cols = ['guardian_mother', 'guardian_other']
        for col in guardian_cols:
            new_student_processed[col] = 0
        if new_student['guardian'].iloc[0] != 'father':  # father é a categoria de referência
            col_name = f"guardian_{new_student['guardian'].iloc[0]}"
            if col_name in guardian_cols:
                new_student_processed[col_name] = 1
        
        # Remover colunas categóricas originais
        new_student_processed = new_student_processed.drop(['Mjob', 'Fjob', 'reason', 'guardian'], axis=1)
        
        # Garantir que todas as colunas necessárias estão presentes
        for col in X.columns:
            if col not in new_student_processed.columns:
                new_student_processed[col] = 0
        
        # Reordenar colunas para corresponder ao dataset de treino
        new_student_processed = new_student_processed[X.columns]
        
        # Fazer previsões com todos os modelos
        predictions = {}
        probabilities = {}
        
        for name, model in models.items():
            prob = model.predict_proba(new_student_processed)[0][1]  # Probabilidade da classe positiva (passed=1)
            probabilities[name] = prob
            predictions[name] = 1 if prob >= 0.5 else 0  # Previsão binária de cada modelo
        
        # Contar quantos modelos aprovam o aluno
        approving_models = sum(predictions.values())
        total_models = len(models)
        
        # Decisão baseada na maioria dos modelos (mais de metade)
        ensemble_prediction = 1 if approving_models > total_models / 2 else 0
        
        # Exibir resultados
        st.markdown("---")
        st.subheader("Resultado da Previsão")
        
        # Mostrar resultado baseado na maioria dos modelos
        st.markdown(f"### Resultado Final")
        
        if ensemble_prediction == 1:
            st.success(f"O aluno tem alta probabilidade de ser **APROVADO**")
        else:
            st.error(f"O aluno tem alta probabilidade de ser **REPROVADO**")
        