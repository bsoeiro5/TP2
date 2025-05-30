# Sistema de Previsão de Desempenho Académico
Trabalho realizado por:
Àlvaro Castro (FCUP_IACD:202405722) 
Bernardo Soeiro (FCUP_IACD:2024    ) 
Franciso Machado (FCUP_IACD_202403514)


<p align="center">
  <img src="fotos/Cienciasporto.png" alt="FCUP" width="200"/>
  &nbsp;&nbsp;&nbsp;
  <img src="fotos/Feuporto.png" alt="FEUP" width="200"/>
</p>


## Projeto

O objetivo deste projeto é desenvolver um sistema de intervenção para estudantes, baseado no conjunto de dados UCI Student Performance. Pretende-se construir um pipeline de machine learning capaz de prever se um estudante passará no exame final (`passed` = 'yes'), permitindo a identificação precoce de alunos que necessitam de atenção e suporte adicional.

Este trabalho foi desenvolvido no âmbito da unidade curricular de Extração de Informação e Análise de Conhecimento em Dados (EIACD), seguindo as diretrizes do Trabalho Prático 2 para o ano letivo 2024/2025.

## Streamlit

Para visitar a aplicação Streamlit onde está documentado todo o processo, auxiliado por gráficos relevantes ao projeto, ([clica aqui](https://ketqtv8t6re4ry4mzd4zif.streamlit.app/)).

## Como opera o programa

Este programa trabalha com um interpretador Python e utiliza um ambiente virtual conda para facilitar a instalação das dependências necessárias para a utilização do Jupyter Notebook. O notebook principal está guardado no ficheiro `mfain.ipynb` que, passo a passo, mostra a progressão do projeto, bem como o processo lógico e a forma como abordámos o problema.

A escolha da utilização de um ambiente conda derivou dos seguintes fatores:

Bibliotecas extensas: a implementação de um ambiente virtual automatiza a instalação das bibliotecas, facilitando o acesso e reduzindo o tempo perdido na instalação das mesmas;

Isolamento de dependências: ao criar um ambiente separado, evitam-se conflitos entre bibliotecas de outros projetos e garante-se a compatibilidade;

Organização: sendo este ambiente naturalmente mais reduzido em relação ao ambiente nativo da máquina, a sua utilização mantém a pasta do Python organizada e facilita a identificação de bibliotecas;

Reprodutibilidade: a criação de um ficheiro requirements.txt facilita a partilha e a execução do código em diferentes máquinas, tornando o programa compatível em qualquer máquina;

Gestão de versões: pela simplicidade da ferramenta conda, torna-se fácil instalar e manter diferentes versões de bibliotecas para cada projeto, sem nunca correr o risco de causar conflitos de dependências;

Leveza: a transmissão e instalação do ambiente é facilitada com o ficheiro requirements.txt sendo portanto apenas necessários menos que 30 KB de espaço livre em disco para obter a lista detalhada com todas as bibliotecas utilizadas.

## Etapas do Projeto

O nosso pipeline de machine learning segue as seguintes etapas:

1. **Setup e Carregamento de Dados:** Importação das bibliotecas necessárias e carregamento do dataset UCI Student Performance.

2. **Exploração de Dados (EDA):** Análise inicial para compreender as características do dataset, com foco em visualizações e estatísticas descritivas.

3. **Pré-processamento e Engenharia de Features:** Preparação dos dados para modelagem, incluindo codificação de variáveis categóricas, escalonamento de variáveis numéricas e análise de outliers com PCA.

4. **Modelação:** Treino e avaliação inicial de diversos modelos de aprendizagem supervisionada, incluindo Regressão Logística, KNN, Árvores de Decisão, Random Forest, SVM, Gradient Boosting e Redes Neuronais.

5. **Otimização e Avaliação Avançada:** Otimização de hiperparâmetros, tratamento de desbalanceamento de classes através de técnicas como SMOTE, e avaliação detalhada dos modelos.

6. **Interpretação e Conclusões:** Análise dos resultados, importância das features e conclusões finais.

7. **Análise de IA Responsável:** Investigação de potenciais vieses nos dados e nos modelos.

8. **Deployment:** Implementação de uma aplicação Streamlit para visualização interativa dos resultados.

## Instalar o programa

### Pré-Requisitos
- Conda
- VSCode
- Git (opcional)

### Primeiro passo
Extrair o .zip da página GitHub e descomprimir o ficheiro

OU

Abrir terminal (CMD, PowerShell, Anaconda Prompt, ou outros que reconheçam o comando conda), navegar até a pasta onde deseja instalar o repositório, e introduzir o seguinte código:

```
git clone https://github.com/bsoeiro5/TP2.git
```

### Segundo passo
Caso ainda não o tenha feito, abrir um dos terminais mencionados no passo anterior

### Terceiro passo
Introduzir o seguinte código:

```
cd <diretorio_do_repositorio>
conda create -n dataSci --file requirements.txt
```

E esperar que a instalação esteja concluída

### Quarto passo
Abrir o VSCode e, na barra de pesquisa no topo do ecrã, digitar:

```
>Python: Select Interpreter
```

Clicar Enter, e selecionar o interpretador de Python que tenha como nome Python 3.11.7 ('dataSci')

### Quinto passo
Navegar até ao diretório correto através do terminal e abrir o ficheiro .ipynb:

```
cd <diretorio_do_repositorio>
mfain.ipynb
```

## Bibliotecas utilizadas e as suas versões

As bibliotecas principais são:

- pandas 1.5.3
- numpy 1.26.4
- matplotlib 3.8.4
- seaborn
- plotly.express
- plotly.graph_objects
- scikit-learn 1.4.2
- imblearn (para balanceamento de classes)
- jupyterlab 4.0.11
- streamlit 1.32.0
- altair 5.0.1



