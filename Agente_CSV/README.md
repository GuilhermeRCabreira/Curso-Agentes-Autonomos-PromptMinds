# 🤖 Agente de Análise de Notas Fiscais com IA

Este projeto é uma aplicação web que permite conversar com dados de Notas Fiscais. Utilizando o Google Gemini e o LangChain, você pode fazer perguntas em português para analisar arquivos CSV de forma simples e intuitiva.
A interface foi construída com Streamlit.

## 🚀 Guia Rápido de Execução
Siga estes 4 passos para executar o projeto localmente.

### Passo 1: Preparar o Projeto
Clone o repositório e navegue até a pasta do projeto.

  ```bash
  git clone https://github.com/GuilhermeRCabreira/Curso-Agentes-Autonomos-PromptMinds.git
  cd Agente_CSV
````
### Passo 2: Instalar as Dependências
Crie um arquivo requirements.txt com o conteúdo abaixo:
 ```bash
  streamlit
  pandas
  python-dotenv
  langchain
  langchain-google-genai
  langchain-experimental
  tabulate
  ```
Em seguida, execute a instalação (o uso de um ambiente virtual é recomendado):
 ```bash
  # Opcional: python -m venv venv && source venv/bin/activate
  pip install -r requirements.txt
````
### Passo 3: Configurar a Chave de API
1. Crie um arquivo chamado .env na pasta principal.

2. Adicione sua chave de API do Google Gemini dentro dele:
 ```bash
GOOGLE_API_KEY="SUA_CHAVE_AQUI"
````
### Passo 4: Executar a Aplicação
Certifique-se de que os arquivos .csv estejam dentro de uma pasta data/. Com tudo pronto, inicie o servidor do Streamlit:
 ```bash
streamlit run app.py
````
A aplicação será aberta automaticamente no seu navegador.
