import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from typing import Optional

# Importações do LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# --- CONFIGURAÇÃO DA PÁGINA E DO PROJETO ---
st.set_page_config(page_title="Agente de Análise de NF-e", layout="centered")

FILE_CABECALHO = "data/202401_NFs_Cabecalho.csv"
FILE_ITENS = "data/202401_NFs_Itens.csv"
MODEL_NAME = "gemini-1.5-pro-latest"

# --- FUNÇÕES DE INICIALIZAÇÃO DO AGENTE ---


@st.cache_resource
def inicializar_agente():
    """
    Função única que carrega a API, prepara os dados e cria o agente.
    Retorna o agente pronto para uso.
    """
    st.write("Inicializando o agente pela primeira vez...") # Mensagem de status
    
    # 1. Carrega a chave de API
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("A chave GOOGLE_API_KEY não foi encontrada. Verifique seu arquivo .env")

    # 2. Prepara o DataFrame
    try:
        df_cabecalho = pd.read_csv(FILE_CABECALHO)
        df_itens = pd.read_csv(FILE_ITENS)
        
        chave_de_uniao = 'CHAVE DE ACESSO'
        colunas_a_adicionar = [col for col in df_cabecalho.columns if col not in df_itens.columns]
        colunas_a_adicionar.append(chave_de_uniao)
        
        df_completo = pd.merge(df_itens, df_cabecalho[colunas_a_adicionar], on=chave_de_uniao, how='left')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {e}. Verifique a pasta 'data'.")

    # 3. Cria o agente LangChain
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df_completo,
        verbose=True,
        agent_executor_kwargs={"handle_parsing_errors": True},
        allow_dangerous_code=True
    )
    
    return agent

