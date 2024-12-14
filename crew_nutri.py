import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import streamlit as st

from textwrap import dedent

from MyLLM import MyLLM

#from groq import Groq

# Carregar variáveis de ambiente
#load_dotenv()

# Obter a chave da API GROQ
#GROQ_API_KEY = os.getenv('GROQ_API_KEY')

llama = MyLLM.GROQ_LLAMA # model='groq/llama-3.2-3b-preview'

class CrewNutri:
    def __init__(self):
        self.crew = self._criar_crew()

    def _criar_crew(self):
        # Definindo os agentes
        nutri = Agent(
         role="Nutricionista",
         goal="Identificar se alimentos na descrição  são saudáveis ou não.",
         allow_delegation=False,  # Define se o agente pode delegar tarefas
         tools=[], 
         llm=llama,
         verbose=True,
         backstory=dedent("""
              Você é um especialista em nutrição com experiência em identificar comidas saudáveis ou não."""
     )
     )


        # Tarefas
        analise = Task(
        name='analise_imagem',
        description= dedent("""
        Análise os alimentos na descrição: {descricao}.
        Informar o que é saudável ou não.
        """),
        expected_output=dedent(
        """
             Texto claro, traduzido em {idioma}.         
             Um relatório detalhado com:            
             1 - Alimentos identificados na descrição;
             2 - Identificar as vitaminas presentes em cada alimento;
             3 - Informar se alimento é saudável ou não;
             4-  Informar as calorias de cada alimento;
             5 - Informar qual beneficio o alimento oferece para a saúde;
             6 - Informar alternativa mais saudável para trocar pelo alimento.
             7 - Um resumo final do conjunto de alimentos descritos no texto.
             
             Exemplo a ser seguido:
   Exemplo em Português:
* Arroz: boa fonte de carboidratos. Calorias: 216 por xícara. Benefício: regula o açúcar no sangue. Alternativa: arroz integral.

Example in English:
* Rice: good source of carbohydrates. Calories: 216 per cup. Benefit: regulates blood sugar. Alternative: brown rice.
 """),
        
        agent=nutri
    )    

        # Criando o Crew
        return Crew(
            agents=[nutri],
            tasks=[analise],
            process=Process.sequential
        )

    def kickoff(self, inputs):
        # Executa o Crew com o tópico fornecido
        resposta = self.crew.kickoff(inputs=inputs)
        return resposta.raw
