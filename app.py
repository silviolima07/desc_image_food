from PIL import Image
import streamlit as st
from MyLLM import MyLLM
from textwrap import dedent
from dotenv import load_dotenv
from groq import Groq
import os
from utils import encode_image, image_to_text, selecionar_idioma, executar_crew

#gpt = MyLLM.GPT4o_mini # model='gpt-4o-mini'
llama_mm = 'llama-3.2-11b-vision-preview' # Modelo Multi Modal para ler a imagem e descrever

llama = MyLLM.GROQ_LLAMA # model='groq/llama-3.2-3b-preview'

from crew_nutri import CrewNutri

crew_nutricao = CrewNutri()

# Carregar variáveis de ambiente
load_dotenv()

# Obter a chave da API GROQ
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
#OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

st.write("GROQ_API_KEY: ",GROQ_API_KEY)
#st.write("OPENAI_API_KEY: ",OPENAI_API_KEY)

client = Groq(api_key=GROQ_API_KEY)


# Configuração do Streamlit
#st.title ('Sistema de Postagem com CrewAI')

html_page_title = """
<div style="background-color:black;padding=60px">
        <p style='text-align:center;font-size:60px;font-weight:bold; color:red'>Descrição de Alimentos em Imagens</p>
</div>
"""               
st.markdown(html_page_title, unsafe_allow_html=True)

garfo_colher = Image.open("img/garfo_colher1.png")
st.sidebar.image(garfo_colher,caption="",use_container_width=False)

st.sidebar.markdown("# Menu")
option = st.sidebar.selectbox("Menu", ["Image", 'About'], label_visibility='hidden')

if option == 'Image':
    st.markdown("#### Objetivo: informar se é saudável ou não.")
    
    try:
        st.markdown("## Upload Image")
        uploaded_img = st.file_uploader("Envie imagem em PNG/JPG", type=['png', 'jpeg'])
        if uploaded_img is not None:
            img = Image.open(uploaded_img) # Load the image
            image_path="image.png"
            img.save(image_path)
            # Getting the base64 string
            base64_image = encode_image(image_path)
            # Usando HTML para centralizar a imagem
            st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{base64_image}" alt="Imagem" style="width: 80%; height: auto;">
        </div>
        """,
        unsafe_allow_html=True
    )
            
            prompt = dedent("""         
    You are an expert assistant in recognizing and describing foods in images with precision.
    
    Your role is to analyze images and provide description of foods, giving details, like color,size and accurate descriptions of foods.

    Always considering only foods present in the image.
    
    Never describe cars, trucks,  places in image, focus on foods only.
    
    If the image contains no food, respond only with the phrase: 'None food in image.'
    
    
}
    """)
        st.write(" ")
        
        idioma = selecionar_idioma()

        st.write("LLM Multi Modal to describe image:", llama_mm)    
        st.write("LLM Nutri to answer:", llama.model)
    
        st.write(" ")

        # Botão para iniciar o processo
        if st.button ('Iniciar Processo ') :
            # Quanto clicar no botão carrega um loader
            with st.spinner ('Wait for it...we are working...please') :
                #result = crew_postagem.kickoff ( inputs ={ 'topic': tema })
                try:
                    descricao = image_to_text(client, llama_mm, base64_image, prompt)
                    # Exibindo a descricao
                    #st.write("Descrição da imagem:")
                        
                    #st.write(descricao)
                        
                    inputs = {
                      'descricao': descricao,
                      'idioma': idioma}
            
                    # Executando a crew
                    answer = "None food in image"
                    if descricao.lower() != answer.lower():
                        if idioma == "Portuguese":
                            st.markdown("### É saudável ou não ?")
                            answer = 'Resultado da Análise'
                        else:
                            st.markdown("### Junk food or Health food ?")
                            answer = "Analysis Result"                            
                        
                        resultado = executar_crew(crew_nutricao, inputs)
                        #result_text = resultado.raw
                        st.markdown("### "+ answer)
                        st.write(resultado)
                        # Exibindo o texto com um tamanho de fonte maior
                        # Substituindo quebras de linha por <br> e aplicando o estilo a todo o texto
                        #st.markdown(f"<div style='font-size:23px'>{result_text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("### "+descricao)
                        
                except Exception as e:
                    st.error(f"Ocorreu um erro ao executar a crew: {e}")
    except Exception as e:
                    st.error(f"Houston we have a problem.")   
            
if option == 'About':
    prato = Image.open("img/prato.png")
    st.sidebar.image(prato,caption="",use_container_width=True)
    st.markdown("### Dois agentes: um agente especialista em imagens, usando um modelo multimodal, descreve a imagem, focando apenas em alimentos. A descrição é analisada por agente nutricionista, que informa se alimento é saudável.")
    st.markdown("### Modelos acessados via Groq.")
    st.markdown("### Exemplo:")
    st.write("""
    Ao analisar os alimentos descritos na imagem, podemos identificar os seguintes alimentos saudáveis:

* Salada com alface, tomate e cenoura: a alface é rica em fibras e vitamina A, enquanto os tomates fornecem vitamina C e lycopene, e as cenouras são ricas em vitamina A e fibras.
* Tomates: ricos em vitamina C e lycopene, um antioxidante que ajuda a prevenir doenças crônicas.
* Cenouras: são ricas em vitamina A e fibras, importante para a saúde dos olhos e do sistema imunológico.

Já os alimentos que podem ser considerados menos saudáveis ou que devem ser consumidos em moderação são:

* Carne (beef e sausage): embora sejam boas fontes de proteínas, a carne vermelha consumida em excesso pode contribuir para o aumento do risco de doenças cardíacas e câncer. Além disso, a gordura presente na carne pode ser alta em calorias e gorduras saturadas. O consumo exagerado de sausage também pode levar ao consumo excessivo de sódio e preservantes.
* Arroz: embora seja uma boa fonte de carboidratos, o arroz branco refinado pode ser pobre em fibras e nutrientes essenciais. É importante optar por opções mais saudáveis, como arroz integral.
Em resumo, a imagem apresenta uma refeição equilibrada, com uma balança adequada entre proteínas, 
carboidratos e vegetais. No entanto, é importante ter cuidado com a quantidade de carne e arroz 
consumidos e optar por opções mais saudáveis. Além disso, a adição de salsa ao prato pode ser benéfica, 
pois fornece vitamina C e flavonoides..""")                
