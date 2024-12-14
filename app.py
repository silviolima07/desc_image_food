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

#st.write("GROQ_API_KEY: ",GROQ_API_KEY)

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
        
        #idioma = selecionar_idioma()
        idioma = 'Portuguese'
            

        st.write("LLM Multi Modal to describe image:", llama_mm)    
        st.write("LLM Nutri to answer:", llama.model)

        if idioma == "Portuguese":
            st.markdown("### É saudável ou não ?")
            answer = 'Resultado da Análise'
        else:
            st.markdown("### Junk food or Health food ?")
            answer = "Analysis Result"               
    
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
                    answer_desc = "None food in image"
                    if descricao.lower() != answer_desc.lower():
                        #if idioma == "Portuguese":
                        #    st.markdown("### É saudável ou não ?")
                        #    answer = 'Resultado da Análise'
                        #else:
                        #    st.markdown("### Junk food or Health food ?")
                        #    answer = "Analysis Result"                            
                        
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
    image_path="img/prato.png"    
    prato = Image.open(image_path)
    # Getting the base64 string
    base64_image = encode_image(image_path)
       
    #st.sidebar.image(prato,caption="",use_container_width=True)
    st.markdown("#### Dois agentes: um agente especialista em imagens e um agente nutricionista.")
    st.markdown("#### O primeiro descreve a imagem, focando apenas em alimentos.")
    st.markdown("#### O segundo informa se alimento é saudável.")
    st.markdown("#### Modelos acessados via Groq.")
    st.markdown("#### Exemplo:")
    # Usando HTML para centralizar a imagem
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{base64_image}" alt="Imagem" style="width: 80%; height: auto;">
        </div>
        """,
        unsafe_allow_html=True
    )  
    st.write(" ")     
    st.markdown("### Resultado da análise:")    
    st.write("""
Arroz: não há informações explicitas sobre os nutrientes contidos na descrição, mas como o arroz tem sua classificação em diferentes tipos como, arroz branco e arroz integral, ele contém carboidratos no calórico de 130 e fibras como o RIAI. O benefício para a saúde é que ajuda a controlar o nível de açúcar no sangue e a regular o trânsito intestinal. Alternativa mais saudável: no caso de que você optar pelo arroz integral, pois possui 100 calorias e 2gr de fibras por 100gr.

Vegetais (legumes e verduras): como a descrição, não haver informações sobre os vegetais contidos, os vegetais contém vitaminas, minerais e fibras. Cada xícara de vegetais cozidos fornece cerca de 50 calorias. Mas como na descrição indica, quando se faza desta forma, não se podem dizer qual tipo de vegetal os inclua, mas qual seja ficou para ele ser definido por um especialista. Mas, É provável que os vegetais incluídos sejam ricos em vitamina C, vitamina A, potássio e fibras. O benefício para a saúde é que ajudam a prevenir doenças crônicas, como câncer e doenças cardíacas, e a regular o trânsito intestinal. Alternativa mais saudável: no caso de que você optar pelos vegetais mais probamos saudáveis e menos calórias, pois os vegetais diminuíram as calorias. Porém, como não sabe qual os vegetais inclui a descrição, é difícil identificar qual tipo de vegetal.

Já os alimentos que podem ser considerados menos saudáveis ou que devem ser consumidos em moderação são:

Carne (meat): como a descrição não indica qual tipo de carne, não se sabe quais nutrientes contém, mas é importante notar que a carne vermelha consumida em excesso pode contribuir para o aumento do risco de doenças cardíacas e câncer. Além disso, a gordura presente na carne pode ser alta em calorias e gorduras saturadas. Cada 100g de carne fornece cerca de 250 calorias. O benefício para a saúde é que fornece proteínas essenciais para o crescimento e desenvolvimento. Alternativa mais saudável: optar por cortes de carne mais magros ou por fontes de proteínas mais saudáveis, como peixe ou frango.
Em resumo:

Como mencionado nos alimentos saudáveis citidos, torna-se difícil de saber qual nutrencias correspondem aos alimentdos saudáveis.

Um resumo final:

Relembrando, a avaliação acima pode variar dependendo do tipo de vegetal. Em definitivo, verificado isso faz com que os alimentos possam ou não ser saudáveis.
    .""")                
