from langchain_community.document_loaders import YoutubeLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

# testado e feito no colab com os seguintes pip
#  pip install -q langchain_community langchain-huggingface langchain_ollama langchain_openai
# pip install youtube-transcript-api
# pip install pytube
# pip install pydantic==1.*
# pip install langchain>=0.0.267

def model_hf_hub(model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1):
    llm = HuggingFaceHub(
        repo_id=model,
        huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN'),
        model_kwargs={
            "temperature": temperature,
            "top_p": 0.9,
            "return_full_text": False,
            "max_new_tokens": 1024,
        }
    )
    return llm

llm = model_hf_hub()

#* Pedir a url do video no terminal

url_video = input("Apenas cole aqui a url do Vídeo do Youtube")

if not url_video:
    print("não detectamos nenhuma URL, tente novamente")
    exit()

#* Caso queira informações só desativar o comentario do add

try:
    video_loader = YoutubeLoader.from_youtube_url(url_video, language=['pt', 'pt-BR'])
    infos = video_loader.load()
    transcricao = infos[0].page_content

except:
    print('Falha ao recarregar o vídeo e as informações ')
    exit()

system_prompt = "Você é um assistente virtual prestativo e deve responder a uma consulta com base na transcrição de um vídeo, que será fornecida abaixo, por favor evite repetições ."

#* Fazer o input

inputs = 'Consulta: {consulta} \n Transcrição: {transcricao} \n Resuma a transcrição de forma clara e evite repetições.'

user_prompt = '{}'.format(inputs)

prompt_template = ChatPromptTemplate.from_messages([('system', system_prompt), ('user', user_prompt)])

from langchain.chains import LLMChain

chain = LLMChain(
    prompt=prompt_template,
    llm=llm
)


consulta = input("Qual função gostaria de executar? ex: resumir e etc ?")

res = chain.run({'transcricao': transcricao, 'consulta': consulta})
print(res)