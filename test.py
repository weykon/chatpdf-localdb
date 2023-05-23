from langchain.llms import OpenAI
import os
apikey=os.environ.get('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = str(apikey)
OpenAI(temperature=0.6)
