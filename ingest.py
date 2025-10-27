from langchain_astradb import AstraDBVectorStore
# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import pandas as pd
from ecommbot.data_converter import dataconverter

load_dotenv()

load_dotenv()
## load the GOOGLE API KEY 
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")


ASTRA_DB_API_ENDPOINT = os.getenv('ASTRA_DB_API_ENDPOINT')
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')

embedding = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

def ingestdata(status):
    vstore = AstraDBVectorStore(
            embedding=embedding,
            collection_name="movie_plots",
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            # namespace=ASTRA_DB_KEYSPACE,
        )
    
    storage=status
    
    if storage==None:
        docs=dataconverter()
        inserted_ids = vstore.add_documents(docs)
    else:
        return vstore
    return vstore, inserted_ids

if __name__=='__main__':
    vstore,inserted_ids=ingestdata(None)
    print(f"\nInserted {len(inserted_ids)} documents.")
    results = vstore.similarity_search("Which film first introduced the character of HAL 9000?")
    for res in results:
            print(f"* {res.page_content} [{res.metadata}]")
            

   
