import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class Chatbot:
    def __init__(self):
        self.loader = TextLoader('./horoscope.txt')
        self.documents = self.loader.load()

        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        self.docs = self.text_splitter.split_documents(self.documents)

        self.embeddings = HuggingFaceEmbeddings()

        pinecone.init(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment='gcp-starter'
        )

        index_name = "chatbot-langchain"

        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, metric='cosine', dimension=768)
            self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=index_name)
        else:
            self.docsearch = Pinecone.from_existing_index(index_name, self.embeddings)

        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

        self.llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temprature": 0.8, "top_k": 50},
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        template = """
        You are a fortune teller. These Human will ask you questions about their life.
        Use the following piece of context to answer the question.
        If you don't know the answer, just say you don't know.
        Keep the answer within 2 sentences and concise.

        Context: {context}
        Question: {question}
        Answer:
        """

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        self.rag_chain = (
            {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask_question(self, question):
        result = self.rag_chain.invoke(question)
        print(result.split("Answer:")[-1].strip())

if __name__ == "__main__":
    bot = Chatbot()
    user_input = input("Ask me anything: ")
    bot.ask_question(user_input)
