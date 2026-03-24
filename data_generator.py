import os
from langchain_core.documents import Document
from agents.db_hooks import get_chunks_to_generate
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from utils.get_llm import get_gpt
from utils.get_embedd_model import embedding_model


response = get_chunks_to_generate()

docs = [Document(page_content=row['text'], metadata=row['metadata']) for row in response]

mercant = Persona(
    name="Mukesh, the Kirana Merchant",
    role_description="A Fino Mitra agent running a neighborhood shop. He serves as a physical touchpoint for rural customers. His primary goal is accuracy in transaction rules, commission structures, and step-by-step operational guidance for services like AePS, DMT, and account opening."
)
urban_worker = Persona(
    name="Rajesh, the Urban Migrant",
    role_description="A construction worker in a metro city who sends money home to his village. He is cautious about fees and transaction security. He requires simple, jargon-free language and high reassurance ('Fikar Not') regarding the safety of his hard-earned money."
)
tech_savvy_youngster = Persona(
    name="Ananya, the Digital Student",
    role_description="A young, mobile-first user holding a Bhavishya Savings Account. She is highly comfortable with UPI and mobile banking. She expects instant, crisp answers about digital features, account upgrades, and modern banking benefits."
)
compliance_officer = Persona(
    name="Suresh, the Operations Manager",
    role_description="An internal Fino Bank employee based at the Mumbai HQ. He uses the system to verify RBI regulatory compliance, internal policy updates, and KYC guidelines. He requires high technical precision, specific data points, and document citations for every answer."
)

personas = [mercant, urban_worker, tech_savvy_youngster, compliance_officer]

gen_llm = get_gpt()
embeddings = embedding_model()

generator = TestsetGenerator.from_langchain(
    llm=gen_llm,
    embedding_model=embeddings
    )

testset = generator.generate_with_langchain_docs(
    documents=docs,
    testset_size=4,
    query_distribution={"simple":0.4, "reasoning":0.4, "multi_context":0.2}
)

testset.save("testset.jsonl")



    