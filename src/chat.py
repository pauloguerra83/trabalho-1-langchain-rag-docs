import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from search import search

load_dotenv()

LLM_MODEL = "gpt-5-nano"

PROMPT_TEMPLATE = """CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def build_prompt(pergunta: str, resultados) -> str:
    contexto = "\n\n".join(doc.page_content for doc, _score in resultados)
    return PROMPT_TEMPLATE.format(contexto=contexto, pergunta=pergunta)


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Erro: defina OPENAI_API_KEY no arquivo .env")
        sys.exit(1)

    llm = ChatOpenAI(model=LLM_MODEL)

    print("Chat pronto. Digite sua pergunta (ou 'sair' para encerrar).\n")

    while True:
        try:
            pergunta = input("PERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if not pergunta:
            continue
        if pergunta.lower() in ("sair", "exit", "quit"):
            print("Encerrando.")
            break

        resultados = search(pergunta, k=10)
        prompt = build_prompt(pergunta, resultados)
        resposta = llm.invoke(prompt)
        print(f"RESPOSTA: {resposta.content}\n")


if __name__ == "__main__":
    main()
