# Ingestão e Busca Semântica com LangChain e Postgres

Pipeline RAG simples: ingere um PDF em um Postgres com pgVector e responde perguntas
via CLI usando apenas o conteúdo do PDF.

## Tecnologias
- Python + LangChain
- PostgreSQL + pgVector (via Docker)
- OpenAI: embeddings `text-embedding-3-small`, LLM `gpt-5-nano`

## Pré-requisitos
- Python 3.10+
- Docker e Docker Compose
- Uma API Key da OpenAI

## Passo a passo

### 1. Criar e ativar o ambiente virtual

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 3. Configurar variáveis de ambiente

```bash
cp .env.example .env
```

Edite `.env` e preencha `OPENAI_API_KEY` com sua chave da OpenAI.

### 4. Subir o banco de dados

```bash
docker compose up -d
```

### 5. Adicionar o PDF

Coloque o arquivo que deseja indexar na raiz do projeto com o nome `document.pdf`.

### 6. Executar a ingestão

```bash
python src/ingest.py
```

Isso lê o PDF, divide o conteúdo em chunks de 1000 caracteres (overlap de 150),
gera os embeddings e salva tudo no PostgreSQL/pgVector.

### 7. Rodar o chat

```bash
python src/chat.py
```

Digite suas perguntas no terminal. O sistema busca os 10 trechos mais relevantes
do PDF (k=10) e responde somente com base nesse conteúdo. Perguntas fora do
contexto do documento recebem a resposta:
`"Não tenho informações necessárias para responder sua pergunta."`

Digite `sair` para encerrar o chat.

## Estrutura do projeto

```
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── src/
│   ├── ingest.py     # ingestão do PDF no banco vetorial
│   ├── search.py     # busca semântica (similarity_search_with_score)
│   └── chat.py        # CLI de perguntas e respostas
├── document.pdf       # PDF a ser indexado (adicionar antes da ingestão)
└── README.md
```
