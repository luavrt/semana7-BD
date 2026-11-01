import os
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()

modelo = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

conn = psycopg2.connect(os.environ["DATABASE_URL"])
register_vector(conn)
cur = conn.cursor()

documentos = [
    ("1FN", "A primeira forma normal exige valores atômicos.", "semanas_1_6"),
    ("JOIN", "INNER JOIN combina linhas de tabelas relacionadas.", "semanas_1_6"),
    ("GROUP BY", "GROUP BY agrupa linhas para funções como COUNT e SUM.", "semanas_1_6"),
    ("Chave estrangeira", "Uma chave estrangeira referencia outra tabela.", "semanas_1_6"),
    ("Redis", "Redis é muito usado para cache e acesso chave-valor.", "extra"),
    ("NoSQL", "NoSQL inclui modelos documento, chave-valor, coluna e grafo.", "semanas_1_6"),
]

cur.execute("DELETE FROM documentos;")

for titulo, conteudo, fonte in documentos:
    embedding = modelo.encode(f"{titulo}. {conteudo}")

    cur.execute(
        '''
        INSERT INTO documentos
            (titulo, conteudo, fonte, embedding)
        VALUES (%s, %s, %s, %s)
        ''',
        (titulo, conteudo, fonte, embedding)
    )

conn.commit()

pergunta = "como fazer um JOIN no SQL"
query_embedding = modelo.encode(pergunta)

cur.execute(
    '''
    SELECT titulo, conteudo, fonte,
           1 - (embedding <=> %s) AS similaridade
    FROM documentos
    ORDER BY embedding <=> %s
    LIMIT 3
    ''',
    (query_embedding, query_embedding)
)

print("TOP-3 PARA:", pergunta)

for titulo, conteudo, fonte, similaridade in cur.fetchall():
    print(f"{similaridade:.4f} | {titulo} | {conteudo}")

#Pratica 4

pergunta4 = "banco de dados para cache chave-valor"
query_embedding4 = modelo.encode(pergunta4)

cur.execute(
    '''
    SELECT titulo, conteudo, fonte,
           1 - (embedding <=> %s) AS similaridade
    FROM documentos
    WHERE fonte = 'semanas_1_6'
    ORDER BY embedding <=> %s
    LIMIT 3
    ''',
    (query_embedding4, query_embedding4)
)
print()
print("PRÁTICA 4 - pergunta:", pergunta4)
for titulo, conteudo, fonte, similaridade in cur.fetchall():
    print(f"{similaridade:.4f} | {titulo} | {fonte} | {conteudo}")

#Pratica 5

def buscar_contexto(pergunta, top_k=3):
    query_embedding = modelo.encode(pergunta)

    cur.execute(
        '''
        SELECT titulo, conteudo, fonte,
               1 - (embedding <=> %s) AS similaridade
        FROM documentos
        ORDER BY embedding <=> %s
        LIMIT %s
        ''',
        (query_embedding, query_embedding, top_k)
    )

    resultados = []

    for titulo, conteudo, fonte, similaridade in cur.fetchall():
        resultados.append({
            "titulo": titulo,
            "conteudo": conteudo,
            "fonte": fonte,
            "similaridade": float(similaridade)
        })

    return resultados


def montar_prompt(pergunta):
    contextos = buscar_contexto(pergunta, top_k=3)

    contexto_texto = "\n\n---\n\n".join(
        f"[{c['titulo']}] {c['conteudo']}"
        for c in contextos
    )

    prompt = f'''
Contexto confiável:
{contexto_texto}

Pergunta do usuário:
{pergunta}

Responda somente com base no contexto acima.
Se o contexto não for suficiente, diga que não encontrou base.
'''

    return contextos, prompt


perguntas = [
    "o que é uma chave estrangeira?",
    "qual banco de dados usar para cache?",
    "o que a primeira forma normal exige?",
]

prompts = []

for i, p in enumerate(perguntas, start=1):
    contextos, prompt = montar_prompt(p)
    prompts.append(prompt)

    print()
    print(f"PERGUNTA {i}: {p}")
    for c in contextos:
        print(f"{c['similaridade']:.4f} | {c['titulo']} | {c['conteudo']}")

print("PROMPT FINAL:")
print(prompts[0])

cur.close()
conn.close()