"""Indexação e recuperação dos protocolos com Chroma.

O índice persiste em disco (`.chroma/`), então a ingestão roda uma vez e as
execuções seguintes apenas abrem a coleção existente.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src import config
from src.rag.documentos import carregar_protocolos, dividir_em_chunks

# O Chroma usa L2 quando a coleção não declara a métrica, e o LangChain então
# converte a distância com a transformação euclidiana (1 - d/√2). O resultado
# é uma faixa comprimida que não é similaridade de cosseno — medido nos 8
# pacientes, relevantes e irrelevantes ficaram todos entre 0.21 e 0.46, com
# medianas separadas por 0.004. Declarar cosseno torna o score interpretável e
# o limiar comparável ao que se publica sobre o bge-m3.
#
# Trocar esta constante invalida o índice: reindexe com `recriar=True`.
ESPACO_DISTANCIA = {"hnsw:space": "cosine"}


def criar_embeddings(nome_modelo: str | None = None) -> Embeddings:
    """Instancia o modelo de embeddings.

    Import tardio: `sentence-transformers` puxa torch e leva alguns segundos
    para carregar. Módulos que só precisam do parsing de documentos não devem
    pagar esse custo.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=nome_modelo or config.MODELO_EMBEDDINGS,
        encode_kwargs={"normalize_embeddings": True},
    )


def indexar(
    embeddings: Embeddings | None = None,
    diretorio_persistencia: Path | None = None,
    recriar: bool = False,
):
    """Constrói (ou reabre) a coleção de protocolos.

    Com `recriar=True`, apaga o índice existente antes — necessário sempre que
    os documentos ou os parâmetros de chunking mudarem, já que o Chroma não
    detecta essas alterações sozinho.
    """
    from langchain_chroma import Chroma

    diretorio = Path(diretorio_persistencia or config.DIR_VECTORSTORE)
    embeddings = embeddings or criar_embeddings()

    if recriar and diretorio.exists():
        import shutil

        shutil.rmtree(diretorio)

    ja_existe = diretorio.exists() and any(diretorio.iterdir())

    if ja_existe:
        return Chroma(
            collection_name=config.NOME_COLECAO,
            embedding_function=embeddings,
            persist_directory=str(diretorio),
        )

    chunks = dividir_em_chunks(carregar_protocolos())
    diretorio.mkdir(parents=True, exist_ok=True)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.NOME_COLECAO,
        persist_directory=str(diretorio),
        collection_metadata=ESPACO_DISTANCIA,
    )


class RetrieverFiltrado:
    """Retriever com corte por relevância e escopo opcional por protocolo.

    Duas diferenças em relação ao retriever padrão do Chroma:

    1. **Corte por relevância.** O `as_retriever` devolve sempre `k`
       documentos, mesmo quando nenhum é pertinente. Em execução real, uma
       consulta sobre tromboembolismo recuperou o protocolo de anafilaxia, e o
       modelo — treinado para sempre citar uma fonte — produziu conduta de
       anafilaxia com citação formalmente correta. O risco não é alucinação de
       código, é conduta errada com fonte legítima.

    2. **Escopo por código.** Quando o prontuário indica quais protocolos se
       aplicam ao paciente (`protocolos_relacionados`), a busca é restrita a
       eles. É a informação mais confiável disponível: veio da curadoria do
       prontuário, não de similaridade de texto.

    O corte **só se aplica à busca livre**, sem escopo. Dentro de um escopo
    curado a pergunta já não é "isto é pertinente?", e sim "qual trecho destes
    protocolos responde melhor" — aí o corte só subtrai.

    Havia aqui um fallback que repetia a busca sem escopo quando nada passava
    do corte. Foi removido: medido nos 8 pacientes, ele disparava exatamente
    nos casos em que o escopo era necessário e anulava a proteção. Em 3 dos 8,
    o escopo correto ficava abaixo do corte e o fallback devolvia protocolo de
    outra condição — PROT-007 (TEP) para uma sepse, e anafilaxia, controle
    glicêmico e AVC para uma cefaleia súbita. A recuperação passava a produzir
    a contaminação que o escopo existia para impedir.
    """

    def __init__(self, vectorstore, top_k: int, limite: float):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.limite = limite

    def invoke(self, consulta: str, codigos: list[str] | None = None):
        filtro = None
        if codigos:
            # Chroma aceita `$in` para restringir a um conjunto de valores.
            filtro = {"codigo": {"$in": list(codigos)}}

        try:
            pares = self.vectorstore.similarity_search_with_relevance_scores(
                consulta, k=self.top_k, filter=filtro
            )
        except (TypeError, NotImplementedError):
            # Backends sem suporte a score de relevância: cai para a busca
            # simples, sem corte. Perde o filtro, não a funcionalidade.
            documentos = self.vectorstore.similarity_search(
                consulta, k=self.top_k, filter=filtro
            )
            return documentos

        # Dentro de um escopo curado, todos os candidatos já são pertinentes
        # por construção — devolve os melhores sem corte.
        if codigos:
            return [doc for doc, _ in pares]

        return [doc for doc, score in pares if score >= self.limite]


def criar_retriever(
    vectorstore,
    top_k: int | None = None,
    limite: float | None = None,
):
    """Retriever com corte de relevância."""
    return RetrieverFiltrado(
        vectorstore=vectorstore,
        top_k=top_k or config.TOP_K,
        limite=config.LIMITE_RELEVANCIA if limite is None else limite,
    )


def formatar_contexto(documentos: list[Document]) -> str:
    """Monta o bloco de contexto que vai no prompt.

    Cada trecho é prefixado com o código e o título do protocolo. Isso dá ao
    modelo a informação necessária para citar a fonte correta, em vez de
    inventar um código — que é o comportamento que o fine-tuning ensinou.
    """
    if not documentos:
        return "(nenhum protocolo relevante encontrado)"

    blocos = []
    for doc in documentos:
        codigo = doc.metadata.get("codigo", "?")
        titulo = doc.metadata.get("titulo", "")
        blocos.append(f"[{codigo} — {titulo}]\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(blocos)


def extrair_fontes(documentos: list[Document]) -> list[dict]:
    """Lista de fontes únicas, para o registro de explainability.

    Deduplica por código: vários chunks do mesmo protocolo contam como uma
    fonte só, mas os índices de chunk são preservados para rastreabilidade.
    """
    fontes: dict[str, dict] = {}

    for doc in documentos:
        codigo = doc.metadata.get("codigo", "?")
        if codigo not in fontes:
            fontes[codigo] = {
                "codigo": codigo,
                "titulo": doc.metadata.get("titulo", ""),
                "versao": doc.metadata.get("versao", ""),
                "setor": doc.metadata.get("setor", ""),
                "arquivo": doc.metadata.get("arquivo", ""),
                "chunks": [],
            }
        indice = doc.metadata.get("chunk")
        if indice is not None:
            fontes[codigo]["chunks"].append(indice)

    return list(fontes.values())
