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
    )


def criar_retriever(vectorstore, top_k: int | None = None):
    """Retriever de similaridade simples."""
    return vectorstore.as_retriever(
        search_kwargs={"k": top_k or config.TOP_K}
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
