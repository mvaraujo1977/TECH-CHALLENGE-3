"""Carregamento dos protocolos em markdown com extração de metadados.

Cada arquivo tem um frontmatter YAML com `codigo`, `titulo`, `versao`, `setor`
e `revisao`. Esses campos viram metadados dos chunks, que é o que permite citar
a fonte exata na resposta final — o requisito de explainability.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src import config

PADRAO_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def separar_frontmatter(texto: str) -> tuple[dict, str]:
    """Divide o arquivo em (metadados, corpo).

    Retorna metadados vazios se não houver frontmatter, em vez de falhar —
    um documento sem cabeçalho ainda é indexável, só perde a citação rica.
    """
    correspondencia = PADRAO_FRONTMATTER.match(texto)
    if not correspondencia:
        return {}, texto

    try:
        metadados = yaml.safe_load(correspondencia.group(1)) or {}
    except yaml.YAMLError:
        metadados = {}

    corpo = texto[correspondencia.end():]
    return metadados, corpo


def carregar_protocolos(diretorio: Path | None = None) -> list[Document]:
    """Lê todos os .md do diretório de protocolos."""
    diretorio = diretorio or config.DIR_PROTOCOLOS

    if not diretorio.exists():
        raise FileNotFoundError(
            f"Diretório de protocolos não encontrado: {diretorio}"
        )

    documentos = []
    for arquivo in sorted(diretorio.glob("*.md")):
        texto = arquivo.read_text(encoding="utf-8")
        metadados, corpo = separar_frontmatter(texto)

        documentos.append(
            Document(
                page_content=corpo.strip(),
                metadata={
                    "arquivo": arquivo.name,
                    "codigo": str(metadados.get("codigo", arquivo.stem)),
                    "titulo": str(metadados.get("titulo", arquivo.stem)),
                    "versao": str(metadados.get("versao", "")),
                    "setor": str(metadados.get("setor", "")),
                    "revisao": str(metadados.get("revisao", "")),
                },
            )
        )

    if not documentos:
        raise ValueError(f"Nenhum arquivo .md encontrado em {diretorio}")

    return documentos


def dividir_em_chunks(documentos: list[Document]) -> list[Document]:
    """Fatia os documentos preservando a estrutura de seções.

    Os separadores estão em ordem de preferência: quebra por cabeçalho markdown
    primeiro, depois parágrafo, linha e só então caractere. Isso evita cortar
    uma tabela ou lista numerada no meio, o que produziria um chunk sem sentido.
    """
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=config.TAMANHO_CHUNK,
        chunk_overlap=config.SOBREPOSICAO_CHUNK,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        keep_separator=True,
    )

    chunks = divisor.split_documents(documentos)

    # Numera os chunks dentro de cada documento, para citação mais precisa.
    contagem: dict[str, int] = {}
    for chunk in chunks:
        codigo = chunk.metadata.get("codigo", "?")
        indice = contagem.get(codigo, 0)
        chunk.metadata["chunk"] = indice
        contagem[codigo] = indice + 1

    return chunks


def resumir_carga(chunks: list[Document]) -> str:
    """Texto de diagnóstico da indexação, útil para log e demonstração."""
    por_codigo: dict[str, int] = {}
    for chunk in chunks:
        codigo = chunk.metadata.get("codigo", "?")
        por_codigo[codigo] = por_codigo.get(codigo, 0) + 1

    linhas = [f"{len(chunks)} chunks de {len(por_codigo)} documentos:"]
    for codigo, quantidade in sorted(por_codigo.items()):
        linhas.append(f"  {codigo:15s} {quantidade:2d} chunk(s)")
    return "\n".join(linhas)
