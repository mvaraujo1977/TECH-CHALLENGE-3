"""Fixtures compartilhadas.

Os testes exercitam o pipeline inteiro **sem carregar o modelo de 3B nem o de
embeddings**. Isso é possível porque os nós do grafo recebem o retriever e a
função de geração por injeção de dependência — decisão de projeto tomada
justamente para viabilizar teste em máquina sem GPU.

Consequência prática: a suíte roda em segundos, e pode entrar em CI.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))


class EmbeddingsSinteticos(Embeddings):
    """Embeddings determinísticos por hash.

    Não têm significado semântico — servem para exercitar a mecânica do vector
    store (indexação, filtro por metadado, persistência) sem baixar os ~2 GB do
    bge-m3. Testes que dependem de semântica real não usam este dublê; a
    qualidade da recuperação é medida à parte, em `docs/resultados/validacao_rag.md`.
    """

    dim = 64

    def _vetor(self, texto: str) -> list[float]:
        digest = hashlib.sha256(texto.encode()).digest()
        bruto = [digest[i % len(digest)] / 255.0 for i in range(self.dim)]
        norma = sum(x * x for x in bruto) ** 0.5
        return [x / norma for x in bruto]

    def embed_documents(self, textos):
        return [self._vetor(t) for t in textos]

    def embed_query(self, texto):
        return self._vetor(texto)


class RetrieverDuble:
    """Retriever que devolve documentos fixos, respeitando o escopo pedido."""

    def __init__(self, documentos: list[Document] | None = None):
        self.documentos = documentos if documentos is not None else [
            Document(
                page_content="## Bundle da primeira hora\n1. Coletar lactato...",
                metadata={
                    "codigo": "PROT-001",
                    "titulo": "Sepse e Choque Séptico",
                    "versao": "3",
                    "chunk": 0,
                    "arquivo": "PROT-001-sepse.md",
                },
            )
        ]
        self.ultima_consulta: str | None = None
        self.ultimo_escopo: list[str] | None = None

    def invoke(self, consulta: str, codigos: list[str] | None = None):
        self.ultima_consulta = consulta
        self.ultimo_escopo = codigos
        if codigos:
            return [
                d for d in self.documentos
                if d.metadata.get("codigo") in codigos
            ]
        return self.documentos


class GeradorDuble:
    """Gerador com resposta programável, registrando o que recebeu."""

    def __init__(self, resposta: str = "Conforme o protocolo, avaliar o caso."):
        self.resposta = resposta
        self.ultima_pergunta: str | None = None
        self.ultimo_contexto: str | None = None

    def __call__(self, pergunta: str, contexto: str = "") -> str:
        self.ultima_pergunta = pergunta
        self.ultimo_contexto = contexto
        if isinstance(self.resposta, Exception):
            raise self.resposta
        return self.resposta


@pytest.fixture
def embeddings():
    return EmbeddingsSinteticos()


@pytest.fixture
def retriever():
    return RetrieverDuble()


@pytest.fixture
def gerador():
    return GeradorDuble()


@pytest.fixture
def auditoria(tmp_path):
    from src.auditoria.registro import Auditoria

    return Auditoria(diretorio=tmp_path / "logs", nome="teste.jsonl")


@pytest.fixture
def assistente(retriever, gerador, auditoria):
    from src.graph.fluxo import AssistenteClinico

    return AssistenteClinico(
        retriever=retriever,
        gerar=gerador,
        auditoria=auditoria,
        nome_modelo="duble",
    )
