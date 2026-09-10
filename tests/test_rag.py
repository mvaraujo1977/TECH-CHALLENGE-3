"""Testes da camada de recuperação.

Usam o Chroma real com embeddings sintéticos: verificam a mecânica de
indexação, filtro por metadado e corte de relevância, sem baixar o modelo de
embeddings de ~2 GB.

A qualidade semântica da recuperação não é testada aqui — ela foi medida com o
`bge-m3` real e está registrada em `docs/resultados/validacao_rag.md`.
"""

from __future__ import annotations

import pytest

from src import config
from src.rag.documentos import carregar_protocolos, dividir_em_chunks, separar_frontmatter
from src.rag.vectorstore import criar_retriever, extrair_fontes, formatar_contexto, indexar


class TestCargaDosProtocolos:
    def test_carrega_todos_os_documentos(self):
        assert len(carregar_protocolos()) == 14

    def test_extrai_metadados_do_frontmatter(self):
        for doc in carregar_protocolos():
            assert doc.metadata["codigo"], doc.metadata["arquivo"]
            assert doc.metadata["titulo"]

    def test_frontmatter_ausente_nao_quebra(self):
        """Documento sem cabeçalho ainda é indexável; perde a citação rica."""
        metadados, corpo = separar_frontmatter("# Título\n\nConteúdo.")
        assert metadados == {}
        assert "Conteúdo" in corpo

    def test_frontmatter_malformado_nao_quebra(self):
        metadados, corpo = separar_frontmatter("---\n: : inválido :\n---\n\nTexto.")
        assert isinstance(metadados, dict)
        assert "Texto" in corpo


class TestChunking:
    def test_produz_chunks(self):
        assert len(dividir_em_chunks(carregar_protocolos())) > 14

    def test_nenhum_chunk_vazio(self):
        for chunk in dividir_em_chunks(carregar_protocolos()):
            assert len(chunk.page_content.strip()) >= 20

    def test_respeita_o_tamanho_configurado(self):
        for chunk in dividir_em_chunks(carregar_protocolos()):
            # a sobreposição pode exceder ligeiramente o alvo
            assert len(chunk.page_content) <= config.TAMANHO_CHUNK * 1.5

    def test_metadados_sobrevivem_ao_chunking(self):
        for chunk in dividir_em_chunks(carregar_protocolos()):
            assert chunk.metadata["codigo"]
            assert "chunk" in chunk.metadata

    def test_chunks_sao_numerados_por_documento(self):
        chunks = dividir_em_chunks(carregar_protocolos())
        por_codigo: dict[str, list[int]] = {}
        for chunk in chunks:
            por_codigo.setdefault(chunk.metadata["codigo"], []).append(
                chunk.metadata["chunk"]
            )

        for codigo, indices in por_codigo.items():
            assert indices == list(range(len(indices))), codigo


@pytest.fixture
def indice(embeddings, tmp_path):
    return indexar(
        embeddings=embeddings,
        diretorio_persistencia=tmp_path / "chroma",
        recriar=True,
    )


class TestIndexacao:
    def test_indexa_todos_os_chunks(self, indice):
        esperado = len(dividir_em_chunks(carregar_protocolos()))
        assert indice._collection.count() == esperado

    def test_reabre_indice_existente(self, embeddings, tmp_path):
        diretorio = tmp_path / "chroma"

        primeiro = indexar(embeddings=embeddings, diretorio_persistencia=diretorio, recriar=True)
        contagem = primeiro._collection.count()

        segundo = indexar(embeddings=embeddings, diretorio_persistencia=diretorio, recriar=False)
        assert segundo._collection.count() == contagem


class TestEscopoDeRecuperacao:
    """O escopo curado do prontuário é o mecanismo que garante pertinência.

    Filtrar por score não separava protocolo pertinente de irrelevante:
    medianas de 0.558 e 0.550 em cosseno. Detalhes em `validacao_rag.md`.
    """

    def test_escopo_restringe_aos_codigos_pedidos(self, indice):
        retriever = criar_retriever(indice, top_k=4, limite=0.0)

        docs = retriever.invoke("qualquer consulta", codigos=["PROT-007", "PROT-011"])

        assert docs
        assert {d.metadata["codigo"] for d in docs} <= {"PROT-007", "PROT-011"}

    def test_sem_escopo_busca_livremente(self, indice):
        retriever = criar_retriever(indice, top_k=4, limite=0.0)
        assert retriever.invoke("qualquer consulta")

    def test_escopo_ignora_o_corte_de_relevancia(self, indice):
        """Dentro do escopo, o corte não se aplica: dois chunks do protocolo de
        sepse ficam abaixo dele em execução real, e descartá-los deixaria o caso
        sem protocolo."""
        retriever = criar_retriever(indice, top_k=4, limite=0.99)

        assert retriever.invoke("consulta", codigos=["PROT-001"])

    def test_corte_vale_para_busca_livre(self, indice):
        """Sem escopo, o corte é o único piso contra recuperação irrelevante."""
        retriever = criar_retriever(indice, top_k=4, limite=0.999)

        assert retriever.invoke("consulta") == []

    def test_escopo_inexistente_devolve_vazio(self, indice):
        """Regressão: havia um fallback que repetia a busca sem escopo quando
        nada era encontrado. Ele anulava o escopo justamente nos casos em que
        o escopo era necessário, reintroduzindo a contaminação entre condições."""
        retriever = criar_retriever(indice, top_k=4, limite=0.0)

        assert retriever.invoke("consulta", codigos=["PROT-INEXISTENTE"]) == []


class TestFormatacaoDoContexto:
    def test_prefixa_cada_trecho_com_codigo_e_titulo(self, indice):
        retriever = criar_retriever(indice, top_k=2, limite=0.0)
        docs = retriever.invoke("consulta", codigos=["PROT-001"])

        contexto = formatar_contexto(docs)

        assert "[PROT-001" in contexto
        assert "Sepse" in contexto

    def test_contexto_vazio_e_explicito(self):
        """Dizer que nada foi encontrado é melhor que devolver string vazia."""
        assert "nenhum protocolo" in formatar_contexto([]).lower()


class TestExtracaoDeFontes:
    def test_deduplica_por_codigo(self, indice):
        retriever = criar_retriever(indice, top_k=4, limite=0.0)
        docs = retriever.invoke("consulta", codigos=["PROT-001"])

        fontes = extrair_fontes(docs)

        assert len(fontes) == 1
        assert fontes[0]["codigo"] == "PROT-001"

    def test_preserva_indices_de_chunk(self, indice):
        """Rastreabilidade até a seção do documento."""
        retriever = criar_retriever(indice, top_k=4, limite=0.0)
        docs = retriever.invoke("consulta", codigos=["PROT-001"])

        assert extrair_fontes(docs)[0]["chunks"]

    def test_inclui_metadados_para_citacao(self, indice):
        retriever = criar_retriever(indice, top_k=2, limite=0.0)
        docs = retriever.invoke("consulta", codigos=["PROT-007"])

        fonte = extrair_fontes(docs)[0]

        for campo in ("codigo", "titulo", "versao", "arquivo"):
            assert campo in fonte

    def test_sem_documentos_devolve_lista_vazia(self):
        assert extrair_fontes([]) == []
