"""Verificação rápida do ambiente e dos módulos já escritos.

Rode com:  python verificar_ambiente.py

Testa cada peça isoladamente e reporta o que funciona e o que quebra, sem
interromper na primeira falha. Nenhum modelo pesado é baixado.
"""

from __future__ import annotations

import sys
import traceback

RESULTADOS: list[tuple[str, bool, str]] = []


def checar(nome: str):
    """Decorador que registra sucesso/falha sem abortar a execução."""
    def wrapper(funcao):
        try:
            detalhe = funcao() or "ok"
            RESULTADOS.append((nome, True, str(detalhe)))
        except Exception as erro:  # noqa: BLE001
            linha = traceback.format_exc().strip().splitlines()[-1]
            RESULTADOS.append((nome, False, linha))
        return funcao
    return wrapper


# --- Versões instaladas -----------------------------------------------------

@checar("versões das bibliotecas")
def _versoes():
    import langchain_core
    import transformers
    partes = [f"langchain-core {langchain_core.__version__}",
              f"transformers {transformers.__version__}"]
    try:
        import langgraph  # noqa: F401
        from importlib.metadata import version
        partes.append(f"langgraph {version('langgraph')}")
    except ImportError:
        partes.append("langgraph ausente")
    try:
        import torch
        partes.append(f"torch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except ImportError:
        partes.append("torch ausente")
    return " | ".join(partes)


# --- Módulos do projeto -----------------------------------------------------

@checar("src.config")
def _config():
    from src import config
    faltando = [
        str(p) for p in (config.DIR_PROTOCOLOS, config.ARQUIVO_PRONTUARIOS)
        if not p.exists()
    ]
    if faltando:
        raise FileNotFoundError(f"caminhos inexistentes: {faltando}")
    return f"raiz={config.RAIZ.name}"


@checar("carregar protocolos")
def _protocolos():
    from src.rag.documentos import carregar_protocolos
    docs = carregar_protocolos()
    sem_codigo = [d for d in docs if not d.metadata.get("codigo")]
    if sem_codigo:
        raise ValueError(f"{len(sem_codigo)} documento(s) sem código no frontmatter")
    return f"{len(docs)} documentos"


@checar("chunking")
def _chunking():
    from src.rag.documentos import carregar_protocolos, dividir_em_chunks
    chunks = dividir_em_chunks(carregar_protocolos())
    vazios = [c for c in chunks if len(c.page_content.strip()) < 20]
    if vazios:
        raise ValueError(f"{len(vazios)} chunk(s) quase vazios")
    tamanhos = [len(c.page_content) for c in chunks]
    return f"{len(chunks)} chunks (min={min(tamanhos)} max={max(tamanhos)})"


@checar("prontuários")
def _prontuarios():
    from src.rag import prontuarios as pr
    pacientes = pr.listar_pacientes()
    p = pr.buscar_paciente(pacientes[0]["id"])
    texto = pr.formatar_para_prompt(p)
    if "Sinais vitais" not in texto:
        raise ValueError("formatação do prompt incompleta")
    com_alerta = sum(
        1 for info in pacientes
        if pr.sinais_de_gravidade(pr.buscar_paciente(info["id"]))
    )
    return f"{len(pacientes)} pacientes, {com_alerta} com sinal de gravidade"


@checar("import do vectorstore")
def _import_vs():
    from src.rag import vectorstore  # noqa: F401
    return "módulo importável"


@checar("langchain_chroma")
def _chroma():
    from langchain_chroma import Chroma  # noqa: F401
    return "disponível"


@checar("langchain_huggingface (embeddings)")
def _hf_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings  # noqa: F401
    return "disponível"


@checar("indexação com embeddings sintéticos")
def _indexacao():
    """Valida a mecânica do Chroma sem baixar o modelo real (2 GB)."""
    import hashlib
    import shutil
    import tempfile
    from pathlib import Path

    from langchain_core.embeddings import Embeddings

    class EmbeddingsSinteticos(Embeddings):
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

    from src.rag.vectorstore import (
        criar_retriever,
        extrair_fontes,
        formatar_contexto,
        indexar,
    )

    temporario = Path(tempfile.mkdtemp())
    try:
        vs = indexar(
            embeddings=EmbeddingsSinteticos(),
            diretorio_persistencia=temporario,
            recriar=True,
        )
        retriever = criar_retriever(vs, top_k=3)
        docs = retriever.invoke("protocolo de sepse")
        if not docs:
            raise ValueError("retriever não devolveu nada")
        contexto = formatar_contexto(docs)
        fontes = extrair_fontes(docs)
        if not contexto or not fontes:
            raise ValueError("contexto ou fontes vazios")
        return f"{len(docs)} trechos, {len(fontes)} fonte(s)"
    finally:
        shutil.rmtree(temporario, ignore_errors=True)


@checar("API do transformers (sem baixar modelo)")
def _transformers_api():
    """Confere se os parâmetros que o código usa ainda existem nesta versão."""
    import inspect

    from transformers import AutoModelForCausalLM

    assinatura = inspect.signature(AutoModelForCausalLM.from_pretrained)
    parametros = set(assinatura.parameters)
    tem_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in assinatura.parameters.values()
    )
    if not tem_kwargs and "device_map" not in parametros:
        raise ValueError("device_map não aceito nesta versão")
    return "assinatura compatível"


@checar("bitsandbytes (quantização 4-bit)")
def _bnb():
    import bitsandbytes  # noqa: F401
    return "disponível"


# --- Relatório --------------------------------------------------------------

def main() -> int:
    largura = max(len(nome) for nome, _, _ in RESULTADOS) + 2
    print("\n" + "=" * 70)
    print("VERIFICAÇÃO DO AMBIENTE")
    print("=" * 70)

    for nome, sucesso, detalhe in RESULTADOS:
        marca = "OK  " if sucesso else "FALHA"
        print(f"  [{marca}] {nome:<{largura}} {detalhe}")

    falhas = [(n, d) for n, ok, d in RESULTADOS if not ok]

    print("=" * 70)
    if not falhas:
        print("Tudo certo — pode seguir para a próxima etapa.")
        return 0

    print(f"{len(falhas)} verificação(ões) falharam:\n")
    for nome, detalhe in falhas:
        print(f"  {nome}\n    {detalhe}\n")
    print("Falha em 'bitsandbytes' é esperada sem GPU NVIDIA — não bloqueia.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
