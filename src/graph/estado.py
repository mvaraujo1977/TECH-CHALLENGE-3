"""Estado compartilhado entre os nós do grafo.

Um TypedDict em vez de dataclass porque é o que o LangGraph espera para fazer
merge parcial: cada nó devolve só as chaves que alterou, e o framework compõe.
"""

from __future__ import annotations

from typing import Any, TypedDict


class EstadoClinico(TypedDict, total=False):
    """Dados que atravessam o grafo, do input à resposta final."""

    # --- Entrada ---
    pergunta: str
    id_paciente: str | None

    # --- Preenchido por carregar_paciente ---
    paciente: dict | None
    dados_paciente: str          # prontuário serializado para o prompt
    exames_pendentes: list[dict]
    sinais_gravidade: list[str]

    # --- Preenchido por recuperar_protocolos ---
    contexto: str                # trechos concatenados para o prompt
    fontes: list[dict]           # metadados para citação
    trechos_recuperados: int
    escopo_protocolos: list[str] # códigos a que a busca foi restrita, se houve

    # --- Preenchido por consultar_modelo ---
    resposta_bruta: str
    desfecho_do_modelo: str | None   # rótulo que o LLM emitiu; None se inválido

    # --- Preenchido por decidir_desfecho ---
    desfecho: str                    # decisão determinística, a que vale
    motivo_desfecho: str             # regra que produziu a decisão
    concorda_com_modelo: bool | None # None quando o LLM não emitiu rótulo

    # --- Preenchido pelos nós de decisão ---
    acoes: list[str]             # o que o nó determinou que deve acontecer
    resposta_final: str
    guardrail_adicionado: bool

    # --- Rastreamento ---
    caminho: list[str]           # nós percorridos, na ordem
    erro: str | None
    extras: dict[str, Any]
