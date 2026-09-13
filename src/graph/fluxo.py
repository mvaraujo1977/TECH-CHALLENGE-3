"""Montagem do grafo de decisão e API de alto nível do assistente.

Fluxo:

    START
      ↓
    classificar_risco        (guardrail de entrada, determinístico)
      ↓
    [BLOQUEADO?] ──sim──> recusar ──> END
      ↓ não
    carregar_paciente        (base estruturada)
      ↓
    recuperar_protocolos     (RAG)
      ↓
    consultar_modelo         (LLM fine-tuned — gera o texto)
      ↓
    decidir_desfecho         (regra determinística sobre o prontuário)
      ↓
    [roteamento condicional]
      ├──> verificar_exames
      ├──> sugerir_conduta
      └──> emitir_alerta
              ↓
           finalizar         (guardrail + citação de fontes)
              ↓
            END
"""

from __future__ import annotations

import time
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from src import config
from src.auditoria.registro import Auditoria, RegistroConsulta
from src.graph import nos
from src.graph.estado import EstadoClinico


def construir_grafo(retriever, gerar: Callable[..., str]):
    """Monta e compila o grafo.

    Tanto o retriever quanto a função de geração são injetados, o que permite
    exercitar o grafo inteiro com dublês em teste — sem carregar embeddings
    nem o modelo de 3B.
    """
    grafo = StateGraph(EstadoClinico)

    grafo.add_node("classificar_risco", nos.classificar_risco)
    grafo.add_node("recusar", nos.recusar)
    grafo.add_node("carregar_paciente", nos.carregar_paciente)
    grafo.add_node("recuperar_protocolos", nos.criar_no_recuperar(retriever))
    grafo.add_node("consultar_modelo", nos.criar_no_consultar(gerar))
    grafo.add_node("decidir_desfecho", nos.decidir_desfecho)

    grafo.add_node("verificar_exames", nos.verificar_exames)
    grafo.add_node("sugerir_conduta", nos.sugerir_conduta)
    grafo.add_node("emitir_alerta", nos.emitir_alerta)

    grafo.add_node("finalizar", nos.finalizar)

    grafo.add_edge(START, "classificar_risco")

    # Guardrail de entrada: em BLOQUEADO o fluxo termina sem consultar o modelo
    # nem recuperar protocolo.
    grafo.add_conditional_edges(
        "classificar_risco",
        nos.rotear_risco,
        {"recusar": "recusar", "carregar_paciente": "carregar_paciente"},
    )

    grafo.add_edge("recusar", END)
    grafo.add_edge("carregar_paciente", "recuperar_protocolos")
    grafo.add_edge("recuperar_protocolos", "consultar_modelo")

    grafo.add_edge("consultar_modelo", "decidir_desfecho")

    grafo.add_conditional_edges(
        "decidir_desfecho",
        nos.rotear,
        {
            "verificar_exames": "verificar_exames",
            "sugerir_conduta": "sugerir_conduta",
            "emitir_alerta": "emitir_alerta",
        },
    )

    for no in ("verificar_exames", "sugerir_conduta", "emitir_alerta"):
        grafo.add_edge(no, "finalizar")

    grafo.add_edge("finalizar", END)

    return grafo.compile()


class AssistenteClinico:
    """API de alto nível: recebe pergunta, devolve resposta auditada."""

    def __init__(
        self,
        retriever,
        gerar: Callable[..., str],
        auditoria: Auditoria | None = None,
        nome_modelo: str = "",
    ):
        self.grafo = construir_grafo(retriever, gerar)
        self.auditoria = auditoria or Auditoria()
        self.nome_modelo = nome_modelo

    def consultar(
        self,
        pergunta: str,
        id_paciente: str | None = None,
    ) -> tuple[str, RegistroConsulta]:
        """Executa o fluxo completo e grava o registro de auditoria.

        Devolve a resposta formatada e o registro, para que o chamador possa
        inspecionar o caminho percorrido e as fontes sem reler o arquivo de log.
        """
        inicio = time.time()

        estado_inicial: EstadoClinico = {
            "pergunta": pergunta,
            "id_paciente": id_paciente,
            "caminho": [],
        }

        final: dict[str, Any] = self.grafo.invoke(estado_inicial)
        duracao = time.time() - inicio

        registro = RegistroConsulta(
            pergunta=pergunta,
            id_paciente=id_paciente,
            fontes=final.get("fontes") or [],
            trechos_recuperados=final.get("trechos_recuperados", 0),
            desfecho=final.get("desfecho", ""),
            motivo_desfecho=final.get("motivo_desfecho", ""),
            risco=final.get("risco", ""),
            regras_de_risco=final.get("regras_de_risco") or [],
            versao_politica=final.get("versao_politica", ""),
            desfecho_do_modelo=final.get("desfecho_do_modelo"),
            concorda_com_modelo=final.get("concorda_com_modelo"),
            sinais_gravidade=final.get("sinais_gravidade") or [],
            exames_pendentes=[e["nome"] for e in (final.get("exames_pendentes") or [])],
            resposta=final.get("resposta_final", ""),
            guardrail_adicionado=final.get("guardrail_adicionado", False),
            caminho_no_grafo=final.get("caminho") or [],
            duracao_s=round(duracao, 2),
            modelo=self.nome_modelo,
            erro=final.get("erro"),
        )

        self.auditoria.registrar(registro)
        return registro.resposta, registro


# --- Construção pronta para uso --------------------------------------------

def criar_assistente(
    forcar_cpu: bool = False,
    modelo_base: str | None = None,
    adapter: str | None = None,
    dtype_cpu: str | None = None,
    max_new_tokens: int | None = None,
):
    """Carrega tudo (embeddings, índice, modelo) e devolve o assistente.

    Custosa: baixa o modelo de embeddings e o LLM na primeira execução.
    Para testes, prefira montar o grafo com dublês via `construir_grafo`.

    `dtype_cpu` vai para `carregar_modelo`; `max_new_tokens` é repassado a cada
    chamada de `gerar`, com fallback para `config.MAX_NEW_TOKENS`.
    """
    from src.llm.modelo import carregar_modelo
    from src.llm.modelo import gerar as gerar_resposta
    from src.rag.vectorstore import criar_retriever, indexar

    vectorstore = indexar()
    retriever = criar_retriever(vectorstore)

    mc = carregar_modelo(
        nome_base=modelo_base,
        adapter=adapter,
        forcar_cpu=forcar_cpu,
        dtype_cpu=dtype_cpu,
    )

    tokens = max_new_tokens or config.MAX_NEW_TOKENS

    def gerar(pergunta: str, contexto: str = "") -> str:
        return gerar_resposta(mc, pergunta, contexto=contexto, max_new_tokens=tokens)

    return AssistenteClinico(
        retriever=retriever,
        gerar=gerar,
        nome_modelo=mc.descrever(),
    )
