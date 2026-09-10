"""Nós do grafo de decisão clínica.

O fluxo tem duas fases. Primeiro a preparação — carregar o prontuário,
recuperar protocolos, consultar o modelo. Depois o roteamento para um dos três
nós de decisão exigidos pelo desafio: verificar exames pendentes, sugerir
conduta ou emitir alerta.

Cada nó recebe o estado e devolve apenas as chaves que alterou.
"""

from __future__ import annotations

from typing import Callable

from src import config
from src.graph.estado import EstadoClinico
from src.llm import modelo as m
from src.rag import prontuarios as pr


# --- Fase de preparação -----------------------------------------------------

def carregar_paciente(estado: EstadoClinico) -> dict:
    """Busca o prontuário na base estruturada.

    Sem `id_paciente`, a pergunta é tratada como consulta geral a protocolo —
    o fluxo segue normalmente, apenas sem dados de paciente no contexto.
    """
    caminho = [*estado.get("caminho", []), "carregar_paciente"]
    id_paciente = estado.get("id_paciente")

    if not id_paciente:
        return {
            "paciente": None,
            "dados_paciente": "",
            "exames_pendentes": [],
            "sinais_gravidade": [],
            "caminho": caminho,
        }

    paciente = pr.buscar_paciente(id_paciente)

    if paciente is None:
        return {
            "paciente": None,
            "dados_paciente": "",
            "exames_pendentes": [],
            "sinais_gravidade": [],
            "erro": f"Paciente {id_paciente} não encontrado na base",
            "caminho": caminho,
        }

    return {
        "paciente": paciente,
        "dados_paciente": pr.formatar_para_prompt(paciente),
        "exames_pendentes": pr.exames_pendentes(paciente),
        "sinais_gravidade": pr.sinais_de_gravidade(paciente),
        "caminho": caminho,
    }


def criar_no_recuperar(retriever) -> Callable[[EstadoClinico], dict]:
    """Fábrica do nó de recuperação — o retriever vem injetado.

    Injeção de dependência em vez de import direto: permite testar o grafo com
    um retriever falso, sem carregar o modelo de embeddings.
    """
    from src.rag.vectorstore import extrair_fontes, formatar_contexto

    def recuperar_protocolos(estado: EstadoClinico) -> dict:
        caminho = [*estado.get("caminho", []), "recuperar_protocolos"]

        # A busca combina a pergunta com a queixa do paciente. Perguntas como
        # "qual a conduta?" são vagas demais isoladamente; a queixa é o que
        # ancora a recuperação no tema certo.
        consulta = estado.get("pergunta", "")
        paciente = estado.get("paciente")
        if paciente:
            consulta = f"{consulta} {paciente['admissao']['queixa']}"

        try:
            documentos = retriever.invoke(consulta.strip())
        except Exception as erro:  # noqa: BLE001
            return {
                "contexto": "",
                "fontes": [],
                "trechos_recuperados": 0,
                "erro": f"Falha na recuperação: {erro}",
                "caminho": caminho,
            }

        return {
            "contexto": formatar_contexto(documentos),
            "fontes": extrair_fontes(documentos),
            "trechos_recuperados": len(documentos),
            "caminho": caminho,
        }

    return recuperar_protocolos


def criar_no_consultar(gerar: Callable[..., str]) -> Callable[[EstadoClinico], dict]:
    """Fábrica do nó que consulta o modelo.

    `gerar` é injetado para permitir teste com uma função determinística.
    """

    def consultar_modelo(estado: EstadoClinico) -> dict:
        caminho = [*estado.get("caminho", []), "consultar_modelo"]

        pergunta = estado.get("pergunta", "")
        dados = estado.get("dados_paciente", "")
        if dados:
            pergunta = f"{pergunta}\n\nDados do paciente:\n{dados}"

        try:
            resposta = gerar(pergunta, estado.get("contexto", ""))
        except Exception as erro:  # noqa: BLE001
            return {
                "resposta_bruta": "",
                "desfecho": config.DESFECHO_PADRAO,
                "desfecho_do_modelo": False,
                "erro": f"Falha na geração: {erro}",
                "caminho": caminho,
            }

        desfecho_lido = m.extrair_desfecho(resposta)
        veio_do_modelo = resposta.strip().upper().startswith("DESFECHO:")

        return {
            "resposta_bruta": resposta,
            "desfecho": desfecho_lido,
            "desfecho_do_modelo": veio_do_modelo,
            "caminho": caminho,
        }

    return consultar_modelo


# --- Roteamento -------------------------------------------------------------

def rotear(estado: EstadoClinico) -> str:
    """Decide qual nó de decisão executa.

    Três camadas, nesta ordem de precedência:

    1. **Erro** → sugerir_conduta, que sempre carrega a ressalva de validação.
    2. **Sinais de gravidade detectados por código** → emitir_alerta, mesmo que
       o modelo tenha classificado como rotineiro. É um override de segurança:
       na dúvida entre "grave" e "não grave", escalar é o erro mais barato.
    3. **Rótulo do modelo** → o desfecho que ele emitiu.

    A camada 2 é o ponto importante do ponto de vista de segurança. Ela não
    confia na classificação do LLM para casos com sinais vitais alterados.
    """
    if estado.get("erro"):
        return "sugerir_conduta"

    desfecho = estado.get("desfecho", config.DESFECHO_PADRAO)
    gravidade = estado.get("sinais_gravidade") or []

    # Override: gravidade objetiva prevalece sobre classificação branda.
    # Não sobrepõe VERIFICAR_EXAMES — se faltam dados, verificar continua
    # sendo a resposta correta, e o alerta é registrado à parte.
    if gravidade and desfecho == "SUGERIR_CONDUTA":
        return "emitir_alerta"

    mapa = {
        "VERIFICAR_EXAMES": "verificar_exames",
        "SUGERIR_CONDUTA": "sugerir_conduta",
        "EMITIR_ALERTA": "emitir_alerta",
    }
    return mapa.get(desfecho, "sugerir_conduta")


# --- Os três nós de decisão -------------------------------------------------

def verificar_exames(estado: EstadoClinico) -> dict:
    """Faltam resultados essenciais — o assistente recusa conduta definitiva.

    Lista os exames pendentes a partir da base estruturada, não do texto do
    modelo. Se o modelo alucinar um exame que não existe no prontuário, a
    lista aqui continua correta.
    """
    caminho = [*estado.get("caminho", []), "verificar_exames"]
    pendentes = estado.get("exames_pendentes") or []

    acoes = ["Aguardar resultados antes de definir conduta"]
    if pendentes:
        acoes.extend(f"Verificar: {e['nome']}" for e in pendentes)
    else:
        acoes.append(
            "Nenhum exame pendente registrado no prontuário — "
            "confirmar se a informação está atualizada"
        )

    return {"acoes": acoes, "desfecho": "VERIFICAR_EXAMES", "caminho": caminho}


def sugerir_conduta(estado: EstadoClinico) -> dict:
    """Há dados suficientes — descreve o que o protocolo prevê."""
    caminho = [*estado.get("caminho", []), "sugerir_conduta"]

    acoes = ["Revisar a sugestão com o médico responsável antes de executar"]

    fontes = estado.get("fontes") or []
    if fontes:
        codigos = ", ".join(f["codigo"] for f in fontes)
        acoes.append(f"Conferir os protocolos citados: {codigos}")

    if estado.get("erro"):
        acoes.insert(0, "Atenção: houve falha na consulta — resposta pode estar incompleta")

    return {"acoes": acoes, "desfecho": "SUGERIR_CONDUTA", "caminho": caminho}


def emitir_alerta(estado: EstadoClinico) -> dict:
    """Há sinal de gravidade — sinaliza urgência à equipe."""
    caminho = [*estado.get("caminho", []), "emitir_alerta"]

    acoes = ["Acionar imediatamente o médico responsável"]

    gravidade = estado.get("sinais_gravidade") or []
    if gravidade:
        acoes.append("Sinais de alarme identificados:")
        acoes.extend(f"  {sinal}" for sinal in gravidade)

    acoes.append("Manter monitorização contínua até avaliação médica")

    pendentes = estado.get("exames_pendentes") or []
    if pendentes:
        nomes = ", ".join(e["nome"] for e in pendentes)
        acoes.append(f"Priorizar exames pendentes: {nomes}")

    return {"acoes": acoes, "desfecho": "EMITIR_ALERTA", "caminho": caminho}


# --- Finalização ------------------------------------------------------------

def finalizar(estado: EstadoClinico) -> dict:
    """Monta a resposta final: texto do modelo, ações, guardrail e fontes.

    A verificação do guardrail é feita **apenas sobre o texto do modelo**, não
    sobre o bloco de ações. Motivo: ações como "Acionar o médico responsável"
    contêm as mesmas palavras da ressalva de validação, mas não são uma
    ressalva — checar o texto montado produzia falso positivo e deixava a
    resposta sair sem o aviso de segurança.
    """
    caminho = [*estado.get("caminho", []), "finalizar"]

    resposta = (estado.get("resposta_bruta") or "").strip()

    if not resposta and estado.get("erro"):
        resposta = (
            "Não foi possível gerar uma resposta completa para esta consulta. "
            f"Motivo registrado: {estado['erro']}"
        )

    blocos = [resposta] if resposta else []

    acoes = estado.get("acoes") or []
    if acoes:
        # Itens de lista juntos, com quebra simples entre eles; a linha em
        # branco separa apenas o bloco do restante do texto.
        linhas = ["**Ações recomendadas:**", *(f"- {a}" for a in acoes)]
        blocos.append("\n".join(linhas))

    # O guardrail é avaliado sobre a resposta do modelo, isoladamente.
    guardrail_adicionado = not m.tem_guardrail(resposta)
    if guardrail_adicionado:
        blocos.append(config.GUARDRAIL_PADRAO)

    texto = "\n\n".join(blocos) if blocos else "(sem conteúdo)"
    texto = m.citar_fontes(texto, estado.get("fontes") or [])

    return {
        "resposta_final": texto,
        "guardrail_adicionado": guardrail_adicionado,
        "caminho": caminho,
    }
