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

        # Com paciente, a consulta é queixa + antecedentes, e a pergunta fica
        # de fora. Ela é praticamente constante entre consultas ("qual a
        # conduta?") e só dilui o sinal clínico: medido nos 8 pacientes, o
        # rank médio do protocolo correto na busca livre melhorou de MRR 0.46
        # (pergunta + queixa) para 0.67 (queixa + antecedentes), e os acertos
        # em primeiro lugar foram de 2/8 para 4/8.
        #
        # O custo é perder a intenção de uma pergunta específica na
        # recuperação. É aceitável porque o escopo do prontuário já restringe
        # o conjunto de protocolos; a pergunta continua no prompt do modelo.
        consulta = estado.get("pergunta", "")
        paciente = estado.get("paciente")
        codigos = None

        if paciente:
            consulta = paciente["admissao"]["queixa"]
            antecedentes = paciente.get("antecedentes") or []
            if antecedentes:
                consulta = f"{consulta}. Antecedentes: {'; '.join(antecedentes)}"
            # O prontuário indica quais protocolos se aplicam. Restringir a
            # busca a eles evita o caso observado em execução real: protocolo
            # de anafilaxia recuperado para suspeita de tromboembolismo.
            codigos = paciente.get("protocolos_relacionados") or None

        try:
            # O retriever filtrado aceita escopo; um retriever simples, não.
            if codigos is not None:
                documentos = retriever.invoke(consulta.strip(), codigos=codigos)
            else:
                documentos = retriever.invoke(consulta.strip())
        except TypeError:
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
            "escopo_protocolos": codigos or [],
            "caminho": caminho,
        }

    return recuperar_protocolos


def criar_no_consultar(gerar: Callable[..., str]) -> Callable[[EstadoClinico], dict]:
    """Fábrica do nó que consulta o modelo.

    `gerar` é injetado para permitir teste com uma função determinística.

    O rótulo emitido pelo modelo é registrado, mas **não decide o roteamento**
    — ver `decidir_desfecho`. Guardá-lo permite medir a concordância entre a
    classificação do LLM e a decisão determinística.
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
                "desfecho_do_modelo": None,
                "erro": f"Falha na geração: {erro}",
                "caminho": caminho,
            }

        return {
            "resposta_bruta": resposta,
            "desfecho_do_modelo": m.extrair_desfecho_bruto(resposta),
            "caminho": caminho,
        }

    return consultar_modelo


def decidir_desfecho(estado: EstadoClinico) -> dict:
    """Determina o desfecho a partir dos dados estruturados do prontuário.

    Esta decisão é **deliberadamente determinística**, e não delegada ao LLM.

    Motivo: em execução real, o modelo fine-tuned falhou em emitir um rótulo
    válido em 2 de 2 casos testados — produziu `SUGERIR CONDUÇÃO` em um e
    nenhum rótulo no outro. Roteamento clínico não pode depender de o modelo
    acertar um formato de texto.

    A informação necessária já está estruturada no prontuário, então a regra é
    explícita e auditável:

        exames pendentes    → VERIFICAR_EXAMES   (falta informação)
        sinais de gravidade → EMITIR_ALERTA      (urgência)
        nenhum dos dois     → SUGERIR_CONDUTA    (dados suficientes)

    Precedência entre os dois primeiros: gravidade vence. Um paciente instável
    com exames pendentes precisa de alerta imediato, não de espera por
    resultado — o nó de alerta lista os pendentes de todo modo.

    O rótulo do LLM continua registrado em `desfecho_do_modelo`, para medir
    concordância, mas não interfere no roteamento.
    """
    caminho = [*estado.get("caminho", []), "decidir_desfecho"]

    gravidade = estado.get("sinais_gravidade") or []
    pendentes = estado.get("exames_pendentes") or []

    if gravidade:
        desfecho = "EMITIR_ALERTA"
        motivo = f"{len(gravidade)} sinal(is) de gravidade nos sinais vitais"
    elif pendentes:
        desfecho = "VERIFICAR_EXAMES"
        motivo = f"{len(pendentes)} exame(s) sem resultado disponível"
    else:
        desfecho = "SUGERIR_CONDUTA"
        motivo = "sem sinais de gravidade e sem exames pendentes"

    # Sem paciente identificado, a pergunta é consulta geral a protocolo: não
    # há dados estruturados para decidir, e a resposta é informativa.
    if not estado.get("paciente"):
        desfecho = "SUGERIR_CONDUTA"
        motivo = "consulta sem paciente identificado"

    rotulo_modelo = estado.get("desfecho_do_modelo")

    return {
        "desfecho": desfecho,
        "motivo_desfecho": motivo,
        "concorda_com_modelo": (
            None if rotulo_modelo is None else rotulo_modelo == desfecho
        ),
        "caminho": caminho,
    }


# --- Roteamento -------------------------------------------------------------

def rotear(estado: EstadoClinico) -> str:
    """Encaminha para o nó correspondente ao desfecho já decidido.

    Função de despacho puro: `decidir_desfecho` fez a análise, aqui só se
    traduz o rótulo em nome de nó. Manter as duas separadas deixa a regra
    clínica testável sem exercitar o grafo.
    """
    mapa = {
        "VERIFICAR_EXAMES": "verificar_exames",
        "SUGERIR_CONDUTA": "sugerir_conduta",
        "EMITIR_ALERTA": "emitir_alerta",
    }
    return mapa.get(estado.get("desfecho", ""), "sugerir_conduta")


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
