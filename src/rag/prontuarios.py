"""Acesso à base estruturada de prontuários.

Atende ao requisito "realizar consultas em base de dados estruturadas
(prontuários e registros)". Em produção isto seria um banco relacional ou uma
API do sistema hospitalar; aqui é um JSON com a mesma interface de consulta.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from src import config

# Status que indicam que o resultado ainda não está disponível.
STATUS_PENDENTES = ("pendente", "coletado_aguardando")


@lru_cache(maxsize=1)
def _carregar(caminho: str | None = None) -> dict:
    arquivo = Path(caminho) if caminho else config.ARQUIVO_PRONTUARIOS
    if not arquivo.exists():
        raise FileNotFoundError(f"Base de prontuários não encontrada: {arquivo}")
    return json.loads(arquivo.read_text(encoding="utf-8"))


def listar_pacientes() -> list[dict]:
    """Resumo de todos os pacientes, para seleção na interface."""
    base = _carregar()
    return [
        {
            "id": p["id"],
            "idade": p["identificacao"]["idade"],
            "sexo": p["identificacao"]["sexo"],
            "leito": p["identificacao"].get("leito", ""),
            "queixa": p["admissao"]["queixa"],
        }
        for p in base["pacientes"]
    ]


def buscar_paciente(id_paciente: str) -> dict | None:
    """Retorna o prontuário completo, ou None se o id não existir."""
    base = _carregar()
    for paciente in base["pacientes"]:
        if paciente["id"].upper() == id_paciente.strip().upper():
            return paciente
    return None


def exames_pendentes(paciente: dict) -> list[dict]:
    """Exames sem resultado disponível."""
    return [
        e for e in paciente.get("exames", [])
        if e.get("status") in STATUS_PENDENTES
    ]


def exames_disponiveis(paciente: dict) -> list[dict]:
    """Exames com resultado já liberado."""
    return [
        e for e in paciente.get("exames", [])
        if e.get("status") == "resultado_disponivel"
    ]


def formatar_para_prompt(paciente: dict) -> str:
    """Serializa o prontuário no formato que o modelo viu durante o treino.

    O fine-tuning usou blocos de texto com sinais vitais e exames em linguagem
    corrida. Manter esse formato preserva o comportamento aprendido — passar um
    JSON cru degradaria a resposta.
    """
    ident = paciente["identificacao"]
    sexo = "masculino" if ident["sexo"].upper() == "M" else "feminino"

    partes = [
        f"Paciente de {ident['idade']} anos, {sexo}.",
        f"Queixa: {paciente['admissao']['queixa']}",
    ]

    antecedentes = paciente.get("antecedentes") or []
    if antecedentes:
        partes.append("Antecedentes: " + "; ".join(antecedentes))

    alergias = paciente.get("alergias") or []
    partes.append("Alergias: " + ("; ".join(alergias) if alergias else "nenhuma referida"))

    medicamentos = paciente.get("medicamentos_em_uso") or []
    if medicamentos:
        partes.append("Medicamentos em uso: " + "; ".join(medicamentos))

    sv = paciente.get("sinais_vitais", {})
    if sv:
        partes.append(
            "Sinais vitais: PA {pa} mmHg, FC {fc} bpm, FR {fr} irpm, "
            "Tax {temp}°C, SpO2 {spo2}%, Glasgow {gcs}".format(
                pa=sv.get("pa_mmhg", "?"),
                fc=sv.get("fc_bpm", "?"),
                fr=sv.get("fr_irpm", "?"),
                temp=sv.get("temp_c", "?"),
                spo2=sv.get("spo2_pct", "?"),
                gcs=sv.get("glasgow", "?"),
            )
        )

    disponiveis = exames_disponiveis(paciente)
    if disponiveis:
        partes.append("Exames disponíveis:")
        for e in disponiveis:
            partes.append(f"  - {e['nome']}: {e['resultado']}")

    pendentes = exames_pendentes(paciente)
    if pendentes:
        nomes = ", ".join(e["nome"] for e in pendentes)
        partes.append(f"Exames pendentes: {nomes}")
    else:
        partes.append("Exames pendentes: nenhum")

    return "\n".join(partes)


def sinais_de_gravidade(paciente: dict) -> list[str]:
    """Detecta sinais de alarme no prontuário.

    Verificação determinística, independente do modelo. Serve como segunda
    camada: se o LLM classificar um caso grave como rotineiro, esta função
    ainda sinaliza. Em execução real isso ocorreu em 2 de 8 casos (AVC em
    janela terapêutica e cetoacidose grave, ambos classificados pelo modelo
    como `SUGERIR_CONDUTA`).

    Duas fontes de sinal, por motivos diferentes:

    - **Sinais vitais**, com limiares conservadores. Cobrem instabilidade
      hemodinâmica e respiratória.
    - **Resultados de exames e protocolos de urgência.** Necessário porque a
      gravidade não sempre aparece nos sinais vitais: um IAMCSST confirmado por
      eletrocardiograma e troponina pode cursar com pressão e saturação
      normais. Foi o que aconteceu com o PAC-002 na primeira execução — supra
      de ST em três derivações e troponina 95 vezes o valor de referência,
      roteado como conduta de rotina porque nenhum sinal vital estava alterado.

    Os limiares são propositalmente simples: a intenção é sinalizar para
    revisão humana, não diagnosticar.
    """
    alertas: list[str] = []
    alertas.extend(_alertas_sinais_vitais(paciente))
    alertas.extend(_alertas_exames(paciente))
    alertas.extend(_alertas_protocolo_urgencia(paciente))
    return alertas


def _alertas_sinais_vitais(paciente: dict) -> list[str]:
    """Alarmes derivados dos sinais vitais."""
    alertas: list[str] = []
    sv = paciente.get("sinais_vitais", {})

    pa = str(sv.get("pa_mmhg", ""))
    if "/" in pa:
        try:
            sistolica, diastolica = (int(v) for v in pa.split("/", 1))
            if sistolica < 90:
                alertas.append(f"Hipotensão (PAS {sistolica} mmHg)")
            elif sistolica >= 180 or diastolica >= 120:
                alertas.append(f"Crise hipertensiva (PA {pa} mmHg)")
        except ValueError:
            pass

    fc = sv.get("fc_bpm")
    if isinstance(fc, int) and (fc > 120 or fc < 50):
        alertas.append(f"Alteração da frequência cardíaca ({fc} bpm)")

    fr = sv.get("fr_irpm")
    if isinstance(fr, int) and fr >= 24:
        alertas.append(f"Taquipneia (FR {fr} irpm)")

    spo2 = sv.get("spo2_pct")
    if isinstance(spo2, int) and spo2 < 92:
        alertas.append(f"Hipoxemia (SpO2 {spo2}%)")

    gcs = sv.get("glasgow")
    if isinstance(gcs, int) and gcs < 15:
        alertas.append(f"Rebaixamento do nível de consciência (Glasgow {gcs})")

    temp = sv.get("temp_c")
    if isinstance(temp, (int, float)) and (temp >= 38.5 or temp < 35.0):
        alertas.append(f"Alteração térmica ({temp}°C)")

    return alertas


# Achados em resultado de exame que caracterizam urgência por si sós.
# A busca é por substring no texto do resultado, então os termos são escolhidos
# para serem específicos: "supradesnivelamento de ST" não aparece em laudo
# normal, ao contrário de palavras como "alteração" ou "elevado".
ACHADOS_CRITICOS = (
    ("supradesnivelamento de st", "Supradesnivelamento de ST no eletrocardiograma"),
    ("supra de st", "Supradesnivelamento de ST no eletrocardiograma"),
    ("bloqueio de ramo esquerdo novo", "Bloqueio de ramo esquerdo novo"),
    ("hemorragia", "Achado hemorrágico em exame de imagem"),
)

# Exames cujo valor numérico caracteriza urgência acima de um limiar.
# (fragmento do nome, rótulo, limiar) — o valor é extraído do texto livre do
# resultado, então a extração é tolerante a formato.
LIMIARES_CRITICOS = (
    ("troponina", "Troponina elevada", 0.04),
    ("lactato", "Lactato elevado", 4.0),
)


def _primeiro_numero(texto: str) -> float | None:
    """Extrai o primeiro número do texto, aceitando vírgula decimal."""
    correspondencia = re.search(r"(\d+(?:[.,]\d+)?)", texto)
    if not correspondencia:
        return None
    try:
        return float(correspondencia.group(1).replace(",", "."))
    except ValueError:
        return None


# Termos que negam o achado quando aparecem imediatamente antes dele.
# Necessário porque a busca por substring é cega a negação: "sem hemorragia"
# contém "hemorragia" e seria contado como achado positivo. Foi o que ocorreu
# no PAC-007, cujo laudo diz "Sem hemorragia ou isquemia aguda" e gerou um
# alerta hemorrágico — inversão do significado do laudo.
NEGACOES = (
    "sem", "não", "nao", "ausência de", "ausencia de", "ausente",
    "negativo para", "negativa para", "exclui", "descarta",
)

# Janela de caracteres antes do achado em que a negação é procurada. Curta o
# bastante para não capturar negação de outra oração ("sem febre; hemorragia
# subaracnóidea" não deve ser lido como negação da hemorragia).
JANELA_NEGACAO = 24


def _esta_negado(texto: str, posicao: int) -> bool:
    """Verifica se há termo de negação pouco antes da posição indicada."""
    inicio = max(0, posicao - JANELA_NEGACAO)
    antes = texto[inicio:posicao]

    # Uma pontuação forte encerra a oração: negação anterior a ela não se
    # aplica ao achado.
    for separador in (";", ".", " e ", ", com", ", e "):
        if separador in antes:
            antes = antes.rsplit(separador, 1)[1]

    return any(re.search(rf"\b{re.escape(n)}\b", antes) for n in NEGACOES)


def _alertas_exames(paciente: dict) -> list[str]:
    """Alarmes derivados de resultados de exames já disponíveis."""
    alertas: list[str] = []

    for exame in exames_disponiveis(paciente):
        nome = str(exame.get("nome", "")).lower()
        resultado = str(exame.get("resultado", ""))
        resultado_baixo = resultado.lower()

        for termo, rotulo in ACHADOS_CRITICOS:
            posicao = resultado_baixo.find(termo)
            if posicao >= 0 and not _esta_negado(resultado_baixo, posicao):
                alertas.append(f"{rotulo} ({exame['nome']})")
                break

        for fragmento, rotulo, limiar in LIMIARES_CRITICOS:
            if fragmento not in nome:
                continue
            valor = _primeiro_numero(resultado)
            if valor is not None and valor > limiar:
                alertas.append(f"{rotulo}: {valor} (referência ≤ {limiar})")

    # Deduplica preservando a ordem — um mesmo achado pode aparecer em mais de
    # um exame (ECG e laudo, por exemplo).
    vistos, unicos = set(), []
    for alerta in alertas:
        if alerta not in vistos:
            vistos.add(alerta)
            unicos.append(alerta)
    return unicos


# Protocolos cuja ativação é, por definição, urgência com tempo-alvo. Quando o
# prontuário associa o paciente a um deles, o caso é tratado como alerta ainda
# que os sinais vitais estejam normais — é o que a curadoria do prontuário está
# afirmando ao fazer essa associação.
PROTOCOLOS_URGENCIA = {
    "PROT-001": "protocolo de sepse ativado",
    "PROT-002": "protocolo de síndrome coronariana aguda ativado",
    "PROT-003": "protocolo de AVC agudo ativado",
    "PROT-009": "protocolo de cetoacidose diabética ativado",
    "PROT-031": "protocolo de anafilaxia ativado",
}


def _alertas_protocolo_urgencia(paciente: dict) -> list[str]:
    """Alarmes derivados dos protocolos que o prontuário associa ao paciente."""
    return [
        f"Protocolo de urgência: {descricao} ({codigo})"
        for codigo in paciente.get("protocolos_relacionados") or []
        if (descricao := PROTOCOLOS_URGENCIA.get(codigo))
    ]
