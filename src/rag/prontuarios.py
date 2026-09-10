"""Acesso à base estruturada de prontuários.

Atende ao requisito "realizar consultas em base de dados estruturadas
(prontuários e registros)". Em produção isto seria um banco relacional ou uma
API do sistema hospitalar; aqui é um JSON com a mesma interface de consulta.
"""

from __future__ import annotations

import json
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
    """Detecta sinais de alarme a partir dos sinais vitais.

    Verificação determinística, independente do modelo. Serve como segunda
    camada: se o LLM classificar um caso grave como rotineiro, esta função
    ainda sinaliza. Os limiares são conservadores e propositalmente simples —
    a intenção é sinalizar para revisão humana, não diagnosticar.
    """
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
