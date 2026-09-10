"""Testes da regra de decisão de desfecho.

Esta é a parte mais crítica do sistema: define se um paciente é tratado como
urgência, como caso incompleto ou como conduta de rotina. A decisão é
determinística justamente para poder ser testada assim, sem envolver o LLM.
"""

from __future__ import annotations

import pytest

from src import config
from src.graph.nos import decidir_desfecho, rotear


def estado(paciente=True, gravidade=None, pendentes=None, rotulo_llm=None):
    """Monta um estado mínimo para a regra de decisão."""
    return {
        "paciente": {"id": "PAC-TESTE"} if paciente else None,
        "sinais_gravidade": gravidade or [],
        "exames_pendentes": pendentes or [],
        "desfecho_do_modelo": rotulo_llm,
    }


class TestRegraDeDecisao:
    def test_gravidade_produz_alerta(self):
        r = decidir_desfecho(estado(gravidade=["Hipotensão (PAS 85 mmHg)"]))
        assert r["desfecho"] == "EMITIR_ALERTA"
        assert "gravidade" in r["motivo_desfecho"]

    def test_exames_pendentes_produzem_verificacao(self):
        r = decidir_desfecho(estado(pendentes=[{"nome": "D-dímero"}]))
        assert r["desfecho"] == "VERIFICAR_EXAMES"
        assert "resultado" in r["motivo_desfecho"]

    def test_sem_gravidade_nem_pendencia_produz_conduta(self):
        r = decidir_desfecho(estado())
        assert r["desfecho"] == "SUGERIR_CONDUTA"

    def test_gravidade_tem_precedencia_sobre_pendencia(self):
        """Paciente instável com exames pendentes precisa de alerta imediato.

        O nó de alerta lista os pendentes de todo modo, então nada se perde ao
        priorizar a urgência.
        """
        r = decidir_desfecho(estado(
            gravidade=["Hipoxemia (SpO2 88%)"],
            pendentes=[{"nome": "Gasometria"}],
        ))
        assert r["desfecho"] == "EMITIR_ALERTA"

    def test_consulta_sem_paciente_produz_conduta(self):
        """Sem prontuário não há dados estruturados para decidir."""
        r = decidir_desfecho(estado(paciente=False, pendentes=[{"nome": "X"}]))
        assert r["desfecho"] == "SUGERIR_CONDUTA"
        assert "sem paciente" in r["motivo_desfecho"]

    def test_motivo_e_sempre_preenchido(self):
        """O motivo vai para o log de auditoria; nunca deve sair vazio."""
        for est in (
            estado(),
            estado(gravidade=["x"]),
            estado(pendentes=[{"nome": "y"}]),
            estado(paciente=False),
        ):
            assert decidir_desfecho(est)["motivo_desfecho"]


class TestConcordanciaComOModelo:
    """A regra decide; o rótulo do LLM é registrado para medir concordância."""

    def test_rotulo_ausente_produz_concordancia_indefinida(self):
        r = decidir_desfecho(estado(rotulo_llm=None))
        assert r["concorda_com_modelo"] is None

    def test_rotulo_coincidente_registra_concordancia(self):
        r = decidir_desfecho(estado(rotulo_llm="SUGERIR_CONDUTA"))
        assert r["desfecho"] == "SUGERIR_CONDUTA"
        assert r["concorda_com_modelo"] is True

    def test_rotulo_divergente_registra_discordancia(self):
        """Caso observado em execução real: o modelo classificou AVC em janela
        terapêutica como conduta de rotina."""
        r = decidir_desfecho(estado(
            gravidade=["Rebaixamento do nível de consciência (Glasgow 13)"],
            rotulo_llm="SUGERIR_CONDUTA",
        ))
        assert r["desfecho"] == "EMITIR_ALERTA"
        assert r["concorda_com_modelo"] is False

    def test_rotulo_do_modelo_nao_altera_a_decisao(self):
        """Qualquer rótulo produz a mesma decisão, para o mesmo prontuário."""
        base = estado(pendentes=[{"nome": "Hemocultura"}])
        decisoes = {
            decidir_desfecho({**base, "desfecho_do_modelo": r})["desfecho"]
            for r in (None, "SUGERIR_CONDUTA", "EMITIR_ALERTA", "VERIFICAR_EXAMES")
        }
        assert decisoes == {"VERIFICAR_EXAMES"}


class TestRoteamento:
    """`rotear` é despacho puro: traduz rótulo em nome de nó."""

    @pytest.mark.parametrize("desfecho,esperado", [
        ("VERIFICAR_EXAMES", "verificar_exames"),
        ("SUGERIR_CONDUTA", "sugerir_conduta"),
        ("EMITIR_ALERTA", "emitir_alerta"),
    ])
    def test_traduz_cada_desfecho(self, desfecho, esperado):
        assert rotear({"desfecho": desfecho}) == esperado

    @pytest.mark.parametrize("entrada", ["", "INVENTADO", None])
    def test_desfecho_desconhecido_cai_no_conservador(self, entrada):
        """Fallback aponta para o caminho que sempre exige validação humana."""
        assert rotear({"desfecho": entrada}) == "sugerir_conduta"

    def test_estado_sem_desfecho_cai_no_conservador(self):
        assert rotear({}) == "sugerir_conduta"

    def test_fallback_e_o_configurado(self):
        assert config.DESFECHO_PADRAO == "SUGERIR_CONDUTA"
