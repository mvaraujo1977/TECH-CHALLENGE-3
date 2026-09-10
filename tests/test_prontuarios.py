"""Testes da consulta à base estruturada e da detecção de gravidade.

A detecção de gravidade é a segunda camada de segurança: quando o LLM
classifica um caso grave como rotineiro, é ela que sinaliza. Em execução real
isso ocorreu em 2 de 8 casos.
"""

from __future__ import annotations

import pytest

from src.rag import prontuarios as pr
from src.rag.prontuarios import ACHADOS_CRITICOS, _esta_negado


class TestConsultaEstruturada:
    def test_lista_pacientes(self):
        pacientes = pr.listar_pacientes()
        assert len(pacientes) == 8
        assert all("id" in p and "queixa" in p for p in pacientes)

    def test_busca_por_id(self):
        assert pr.buscar_paciente("PAC-001")["id"] == "PAC-001"

    @pytest.mark.parametrize("entrada", ["pac-002", "  PAC-002  ", "Pac-002"])
    def test_busca_tolera_caixa_e_espaco(self, entrada):
        assert pr.buscar_paciente(entrada)["id"] == "PAC-002"

    def test_id_inexistente_devolve_none(self):
        assert pr.buscar_paciente("PAC-999") is None

    def test_separa_exames_por_status(self):
        paciente = pr.buscar_paciente("PAC-001")
        pendentes = pr.exames_pendentes(paciente)
        disponiveis = pr.exames_disponiveis(paciente)

        assert len(pendentes) + len(disponiveis) == len(paciente["exames"])
        assert all(e["resultado"] is None for e in pendentes)
        assert all(e["resultado"] for e in disponiveis)


class TestSerializacaoParaPrompt:
    """O prontuário é serializado no formato que o modelo viu no treino.

    Passar JSON cru degradaria a resposta, porque divergiria do padrão
    aprendido durante o fine-tuning.
    """

    def test_inclui_campos_essenciais(self):
        texto = pr.formatar_para_prompt(pr.buscar_paciente("PAC-008"))

        for esperado in ("anos", "Queixa:", "Sinais vitais:", "Exames"):
            assert esperado in texto

    def test_sinaliza_ausencia_de_pendencia(self):
        """Dizer "nenhum" explicitamente é melhor que omitir a linha."""
        paciente = pr.buscar_paciente("PAC-007")
        assert not pr.exames_pendentes(paciente)
        assert "Exames pendentes: nenhum" in pr.formatar_para_prompt(paciente)

    def test_lista_pendencias_quando_existem(self):
        texto = pr.formatar_para_prompt(pr.buscar_paciente("PAC-001"))
        assert "D-dímero" in texto

    def test_sem_alergia_diz_nenhuma_referida(self):
        texto = pr.formatar_para_prompt(pr.buscar_paciente("PAC-001"))
        assert "nenhuma referida" in texto


class TestGravidadePorSinaisVitais:
    @pytest.mark.parametrize("id_paciente,termo", [
        ("PAC-008", "Hipotensão"),
        ("PAC-007", "Crise hipertensiva"),
        ("PAC-004", "Glasgow"),
    ])
    def test_detecta_alteracoes(self, id_paciente, termo):
        alertas = pr.sinais_de_gravidade(pr.buscar_paciente(id_paciente))
        assert any(termo in a for a in alertas)

    def test_paciente_estavel_sem_alerta(self):
        assert pr.sinais_de_gravidade(pr.buscar_paciente("PAC-003")) == []


class TestGravidadePorExames:
    """Regressão: a heurística original lia apenas sinais vitais, e deixou
    passar um IAMCSST confirmado porque pressão, saturação e frequência
    estavam normais — a gravidade estava no eletrocardiograma."""

    def test_supradesnivelamento_de_st_dispara_alerta(self):
        alertas = pr.sinais_de_gravidade(pr.buscar_paciente("PAC-002"))
        assert any("Supradesnivelamento de ST" in a for a in alertas)

    def test_troponina_elevada_dispara_alerta(self):
        alertas = pr.sinais_de_gravidade(pr.buscar_paciente("PAC-002"))
        assert any("Troponina elevada" in a for a in alertas)

    def test_lactato_elevado_dispara_alerta(self):
        alertas = pr.sinais_de_gravidade(pr.buscar_paciente("PAC-008"))
        assert any("Lactato elevado" in a for a in alertas)

    def test_iamcsst_dispara_por_multiplas_vias(self):
        """Redundância deliberada: ECG, troponina e protocolo associado."""
        alertas = pr.sinais_de_gravidade(pr.buscar_paciente("PAC-002"))
        assert len(alertas) >= 3


class TestGravidadePorProtocolo:
    """Quando a curadoria do prontuário associa o paciente a um protocolo de
    urgência, o caso é tratado como alerta mesmo com sinais vitais normais."""

    @pytest.mark.parametrize("id_paciente,protocolo", [
        ("PAC-002", "PROT-002"),
        ("PAC-004", "PROT-003"),
        ("PAC-005", "PROT-009"),
        ("PAC-008", "PROT-001"),
    ])
    def test_protocolo_de_urgencia_dispara_alerta(self, id_paciente, protocolo):
        alertas = pr.sinais_de_gravidade(pr.buscar_paciente(id_paciente))
        assert any(protocolo in a for a in alertas)

    def test_protocolo_nao_urgente_nao_dispara(self):
        """PAC-003 tem PROT-012 e PROT-011, que não são de urgência."""
        assert pr.sinais_de_gravidade(pr.buscar_paciente("PAC-003")) == []


class TestNegacao:
    """Regressão do falso positivo mais insidioso encontrado.

    A busca por achados em texto livre é feita por substring, e o laudo "Sem
    hemorragia ou isquemia aguda" gerava alerta de achado hemorrágico —
    invertendo o significado do documento.
    """

    def _achou(self, texto: str) -> bool:
        baixo = texto.lower()
        for termo, _ in ACHADOS_CRITICOS:
            posicao = baixo.find(termo)
            if posicao >= 0 and not _esta_negado(baixo, posicao):
                return True
        return False

    @pytest.mark.parametrize("laudo", [
        "Sem hemorragia ou isquemia aguda",
        "Não há hemorragia",
        "Ausência de hemorragia",
        "Negativo para hemorragia",
        "Sem supradesnivelamento de ST",
    ])
    def test_achado_negado_nao_dispara(self, laudo):
        assert not self._achou(laudo)

    @pytest.mark.parametrize("laudo", [
        "Hemorragia subaracnóidea extensa",
        "Supradesnivelamento de ST em DII, DIII, aVF",
        "Bloqueio de ramo esquerdo novo",
    ])
    def test_achado_positivo_dispara(self, laudo):
        assert self._achou(laudo)

    def test_negacao_de_outra_oracao_nao_se_aplica(self):
        """Ponto-e-vírgula encerra a oração: a negação anterior não alcança o
        achado seguinte."""
        assert self._achou("Sem febre; hemorragia subaracnóidea presente")

    def test_laudo_real_do_pac007_nao_gera_alerta_hemorragico(self):
        """O laudo do PAC-007 diz "Sem hemorragia ou isquemia aguda". O único
        alerta legítimo dele é a crise hipertensiva."""
        alertas = pr.sinais_de_gravidade(pr.buscar_paciente("PAC-007"))
        assert not any("hemorrág" in a.lower() for a in alertas)
        assert any("Crise hipertensiva" in a for a in alertas)


class TestCoerenciaDaBase:
    """A base é entregável do projeto; estes testes guardam sua integridade."""

    def test_todo_paciente_tem_campos_obrigatorios(self):
        for info in pr.listar_pacientes():
            paciente = pr.buscar_paciente(info["id"])
            for campo in ("identificacao", "admissao", "exames", "protocolos_relacionados"):
                assert campo in paciente, f"{info['id']} sem {campo}"

    def test_protocolos_referenciados_existem_como_documento(self):
        """Um código no prontuário sem documento correspondente produziria
        escopo de recuperação vazio, sem sinal de erro."""
        from src.rag.documentos import carregar_protocolos

        existentes = {d.metadata["codigo"] for d in carregar_protocolos()}

        for info in pr.listar_pacientes():
            paciente = pr.buscar_paciente(info["id"])
            for codigo in paciente["protocolos_relacionados"]:
                assert codigo in existentes, f"{info['id']} cita {codigo} inexistente"

    def test_os_tres_desfechos_sao_exercitados_pela_base(self):
        """Se a base deixasse de cobrir algum caminho do grafo, ele ficaria sem
        teste de integração."""
        desfechos = set()
        for info in pr.listar_pacientes():
            paciente = pr.buscar_paciente(info["id"])
            if pr.sinais_de_gravidade(paciente):
                desfechos.add("EMITIR_ALERTA")
            elif pr.exames_pendentes(paciente):
                desfechos.add("VERIFICAR_EXAMES")
            else:
                desfechos.add("SUGERIR_CONDUTA")

        assert desfechos == {"EMITIR_ALERTA", "VERIFICAR_EXAMES", "SUGERIR_CONDUTA"}
