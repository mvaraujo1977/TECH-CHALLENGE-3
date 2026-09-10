"""Testes de integração do grafo completo.

Exercitam o pipeline de ponta a ponta com dublês no lugar do LLM e dos
embeddings. Cobrem o roteamento, o registro de auditoria e o comportamento em
face das falhas de modelo observadas em execução real.
"""

from __future__ import annotations

import json

import pytest

from src import config
from src.graph.fluxo import construir_grafo


class TestEstruturaDoGrafo:
    def test_compila(self, retriever, gerador):
        assert construir_grafo(retriever, gerador) is not None

    def test_decisao_precede_o_roteamento(self, retriever, gerador):
        """O roteamento condicional sai de `decidir_desfecho`, não de
        `consultar_modelo` — é o que garante que a decisão seja da regra."""
        grafo = construir_grafo(retriever, gerador)
        mermaid = grafo.get_graph().draw_mermaid()

        assert "consultar_modelo --> decidir_desfecho" in mermaid
        for no in ("verificar_exames", "sugerir_conduta", "emitir_alerta"):
            assert f"decidir_desfecho -.-> {no}" in mermaid

    def test_todos_os_desfechos_convergem_para_finalizar(self, retriever, gerador):
        mermaid = construir_grafo(retriever, gerador).get_graph().draw_mermaid()
        for no in ("verificar_exames", "sugerir_conduta", "emitir_alerta"):
            assert f"{no} --> finalizar" in mermaid


class TestRoteamentoPontaAPonta:
    @pytest.mark.parametrize("id_paciente,esperado", [
        ("PAC-001", "VERIFICAR_EXAMES"),   # exames pendentes, sem gravidade
        ("PAC-002", "EMITIR_ALERTA"),      # IAMCSST: ECG + troponina
        ("PAC-003", "SUGERIR_CONDUTA"),    # estável, exames completos
        ("PAC-008", "EMITIR_ALERTA"),      # sepse: 8 sinais de gravidade
    ])
    def test_cada_paciente_segue_o_caminho_esperado(self, assistente, id_paciente, esperado):
        _, registro = assistente.consultar("Qual a conduta?", id_paciente)

        assert registro.desfecho == esperado
        no_de_decisao = esperado.lower().replace("emitir_alerta", "emitir_alerta")
        assert registro.caminho_no_grafo[-2] in (
            "verificar_exames", "sugerir_conduta", "emitir_alerta"
        )

    def test_caminho_completo_e_registrado(self, assistente):
        _, registro = assistente.consultar("Qual a conduta?", "PAC-001")

        assert registro.caminho_no_grafo == [
            "carregar_paciente",
            "recuperar_protocolos",
            "consultar_modelo",
            "decidir_desfecho",
            "verificar_exames",
            "finalizar",
        ]

    def test_consulta_sem_paciente_funciona(self, assistente):
        resposta, registro = assistente.consultar("Quais exames no pré-operatório?")

        assert registro.desfecho == "SUGERIR_CONDUTA"
        assert registro.id_paciente is None
        assert resposta


class TestResilienciaAsFalhasDoModelo:
    """Todos os casos abaixo foram observados em execução real do modelo.

    Em nenhum deles a falha do LLM deve alterar a decisão de fluxo.
    """

    @pytest.mark.parametrize("resposta_do_modelo,descricao", [
        ("DESFECHO: SUGERIR CONDUÇÃO\nTexto.", "rótulo corrompido (bfloat16/CPU)"),
        ("DESFECHO: VERIFICAR CONTA\nTexto.", "rótulo inventado (4-bit)"),
        ("DESFECHO: AVALIAR\nTexto.", "rótulo inventado (4-bit)"),
        ("Conforme PROT-007 e PROT-011:", "sem rótulo"),
        ("DESFECHO: SUGERIR_CONDUTA\nTexto.", "rótulo válido, clinicamente errado"),
    ])
    def test_sepse_grave_sempre_gera_alerta(self, assistente, gerador,
                                            resposta_do_modelo, descricao):
        gerador.resposta = resposta_do_modelo

        _, registro = assistente.consultar("Conduta?", "PAC-008")

        assert registro.desfecho == "EMITIR_ALERTA", descricao
        assert len(registro.sinais_gravidade) >= 6

    def test_paciente_inexistente_registra_erro_sem_quebrar(self, assistente):
        resposta, registro = assistente.consultar("Conduta?", "PAC-999")

        assert registro.erro is not None
        assert "não encontrado" in registro.erro
        assert resposta   # ainda produz saída

    def test_falha_na_geracao_nao_derruba_o_fluxo(self, assistente, gerador):
        gerador.resposta = RuntimeError("modelo indisponível")

        resposta, registro = assistente.consultar("Conduta?", "PAC-001")

        assert registro.erro is not None
        assert "Falha na geração" in registro.erro
        assert config.GUARDRAIL_PADRAO in resposta

    def test_falha_na_recuperacao_nao_derruba_o_fluxo(self, gerador, auditoria):
        from src.graph.fluxo import AssistenteClinico

        class RetrieverQuebrado:
            def invoke(self, consulta, codigos=None):
                raise RuntimeError("índice corrompido")

        assistente = AssistenteClinico(
            retriever=RetrieverQuebrado(),
            gerar=gerador,
            auditoria=auditoria,
            nome_modelo="duble",
        )

        resposta, registro = assistente.consultar("Conduta?", "PAC-001")

        assert registro.erro is not None
        assert registro.fontes == []
        assert resposta


class TestEscopoDeRecuperacao:
    def test_escopo_do_prontuario_e_repassado_ao_retriever(self, assistente, retriever):
        assistente.consultar("Conduta?", "PAC-008")
        assert retriever.ultimo_escopo == ["PROT-001"]

    def test_consulta_sem_paciente_nao_restringe_escopo(self, assistente, retriever):
        assistente.consultar("Quais exames no pré-operatório?")
        assert retriever.ultimo_escopo is None

    def test_consulta_usa_queixa_e_antecedentes_sem_a_pergunta(self, assistente, retriever):
        """A pergunta do usuário é deliberadamente excluída da consulta de
        embedding. Medição: incluí-la piorava o MRR do protocolo correto de
        0.67 para 0.46, e os acertos em primeiro lugar caíam de 4/8 para 2/8 —
        "qual a conduta?" é quase constante entre pacientes e diluía o sinal
        clínico. A pergunta continua indo no prompt do modelo, só não na busca.
        """
        assistente.consultar("Qual a conduta?", "PAC-001")

        assert "Dispneia" in retriever.ultima_consulta
        assert "cirurgia ortopédica" in retriever.ultima_consulta.lower()
        assert "conduta" not in retriever.ultima_consulta.lower()

    def test_contexto_recuperado_chega_ao_modelo(self, assistente, gerador):
        assistente.consultar("Conduta?", "PAC-008")

        assert "PROT-001" in gerador.ultimo_contexto

    def test_dados_do_paciente_chegam_ao_modelo(self, assistente, gerador):
        """Requisito: contextualizar as respostas com informações do paciente."""
        assistente.consultar("Conduta?", "PAC-008")

        assert "Dados do paciente" in gerador.ultima_pergunta
        assert "Sinais vitais" in gerador.ultima_pergunta


class TestAuditoria:
    def test_registro_e_gravado_em_jsonl_valido(self, assistente, auditoria):
        assistente.consultar("Conduta?", "PAC-001")

        linhas = auditoria.arquivo.read_text(encoding="utf-8").strip().splitlines()
        assert len(linhas) == 1
        assert json.loads(linhas[0])["id_paciente"] == "PAC-001"

    def test_campos_essenciais_preenchidos(self, assistente):
        _, registro = assistente.consultar("Conduta?", "PAC-008")

        assert registro.id
        assert registro.momento
        assert registro.desfecho
        assert registro.motivo_desfecho
        assert registro.caminho_no_grafo
        assert registro.duracao_s is not None
        assert registro.modelo

    def test_registros_acumulam_sem_sobrescrever(self, assistente, auditoria):
        """Log de auditoria é append-only."""
        for id_paciente in ("PAC-001", "PAC-002", "PAC-003"):
            assistente.consultar("Conduta?", id_paciente)

        assert len(auditoria.ler()) == 3

    def test_leitura_devolve_mais_recente_primeiro(self, assistente, auditoria):
        assistente.consultar("Conduta?", "PAC-001")
        assistente.consultar("Conduta?", "PAC-002")

        assert auditoria.ler()[0]["id_paciente"] == "PAC-002"

    def test_estatisticas_agregam_as_metricas(self, assistente, auditoria):
        for id_paciente in ("PAC-001", "PAC-003", "PAC-008"):
            assistente.consultar("Conduta?", id_paciente)

        stats = auditoria.estatisticas()

        assert stats["total"] == 3
        assert set(stats["por_desfecho"]) <= set(config.ROTULOS_DESFECHO)
        assert "llm_emitiu_rotulo_valido" in stats
        assert "llm_concordou_com_a_regra" in stats

    def test_estatisticas_de_base_vazia_nao_quebram(self, auditoria):
        assert auditoria.estatisticas() == {"total": 0}

    def test_concordancia_e_registrada(self, assistente, gerador):
        """A discordância entre LLM e regra é o dado que justifica a decisão de
        não delegar o roteamento ao modelo."""
        gerador.resposta = "DESFECHO: SUGERIR_CONDUTA\nTexto."

        _, registro = assistente.consultar("Conduta?", "PAC-008")

        assert registro.desfecho == "EMITIR_ALERTA"
        assert registro.desfecho_do_modelo == "SUGERIR_CONDUTA"
        assert registro.concorda_com_modelo is False
