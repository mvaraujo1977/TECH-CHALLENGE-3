"""Testes das camadas de segurança.

Cobre o requisito 3 do desafio: limites de atuação (guardrail), rastreabilidade
da fonte (explainability) e o parsing do rótulo de decisão.

Vários casos aqui derivam de defeitos encontrados em execução real, e estão
marcados como tal — servem de regressão.
"""

from __future__ import annotations

import pytest

from src import config
from src.llm.modelo import (
    citar_fontes,
    extrair_desfecho,
    extrair_desfecho_bruto,
    garantir_guardrail,
    tem_guardrail,
)


class TestExtracaoDoDesfecho:
    @pytest.mark.parametrize("texto,esperado", [
        ("DESFECHO: VERIFICAR_EXAMES\nFaltam exames.", "VERIFICAR_EXAMES"),
        ("DESFECHO: SUGERIR_CONDUTA\nOk.", "SUGERIR_CONDUTA"),
        ("DESFECHO: EMITIR_ALERTA\nGrave.", "EMITIR_ALERTA"),
        ("desfecho: sugerir_conduta\ntexto", "SUGERIR_CONDUTA"),
        ("  DESFECHO:EMITIR_ALERTA", "EMITIR_ALERTA"),
    ])
    def test_reconhece_rotulos_validos(self, texto, esperado):
        assert extrair_desfecho_bruto(texto) == esperado

    @pytest.mark.parametrize("texto,motivo", [
        ("DESFECHO: SUGERIR CONDUÇÃO\nTexto.", "corrupção observada em CPU/bfloat16"),
        ("DESFECHO: VERIFICAR", "rótulo truncado, observado em 4-bit"),
        ("DESFECHO: VERIFICAR CONTA", "rótulo inventado, observado em 4-bit"),
        ("DESFECHO: AVALIAR", "rótulo inventado, observado em 4-bit"),
        ("Conforme PROT-001, fazer X.", "sem rótulo"),
        ("", "resposta vazia"),
    ])
    def test_rejeita_rotulos_invalidos(self, texto, motivo):
        """Todos estes casos foram observados em execução real do modelo."""
        assert extrair_desfecho_bruto(texto) is None, motivo

    def test_fallback_e_conservador(self):
        """Falha de parsing degrada para o caminho que exige validação."""
        assert extrair_desfecho("lixo") == config.DESFECHO_PADRAO
        assert extrair_desfecho("") == config.DESFECHO_PADRAO

    def test_distingue_rotulo_valido_de_fallback(self):
        """Regressão: a versão anterior checava só o prefixo `DESFECHO:`, então
        um rótulo corrompido era contado como acerto do modelo, contaminando a
        métrica de auditoria."""
        corrompido = "DESFECHO: SUGERIR CONDUÇÃO\nTexto."
        assert extrair_desfecho_bruto(corrompido) is None
        assert extrair_desfecho(corrompido) == config.DESFECHO_PADRAO


class TestGuardrail:
    def test_detecta_ressalva_padrao(self):
        assert tem_guardrail(f"Fazer X. {config.GUARDRAIL_PADRAO}")

    @pytest.mark.parametrize("texto", [
        "Requer validação humana antes da prescrição.",
        "Esta orientação é um apoio à decisão.",
        "Conduta não deve ser iniciada sem confirmação.",
        "O assistente não emite conduta definitiva.",
    ])
    def test_reconhece_variantes_da_ressalva(self, texto):
        assert tem_guardrail(texto)

    def test_nao_detecta_onde_nao_existe(self):
        assert not tem_guardrail("Administrar dipirona 1g EV agora.")

    def test_insere_quando_ausente(self):
        texto, inserido = garantir_guardrail("Administrar dipirona 1g EV.")
        assert inserido is True
        assert config.GUARDRAIL_PADRAO in texto

    def test_nao_duplica_quando_presente(self):
        original = f"Fazer X. {config.GUARDRAIL_PADRAO}"
        texto, inserido = garantir_guardrail(original)
        assert inserido is False
        assert texto.count("apoio à decisão") == 1


class TestGuardrailNoFluxoCompleto:
    """Regressão do falso positivo mais grave encontrado.

    A verificação era feita sobre o texto final montado, que inclui o bloco de
    ações recomendadas. A ação "Acionar imediatamente o médico responsável"
    contém as mesmas palavras da ressalva de validação, então o detector
    concluía que o aviso estava presente e não o inseria — produzindo respostas
    com sugestão de conduta sem aviso de segurança.

    A correção foi avaliar o guardrail apenas sobre o texto do modelo, isolado
    das ações. Estes testes garantem que a correção não regrida.
    """

    def test_acao_com_medico_responsavel_dispara_falso_positivo(self):
        """Confirma que o texto da ação, isoladamente, casa com os termos."""
        assert tem_guardrail("Acionar imediatamente o médico responsável")

    def test_resposta_sem_ressalva_recebe_guardrail_apesar_das_acoes(self, assistente, gerador):
        gerador.resposta = "Iniciar antibiótico empírico de amplo espectro."

        resposta, registro = assistente.consultar("Conduta?", "PAC-008")

        assert registro.guardrail_adicionado is True
        assert config.GUARDRAIL_PADRAO in resposta
        # o bloco de ações do nó de alerta contém "médico responsável"
        assert "Acionar imediatamente o médico responsável" in resposta

    def test_resposta_com_ressalva_nao_recebe_duplicata(self, assistente, gerador):
        gerador.resposta = f"Iniciar antibiótico. {config.GUARDRAIL_PADRAO}"

        resposta, registro = assistente.consultar("Conduta?", "PAC-008")

        assert registro.guardrail_adicionado is False
        assert resposta.count("apoio à decisão") == 1


class TestCitacaoDeFontes:
    def test_anexa_fontes_com_versao(self):
        fontes = [
            {"codigo": "PROT-007", "titulo": "Tromboembolismo Pulmonar", "versao": "4"},
            {"codigo": "PROT-011", "titulo": "Tromboprofilaxia", "versao": ""},
        ]
        texto = citar_fontes("Resposta.", fontes)

        assert "Fontes consultadas" in texto
        assert "PROT-007 (v4)" in texto
        assert "PROT-011 —" in texto   # sem versão, sem parênteses

    def test_sem_fontes_nao_altera_o_texto(self):
        assert citar_fontes("Resposta.", []) == "Resposta."

    def test_fonte_vem_do_retriever_nao_do_texto_do_modelo(self, assistente, gerador):
        """Explainability depende do documento recuperado, não do código que o
        modelo escreveu — os códigos que ele produz não são confiáveis."""
        gerador.resposta = "Conforme PROT-999 (inexistente), fazer X."

        resposta, registro = assistente.consultar("Conduta?", "PAC-008")

        assert [f["codigo"] for f in registro.fontes] == ["PROT-001"]
        assert "PROT-001" in resposta.split("Fontes consultadas")[1]
