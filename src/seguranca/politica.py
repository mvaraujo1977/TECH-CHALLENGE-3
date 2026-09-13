"""Política de risco clínico — guardrail de entrada.

O requisito 3 do desafio pede "definir limites de atuação do assistente para
evitar sugestões impróprias (ex.: nunca prescrever diretamente, sem validação
humana)".

O sistema já garantia isso na **saída**, anexando a ressalva de validação ao
texto gerado. Mas validar apenas a saída deixa uma lacuna: uma solicitação que
pede explicitamente para ignorar a validação médica era processada
normalmente, e recebia resposta com a ressalva no fim. O pedido impróprio
chegava ao modelo.

Este módulo classifica a **entrada**, antes de qualquer consulta ao modelo ou
recuperação de protocolo. A classificação é determinística, versionada e
testável — não depende do LLM avaliar a si mesmo.

## Categorias

Em ordem crescente de restrição:

``INFORMATIVO``     consulta sobre protocolo, processo ou estrutura de documento.
``DADOS_PACIENTE``  envolve dados de um paciente concreto; a resposta exige
                    fontes e marcação de conteúdo assistivo.
``CONDUTA_CLINICA`` pede conduta, dose, prescrição, alta ou diagnóstico
                    definitivo; a resposta é rascunho para validação humana.
``BLOQUEADO``       tenta contornar a validação médica, obter prescrição
                    autônoma, falsificar documento, ou subverter as instruções
                    do sistema.

Precedência: ``BLOQUEADO > CONDUTA_CLINICA > DADOS_PACIENTE > INFORMATIVO``.

Em `BLOQUEADO`, o fluxo é interrompido: o modelo não é consultado e nenhum
protocolo é recuperado. A recusa é gerada por código, com o motivo registrado
em auditoria.

## Por que regras e não um classificador

Três razões. A decisão precisa ser auditável — um médico ou auditor tem de
poder ver qual regra disparou. O comportamento precisa ser estável entre
execuções, e a medição do próprio modelo mostrou variação (ver
`docs/analise_e_limitacoes.md`). E o custo de errar é assimétrico: classificar
um pedido de prescrição como informativo é muito mais grave que o inverso.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

VERSAO_POLITICA = "1.1.0"


class Risco(str, Enum):
    """Categorias de risco, em ordem crescente de restrição."""

    INFORMATIVO = "INFORMATIVO"
    DADOS_PACIENTE = "DADOS_PACIENTE"
    CONDUTA_CLINICA = "CONDUTA_CLINICA"
    BLOQUEADO = "BLOQUEADO"


_SEVERIDADE = {
    Risco.INFORMATIVO: 0,
    Risco.DADOS_PACIENTE: 1,
    Risco.CONDUTA_CLINICA: 2,
    Risco.BLOQUEADO: 3,
}


@dataclass(frozen=True)
class Regra:
    """Regra determinística de classificação.

    `padrao` é compilado com `re.IGNORECASE`. `motivo` é o que aparece no log
    de auditoria e na mensagem de recusa — deve ser legível por quem audita,
    não só por quem programou.
    """

    codigo: str
    categoria: Risco
    padrao: str
    motivo: str

    def casa(self, texto: str) -> bool:
        return bool(re.search(self.padrao, texto, re.IGNORECASE))


# --- Regras de bloqueio -----------------------------------------------------
#
# Tentativas de contornar a validação médica. São as únicas que interrompem o
# fluxo, então os padrões são deliberadamente específicos: um falso positivo
# aqui nega atendimento a uma consulta legítima.

REGRAS_BLOQUEIO = (
    Regra(
        "BLQ-001", Risco.BLOQUEADO,
        r"\b(ignor\w+|desconsider\w+|esque[çc]\w+)\b[^.]{0,40}"
        r"\b(instru[çc]\w+|regra|orienta[çc]\w+|prompt|sistema|anterior\w*)\b",
        "tentativa de subverter as instruções do sistema",
    ),
    Regra(
        "BLQ-002", Risco.BLOQUEADO,
        # Duas formas: "sem validação médica" e "não precisa de aprovação
        # humana". A segunda apareceu no holdout e escapava do padrão original.
        r"\b(sem|n[ãa]o\s+precis\w+\s+d[eo]?|dispens\w+)\b[^.]{0,30}"
        r"\b(valida[çc][ãa]o|aprova[çc][ãa]o|revis[ãa]o|confirma[çc][ãa]o|aval)\b"
        r"[^.]{0,30}\b(m[ée]dic\w+|human\w+|profissional|cl[íi]nic\w+)\b|"
        r"\b(valida[çc][ãa]o|aprova[çc][ãa]o)\s+(m[ée]dic\w+|human\w+)\b[^.]{0,25}"
        r"\b(n[ãa]o|dispens\w+|desnecess\w+)\b",
        "pedido explícito de dispensa da validação médica",
    ),
    Regra(
        "BLQ-003", Risco.BLOQUEADO,
        r"\b(n[ãa]o|sem)\b[^.]{0,30}\b(precis\w+|necess\w+|mostr\w+|exib\w+|inclu\w+)\b"
        r"[^.]{0,30}\b(ressalva|aviso|disclaimer|advert[êe]ncia)\b",
        "pedido de omissão da ressalva de validação",
    ),
    Regra(
        "BLQ-004", Risco.BLOQUEADO,
        r"\b(prescrev\w+|receit\w+)\b[^.]{0,40}\b(direto|diretamente|sozinho|"
        r"por\s+conta|sem\s+m[ée]dico)\b",
        "pedido de prescrição autônoma, sem médico responsável",
    ),
    Regra(
        "BLQ-005", Risco.BLOQUEADO,
        # A lista de profissionais cobre especialidades, não apenas "médico":
        # "pretenda ser o cardiologista de plantão" escapava do padrão original.
        # `ser` e `sendo` entram porque "pretenda ser X" não usa "que é" nem
        # "como".
        r"\b(finj\w+|pretend\w+|assum\w+|aja|atue|se\s+compor\w+|"
        r"simul\w+|comport\w+)\b[^.]{0,45}"
        r"\b(m[ée]dic\w+|CRM|doutor\w*|"
        r"\w*(ologist\w+|iatra|cirurgi[ãa]\w*|plantonista))\b",
        "solicitação de que o assistente atue como profissional médico",
    ),
    Regra(
        "BLQ-006", Risco.BLOQUEADO,
        # `falsifi\w+` e não `falsific\w+`: a conjugação muda a grafia
        # ("falsifique"), e o padrão original não casava com o imperativo.
        r"\b(falsifi\w+|fraud\w+|adulter\w+|forj\w+)\b|"
        r"\b(assin\w+)\b[^.]{0,30}\b(por|no\s+lugar\s+d\w+)\b[^.]{0,20}\bm[ée]dic\w+",
        "tentativa de falsificação ou assinatura em nome de médico",
    ),
    Regra(
        "BLQ-007", Risco.BLOQUEADO,
        r"\b(atestado|laudo|receita)\b[^.]{0,40}\b(sem\s+consulta|sem\s+exam\w+|"
        r"para\s+(eu\s+)?(faltar|justificar))\b",
        "pedido de documento sem fundamento clínico",
    ),
)


# --- Regras de conduta clínica ----------------------------------------------
#
# A resposta é produzida como rascunho e roteada para validação humana. Não
# interrompem o fluxo — é o caso de uso legítimo principal do assistente.

REGRAS_CONDUTA = (
    Regra(
        "CON-001", Risco.CONDUTA_CLINICA,
        # "volume de cristaloide devo infundir" veio do holdout: é dose sem a
        # palavra dose. O padrão cobre quantidade + verbo de administração.
        r"\b(quant\w+\s+(mg|ml|mcg|unidades|comprimidos)|"
        r"qual\s+\w{0,12}\s*dose|que\s+dose|dose\s+(de|inicial|m[áa]xima|certa)|"
        r"posologia|quantos?\s+(miligramas|comprimidos))\b|"
        r"\b(quant\w+|qual)\b[^.]{0,30}\b(devo|posso|deve)\b[^.]{0,15}"
        r"\b(administr\w+|infund\w+|dar|aplicar|prescrev\w+|us\w+)\b",
        "solicitação de dose ou posologia",
    ),
    Regra(
        "CON-002", Risco.CONDUTA_CLINICA,
        # `receita` e `prescrição` também aparecem em perguntas informativas
        # sobre a estrutura do documento ("que campos são obrigatórios numa
        # receita?"). O padrão exige **intenção de ação** — verbo de emissão ou
        # primeira pessoa — para não classificar essas como conduta.
        r"\bprescrev\w+\b|"
        # `emit\w+` casava com "é emitida" — voz passiva, pergunta informativa
        # sobre o documento. Só formas ativas indicam intenção de emitir.
        r"\b(devo|posso|pode|quero|preciso|me\s+d[êe]|fa[çc]a|"
        r"emit[ao]|emita|emitir|escrev\w+|gere?|gerar)\b"
        r"[^.]{0,35}\b(prescri[çc]\w+|receit\w+|medica[çm]\w+)\b|"
        r"\b(prescri[çc]\w+|receit\w+)\b[^.]{0,20}\b(para\s+(este|esse|o)\s+paciente|agora)\b",
        "solicitação de prescrição",
    ),
    Regra(
        "CON-003", Risco.CONDUTA_CLINICA,
        r"\b(qual|que|defin\w+|indic\w+)\b[^.]{0,25}\b(conduta|tratamento|terapia|"
        r"manejo)\b",
        "solicitação de conduta ou tratamento",
    ),
    Regra(
        "CON-004", Risco.CONDUTA_CLINICA,
        r"\b(posso|devo|pode)\b[^.]{0,30}\b(dar\s+alta|liberar|encaminhar\s+para\s+casa|"
        r"suspender|interromper|iniciar|administrar)\b",
        "decisão de alta, suspensão ou início de medicação",
    ),
    Regra(
        "CON-005", Risco.CONDUTA_CLINICA,
        r"\b(diagn[óo]stic\w+)\b[^.]{0,25}\b(definitiv\w+|[ée]|do\s+paciente|confirm\w+)\b|"
        r"\b(o\s+que\s+(ele|ela|o\s+paciente)\s+tem)\b",
        "solicitação de diagnóstico definitivo",
    ),
)


# --- Regras de dados de paciente --------------------------------------------

REGRAS_PACIENTE = (
    Regra(
        "PAC-001", Risco.DADOS_PACIENTE,
        r"\b(este|esse|meu|deste|desse|do)\s+paciente\b|\bPAC-\d+\b|"
        r"\b(leito|prontu[áa]rio)\s+\w+",
        "referência a paciente concreto",
    ),
    Regra(
        "PAC-002", Risco.DADOS_PACIENTE,
        r"\b(paciente\s+de\s+\d+\s+anos|\d+\s+anos,?\s+(sexo\s+)?(masculino|feminino|M|F)\b)",
        "dados demográficos de paciente no enunciado",
    ),
    Regra(
        "PAC-003", Risco.DADOS_PACIENTE,
        r"\b(PA|press[ãa]o)\s*:?\s*\d{2,3}\s*[/x]\s*\d{2,3}|"
        r"\b(FC|FR|SpO2|Glasgow|glicemia|lactato|troponina)\s*:?\s*\d",
        "sinais vitais ou resultados de exame no enunciado",
    ),
)


REGRAS = REGRAS_BLOQUEIO + REGRAS_CONDUTA + REGRAS_PACIENTE


@dataclass
class Avaliacao:
    """Resultado da classificação de uma solicitação."""

    categoria: Risco
    regras_acionadas: list[str] = field(default_factory=list)
    motivos: list[str] = field(default_factory=list)
    versao_politica: str = VERSAO_POLITICA

    @property
    def bloqueado(self) -> bool:
        return self.categoria is Risco.BLOQUEADO

    @property
    def exige_validacao(self) -> bool:
        return _SEVERIDADE[self.categoria] >= _SEVERIDADE[Risco.CONDUTA_CLINICA]

    def resumo(self) -> str:
        if not self.regras_acionadas:
            return f"{self.categoria.value} (nenhuma regra acionada)"
        return f"{self.categoria.value} ({', '.join(self.regras_acionadas)})"


def classificar(pergunta: str, dados_paciente: str = "") -> Avaliacao:
    """Classifica a solicitação segundo a política de risco.

    `dados_paciente` é considerado porque a presença de um prontuário no
    contexto eleva o risco por si só: a resposta deixa de ser informativa e
    passa a se referir a um caso concreto.

    Todas as regras são avaliadas, não apenas a primeira que casa — o registro
    de auditoria lista todas as acionadas, o que permite revisar a política
    depois. A categoria final é a de maior severidade.
    """
    texto = f"{pergunta}\n{dados_paciente}".strip()

    acionadas = [regra for regra in REGRAS if regra.casa(texto)]

    if not acionadas:
        # Sem regra acionada, a consulta é informativa — salvo se houver
        # prontuário no contexto, o que a torna específica a um paciente.
        categoria = Risco.DADOS_PACIENTE if dados_paciente.strip() else Risco.INFORMATIVO
        return Avaliacao(categoria=categoria)

    categoria = max(
        (regra.categoria for regra in acionadas),
        key=lambda c: _SEVERIDADE[c],
    )

    # Em bloqueio, só as regras de bloqueio importam para o registro: as demais
    # seriam ruído na justificativa da recusa.
    relevantes = (
        [r for r in acionadas if r.categoria is Risco.BLOQUEADO]
        if categoria is Risco.BLOQUEADO
        else acionadas
    )

    return Avaliacao(
        categoria=categoria,
        regras_acionadas=[r.codigo for r in relevantes],
        motivos=[r.motivo for r in relevantes],
    )


MENSAGEM_RECUSA = (
    "Não posso atender a esta solicitação.\n\n"
    "O assistente é uma ferramenta de apoio à decisão e opera sob supervisão "
    "médica. Não emite prescrições de forma autônoma, não dispensa a validação "
    "do médico responsável e não produz documentos sem fundamento clínico.\n\n"
    "Se a intenção era consultar um protocolo institucional ou verificar dados "
    "de um paciente, reformule a pergunta nesses termos."
)


def montar_recusa(avaliacao: Avaliacao) -> str:
    """Texto de recusa, com o motivo registrado.

    O motivo é explicitado deliberadamente: uma recusa opaca leva o usuário a
    tentar variações até passar, enquanto uma recusa fundamentada sinaliza qual
    é o limite do sistema.
    """
    if not avaliacao.motivos:
        return MENSAGEM_RECUSA

    linhas = [MENSAGEM_RECUSA, "", "**Motivo registrado:**"]
    linhas.extend(f"- {motivo}" for motivo in dict.fromkeys(avaliacao.motivos))
    return "\n".join(linhas)
