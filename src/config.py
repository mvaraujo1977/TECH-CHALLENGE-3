"""Configuração central do assistente.

Todos os caminhos são resolvidos a partir da raiz do projeto, então o código
funciona igual rodando local (`uv run`) ou no Colab (após `git clone`).
"""

from pathlib import Path

# --- Caminhos ---------------------------------------------------------------

RAIZ = Path(__file__).resolve().parent.parent

DIR_DADOS = RAIZ / "data"
DIR_PROTOCOLOS = DIR_DADOS / "protocolos"
ARQUIVO_PRONTUARIOS = DIR_DADOS / "prontuarios.json"

DIR_VECTORSTORE = RAIZ / ".chroma"
DIR_LOGS = RAIZ / "logs"

# --- Modelo -----------------------------------------------------------------

MODELO_BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_LORA = "mvaraujo1977/assistente-medico-lora"

# Mesmo prompt usado no fine-tuning. Alterá-lo degrada o comportamento
# aprendido (citação de fonte, guardrail, rótulo de desfecho).
SYSTEM_PROMPT = (
    "Você é um assistente clínico de apoio à decisão do hospital. "
    "Baseie-se nos protocolos internos e cite sempre a fonte. "
    "Nunca prescreva diretamente: toda sugestão de conduta requer validação do médico responsável."
)

MAX_NEW_TOKENS = 400
TEMPERATURA = 0.3

# --- RAG --------------------------------------------------------------------

MODELO_EMBEDDINGS = "BAAI/bge-m3"

# Protocolos são documentos curtos e estruturados por seção. Chunks grandes
# preservam a seção inteira; a sobreposição evita cortar uma lista no meio.
TAMANHO_CHUNK = 800
SOBREPOSICAO_CHUNK = 150

# Quantos trechos recuperar por consulta.
TOP_K = 4

NOME_COLECAO = "protocolos_hospital"

# --- Decisão ----------------------------------------------------------------

ROTULOS_DESFECHO = ("VERIFICAR_EXAMES", "SUGERIR_CONDUTA", "EMITIR_ALERTA")

# Fallback quando o modelo não emite rótulo reconhecível. Aponta para o caminho
# que sempre exige validação humana — se o parsing falhar, o sistema degrada
# para o comportamento mais conservador, nunca para um que dispense revisão.
DESFECHO_PADRAO = "SUGERIR_CONDUTA"

# Frases que caracterizam a ressalva de validação humana. Usadas para verificar
# se a resposta final atende ao requisito de segurança.
TERMOS_GUARDRAIL = (
    "validação do médico",
    "validação humana",
    "médico responsável",
    "requer validação",
    "apoio à decisão",
    "decisão clínica explícita",
    "não emite conduta",
    "não deve ser iniciada sem",
    "sem confirmação",
    "conduta definitiva",
    "validação prévia",
    "confirmação médica",
)

GUARDRAIL_PADRAO = (
    "Esta orientação é um apoio à decisão e requer validação do médico "
    "responsável antes de qualquer conduta."
)
