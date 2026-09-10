"""Carregamento e inferência do modelo fine-tuned.

O mesmo código roda em CPU (desenvolvimento local) e em GPU (Colab), decidindo
em tempo de execução se usa quantização 4-bit. Também absorve a mudança de
`torch_dtype` para `dtype` introduzida no transformers 5.x.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any

from src import config


# --- Detecção de ambiente ---------------------------------------------------

def tem_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _nome_parametro_dtype() -> str:
    """Devolve `dtype` ou `torch_dtype`, conforme a versão do transformers.

    Em transformers 5.x, `torch_dtype` está deprecado em favor de `dtype`, mas
    a assinatura é `(*model_args, **kwargs)` — introspecção não resolve. Por
    isso a decisão é pela versão da biblioteca, não pela assinatura.
    """
    try:
        from importlib.metadata import version
        major = int(version("transformers").split(".")[0])
        return "dtype" if major >= 5 else "torch_dtype"
    except Exception:  # noqa: BLE001
        return "dtype"


def _suporta_bfloat16() -> bool:
    """bfloat16 exige Ampere ou superior. Em T4 (Turing) causa erro."""
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:  # noqa: BLE001
        return False


# --- Carregamento -----------------------------------------------------------

@dataclass
class ModeloClinico:
    """Modelo carregado, pronto para inferência."""

    modelo: Any
    tokenizer: Any
    dispositivo: str
    quantizado: bool
    nome_base: str
    adapter: str | None

    def descrever(self) -> str:
        partes = [f"{self.nome_base} em {self.dispositivo}"]
        if self.adapter:
            partes.append(f"adapter {self.adapter}")
        if self.quantizado:
            partes.append("4-bit")
        return " | ".join(partes)


def carregar_modelo(
    nome_base: str | None = None,
    adapter: str | None = None,
    forcar_cpu: bool = False,
    dtype_cpu: str | None = None,
) -> ModeloClinico:
    """Carrega o modelo base e aplica os adapters LoRA.

    Passe `adapter=""` para carregar apenas o modelo base — útil para comparar
    o comportamento antes e depois do fine-tuning.

    `dtype_cpu` (ex.: "bfloat16") sobrescreve o float32 padrão da CPU. Serve
    para caber em máquina com pouca RAM; ver a justificativa no corpo.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    nome_base = nome_base or config.MODELO_BASE
    adapter = config.ADAPTER_LORA if adapter is None else adapter

    usar_gpu = tem_gpu() and not forcar_cpu
    chave_dtype = _nome_parametro_dtype()

    kwargs: dict[str, Any] = {}

    if usar_gpu:
        kwargs["device_map"] = "auto"
        kwargs[chave_dtype] = torch.bfloat16 if _suporta_bfloat16() else torch.float16

        # Quantização 4-bit só com bitsandbytes disponível.
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=kwargs[chave_dtype],
            )
            quantizado = True
        except ImportError:
            quantizado = False
    else:
        # float32 em CPU: mais lento que float16, porém estável. Muitas
        # operações em half precision não têm kernel de CPU.
        #
        # `dtype_cpu` abre exceção a essa regra por um motivo de capacidade,
        # não de desempenho: o 3B em float32 ocupa ~12 GB residentes, o que
        # não cabe em 16 GB de RAM. Em bfloat16 cai para ~6 GB. É concessão
        # deliberada, por isso fica fora do padrão e explícita na chamada.
        if dtype_cpu:
            if not hasattr(torch, dtype_cpu):
                raise ValueError(f"dtype desconhecido: {dtype_cpu}")
            kwargs[chave_dtype] = getattr(torch, dtype_cpu)
        else:
            kwargs[chave_dtype] = torch.float32
        quantizado = False

    tokenizer = AutoTokenizer.from_pretrained(adapter or nome_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    modelo = AutoModelForCausalLM.from_pretrained(nome_base, **kwargs)

    if adapter:
        modelo = PeftModel.from_pretrained(modelo, adapter)

    if not usar_gpu:
        # O dtype vai junto com o device: o PeftModel carrega os pesos LoRA no
        # dtype em que foram salvos, e divergência em relação à base quebra o
        # forward. O cast aqui alinha base e adapter depois da fusão.
        modelo = modelo.to("cpu", dtype=kwargs[chave_dtype])

    modelo.eval()

    return ModeloClinico(
        modelo=modelo,
        tokenizer=tokenizer,
        dispositivo="cuda" if usar_gpu else "cpu",
        quantizado=quantizado,
        nome_base=nome_base,
        adapter=adapter or None,
    )


# --- Inferência -------------------------------------------------------------

def gerar(
    mc: ModeloClinico,
    pergunta: str,
    contexto: str = "",
    system_prompt: str | None = None,
    max_new_tokens: int | None = None,
    temperatura: float | None = None,
) -> str:
    """Gera uma resposta.

    Quando há `contexto` (trechos recuperados pelo RAG), ele é inserido antes
    da pergunta com instrução explícita de se basear nele. Isso é o que faz o
    modelo citar o protocolo recuperado em vez do código que memorizou no
    fine-tuning.
    """
    import torch

    system_prompt = system_prompt or config.SYSTEM_PROMPT
    max_new_tokens = max_new_tokens or config.MAX_NEW_TOKENS
    temperatura = config.TEMPERATURA if temperatura is None else temperatura

    if contexto:
        conteudo = (
            "Protocolos internos relevantes:\n\n"
            f"{contexto}\n\n"
            "---\n\n"
            "Baseie sua resposta nos protocolos acima e cite o código do "
            "protocolo utilizado.\n\n"
            f"{pergunta}"
        )
    else:
        conteudo = pergunta

    mensagens = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": conteudo},
    ]

    prompt = mc.tokenizer.apply_chat_template(
        mensagens, tokenize=False, add_generation_prompt=True
    )
    entradas = mc.tokenizer(prompt, return_tensors="pt").to(mc.modelo.device)

    parametros: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": mc.tokenizer.eos_token_id,
    }
    if temperatura and temperatura > 0:
        parametros.update(do_sample=True, temperature=temperatura, top_p=0.9)
    else:
        parametros["do_sample"] = False

    with torch.no_grad():
        saida = mc.modelo.generate(**entradas, **parametros)

    gerado = saida[0][entradas["input_ids"].shape[1]:]
    return mc.tokenizer.decode(gerado, skip_special_tokens=True).strip()


# --- Pós-processamento ------------------------------------------------------

_PADRAO_DESFECHO = re.compile(
    r"^\s*DESFECHO:\s*(" + "|".join(config.ROTULOS_DESFECHO) + r")",
    re.IGNORECASE,
)


def extrair_desfecho(resposta: str) -> str:
    """Lê o rótulo de decisão da primeira linha.

    Se o modelo não emitir rótulo reconhecível, devolve `DESFECHO_PADRAO`, que
    aponta para o caminho que sempre exige validação humana. Falha de parsing
    degrada para o comportamento mais conservador, nunca para um que dispense
    revisão médica.
    """
    correspondencia = _PADRAO_DESFECHO.match(resposta.strip())
    if correspondencia:
        return correspondencia.group(1).upper()
    return config.DESFECHO_PADRAO


def tem_guardrail(resposta: str) -> bool:
    baixa = resposta.lower()
    return any(termo in baixa for termo in config.TERMOS_GUARDRAIL)


def garantir_guardrail(resposta: str) -> tuple[str, bool]:
    """Acrescenta a ressalva de validação se ela estiver ausente.

    Segunda camada de segurança: o fine-tuning ensinou o modelo a incluí-la,
    mas nenhum modelo é determinístico. Aqui a presença é garantida por código.

    Retorna a resposta e um booleano indicando se houve intervenção — o log
    registra esse dado, que revela quantas vezes o modelo falhou sozinho.
    """
    if tem_guardrail(resposta):
        return resposta, False
    return f"{resposta.rstrip()}\n\n{config.GUARDRAIL_PADRAO}", True


def citar_fontes(resposta: str, fontes: list[dict]) -> str:
    """Anexa a lista de protocolos efetivamente recuperados.

    A citação vem dos documentos do retriever, não do código que o modelo
    escreveu no texto. O modelo aprendeu o *formato* de citar durante o
    fine-tuning, mas os códigos que ele produz não são confiáveis — o mesmo
    código foi associado a temas diferentes no dataset de treino.
    """
    if not fontes:
        return resposta

    linhas = ["", "---", "**Fontes consultadas:**"]
    for fonte in fontes:
        versao = f" (v{fonte['versao']})" if fonte.get("versao") else ""
        linhas.append(f"- {fonte['codigo']}{versao} — {fonte['titulo']}")

    return resposta.rstrip() + "\n" + "\n".join(linhas)
