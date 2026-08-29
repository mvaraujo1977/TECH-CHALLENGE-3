# Tech Challenge — Fase 3

Assistente virtual médico de apoio à decisão clínica, construído com fine-tuning de LLM, RAG via LangChain e fluxos de decisão automatizados com LangGraph.

> ⚠️ **Aviso importante**
> Todos os dados deste repositório são **sintéticos** e foram gerados para fins acadêmicos.
> Protocolos, códigos internos (`PROT-0XX`, `LAUDO-XXX-XX`), doses e condutas são **fictícios**,
> não foram validados por profissionais de saúde e **não devem ser utilizados para decisões clínicas reais**.

---

## Sobre o projeto

O desafio propõe a criação de um assistente médico treinado com dados próprios de um hospital, capaz de auxiliar em condutas clínicas, responder dúvidas de médicos e sugerir procedimentos com base em protocolos internos — com fluxos de decisão automatizados e seguros coordenados via LangChain e LangGraph.

O sistema combina três camadas complementares:

| Camada | O que faz | Onde vive o conhecimento |
|---|---|---|
| **Modelo fine-tuned** | Aprende o *formato* e o *comportamento* esperados: citar fonte, exigir validação humana, emitir rótulo de decisão | Nos pesos (adapters LoRA) |
| **RAG (LangChain)** | Recupera o conteúdo factual dos protocolos no momento da pergunta | Base vetorial de documentos |
| **Fluxo (LangGraph)** | Roteia a decisão entre verificar exames, sugerir conduta ou emitir alerta | Grafo de estados |

Essa separação é intencional: com fine-tuning LoRA sobre um volume moderado de exemplos, o modelo aprende comportamento, não conhecimento clínico. O conteúdo factual vem do RAG.

---

## Estrutura do repositório

```
TECH-CHALLENGE-3/
├── data/
│   └── dataset_medico.jsonl        # dataset sintético de fine-tuning (95 exemplos)
├── notebooks/
│   ├── 01_gerar_dataset.ipynb      # pipeline de geração, anonimização e curadoria
│   └── 02_finetuning.ipynb         # fine-tuning QLoRA
├── docs/
│   └── resultados/
│       ├── curva_loss.png          # curva de perda do treino
│       └── avaliacao_modelo.json   # respostas e métricas do conjunto de avaliação
└── README.md
```

Estrutura prevista conforme o projeto avança:

```
├── data/
│   ├── protocolos/                 # base de conhecimento do RAG
│   └── prontuarios.json            # base estruturada consultada pelo LangGraph
├── src/
│   ├── llm/                        # carregamento do modelo fine-tuned
│   ├── rag/                        # pipeline LangChain
│   ├── graph/                      # fluxo LangGraph
│   └── logging/                    # auditoria e rastreabilidade
├── logs/
└── docs/
    ├── relatorio_tecnico.md
    └── diagrama_fluxo.png
```

---

## Notebooks

| Notebook | Finalidade | Requisitos | Precisa rodar? |
|---|---|---|---|
| `01_gerar_dataset.ipynb` | Geração, anonimização e curadoria do dataset sintético | API key Anthropic, CPU | **Não** — o dataset já está versionado em `data/` |
| `02_finetuning.ipynb` | Fine-tuning QLoRA do modelo base | GPU (L4 ou T4), conta Hugging Face | **Não** — os adapters já estão publicados |

Ambos os notebooks são **documentação do processo**. Para *usar* o assistente, basta carregar o modelo publicado — ver [Como usar o modelo](#como-usar-o-modelo).

---

## O dataset

`data/dataset_medico.jsonl` — 95 exemplos no formato de instrução (`instruction` / `input` / `output` / `categoria`).

### Distribuição por categoria

| Categoria | Exemplos | Conteúdo |
|---|---:|---|
| `cenarios_clinicos` | 25 | Casos com dados de paciente que exigem decisão de roteamento |
| `protocolos` | 20 | Protocolos clínicos internos (sepse, TEP, AVC, pré-operatório, cetoacidose) |
| `faq_medicos` | 20 | Dúvidas frequentes sobre condutas e procedimentos internos |
| `laudos` | 15 | Estrutura de laudos (imagem, laboratorial, anatomopatológico) |
| `receitas` | 15 | Modelos de receita e regras de prescrição, incluindo controlados |

### Métricas de qualidade

| Métrica | Resultado |
|---|---|
| Citação de fonte | 95/95 (100%) |
| Guardrail de validação humana | 57/95 (60%) — **100% nos casos que sugerem conduta** |
| Exemplos truncados | 0 |
| Instruções duplicadas | 0 |
| Resíduos de anonimização | 0 |
| Tamanho das respostas | 318–687 caracteres (mediana: 485) |

A cobertura de guardrail é **condicional por design**: a ressalva de validação humana aparece onde há sugestão de conduta clínica, e é omitida em perguntas puramente informativas (estrutura de um laudo, campos obrigatórios de uma receita). Um guardrail presente em 100% das respostas seria ruído — o objetivo é que o modelo aprenda *quando* a validação é necessária, não a repetir a frase por reflexo.

### Distribuição dos desfechos (cenários clínicos)

| Desfecho | Exemplos | Quando ocorre |
|---|---:|---|
| `VERIFICAR_EXAMES` | 9 | Faltam resultados essenciais; o assistente recusa conduta definitiva |
| `EMITIR_ALERTA` | 8 | Há sinal de gravidade; sinaliza urgência à equipe |
| `SUGERIR_CONDUTA` | 8 | Dados suficientes; descreve o que o protocolo prevê, com ressalva |

Todos os 25 cenários iniciam com o rótulo padronizado `DESFECHO: <RÓTULO>`, o que torna o roteamento no LangGraph determinístico.

---

## Preparação dos dados

O pipeline em `01_gerar_dataset.ipynb` executa nesta ordem, que não deve ser alterada:

```
gerar → deduplicar → normalizar desfecho → limpar meta-vazamento → ANONIMIZAR → verificar → salvar
```

**Geração.** Lotes de 5 exemplos por chamada, com prompt de regras explícitas. Lotes pequenos eliminam truncamento por limite de tokens — o problema que inviabilizava lotes maiores.

**Preprocessing.** Parsing de JSONL com descarte de linhas malformadas, deduplicação por instrução, remoção de respostas truncadas, normalização do rótulo de desfecho e limpeza de meta-instruções vazadas para dentro do conteúdo.

**Anonimização em duas camadas.** Restrição no prompt de geração e regex de pós-processamento cobrindo CPF, CNS, datas, iniciais (`R.S.`, `R.M.T.S.`) e nomes próprios, com verificação automática de resíduos.

**Curadoria.** Revisão humana por amostragem em cada categoria, mais métricas automáticas de cobertura de guardrail e citação de fonte.

---

## O modelo treinado

| | |
|---|---|
| **Modelo base** | [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |
| **Adapters LoRA** | [`mvaraujo1977/assistente-medico-lora`](https://huggingface.co/mvaraujo1977/assistente-medico-lora) (público) |
| **Técnica** | QLoRA — quantização 4-bit (NF4) + LoRA |
| **Configuração LoRA** | `r=16`, `alpha=32`, `dropout=0.05` |
| **Módulos treinados** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Épocas** | 4 |
| **Learning rate** | 2e-4, scheduler cosine |
| **Batch efetivo** | 8 (batch 2 × gradient accumulation 4) |
| **Split** | 81 treino / 14 avaliação, estratificado por categoria |

### Por que Qwen2.5-3B e não LLaMA

O enunciado sugere LLaMA ou Falcon, mas admite outros modelos. A escolha pelo Qwen2.5-3B-Instruct foi motivada por:

- **Sem gate de licença.** O `meta-llama/Llama-3.2-3B-Instruct` exige aprovação individual da Meta — a tentativa de uso retornou `403 GatedRepoError`. Isso quebraria a reprodutibilidade do projeto para os demais integrantes do grupo e para a banca avaliadora.
- **Mesma faixa de parâmetros** (3B), cabendo em GPU L4/T4 com quantização 4-bit.
- **Template de chat bem definido**, aplicado via `tokenizer.apply_chat_template()`.

A troca é de uma linha: basta alterar `MODEL_NAME` no notebook `02_finetuning.ipynb`.

### Split estratificado

A divisão treino/avaliação é estratificada **por categoria**, não aleatória. Com 95 exemplos, um split simples poderia deixar o conjunto de avaliação sem nenhum cenário clínico — justamente o comportamento mais importante de medir.

| Categoria | Treino | Avaliação |
|---|---:|---:|
| `cenarios_clinicos` | 21 | 4 |
| `protocolos` | 17 | 3 |
| `faq_medicos` | 17 | 3 |
| `laudos` | 13 | 2 |
| `receitas` | 13 | 2 |

---

## Avaliação do modelo

Curva de perda: [`docs/resultados/curva_loss.png`](docs/resultados/curva_loss.png)
Respostas completas do conjunto de avaliação: [`docs/resultados/avaliacao_modelo.json`](docs/resultados/avaliacao_modelo.json)

### Metodologia

A `eval_loss` indica se o modelo prevê tokens melhor, mas não se aprendeu **o comportamento desejado**. Por isso, a avaliação mede diretamente os três comportamentos-alvo sobre o conjunto de avaliação, comparando as respostas do modelo **antes** e **depois** do fine-tuning:

| Comportamento | Como é verificado |
|---|---|
| Cita fonte | Presença de padrão `PROT-`, `LAUDO-`, `RX-` ou `FARM-` na resposta |
| Aplica guardrail | Presença de alguma das 12 formulações de ressalva de validação humana |
| Emite desfecho | Resposta a cenário clínico inicia com `DESFECHO: <RÓTULO>` |

### Resultados

<!-- PREENCHER com os números da execução do notebook 02 -->

| Métrica | Antes do fine-tuning | Depois do fine-tuning |
|---|---:|---:|
| Cita fonte | — | — |
| Aplica guardrail | — | — |
| Emite `DESFECHO:` | — | — |

**Perda de avaliação:** `eval_loss` inicial — → final —

---

## Como usar o modelo

### Instalação

```bash
pip install transformers peft torch accelerate bitsandbytes
```

Ou, com `uv`:

```bash
uv add transformers peft torch accelerate bitsandbytes
```

### Carregar o modelo

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "mvaraujo1977/assistente-medico-lora"

base = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
model.eval()
```

Nenhum token do Hugging Face é necessário — tanto o modelo base quanto os adapters são públicos.

### Fazer uma pergunta

O modelo foi treinado com um `SYSTEM_PROMPT` específico. Usar o mesmo prompt na inferência é importante: ele faz parte do formato aprendido durante o fine-tuning.

```python
SYSTEM_PROMPT = (
    "Você é um assistente clínico de apoio à decisão do hospital. "
    "Baseie-se nos protocolos internos e cite sempre a fonte. "
    "Nunca prescreva diretamente: toda sugestão de conduta requer validação do médico responsável."
)

def perguntar(pergunta: str, dados_paciente: str = "", max_new_tokens: int = 300) -> str:
    conteudo = pergunta
    if dados_paciente:
        conteudo += "\n\nDados do paciente:\n" + dados_paciente

    mensagens = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": conteudo},
    ]
    prompt = tokenizer.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=True)
    entradas = tokenizer(prompt, return_tensors="pt").to(model.device)

    saida = model.generate(
        **entradas,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        saida[0][entradas["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
```

### Exemplos

Pergunta sobre protocolo:

```python
print(perguntar("Qual o protocolo interno para suspeita de sepse?"))
```

Cenário clínico com dados do paciente:

```python
print(perguntar(
    "Posso definir conduta para este paciente ou preciso de mais dados?",
    dados_paciente=(
        "Paciente de 51 anos, masculino. Dispneia súbita há 2h, dor pleurítica. "
        "FC 110 bpm, SpO2 93%. Cirurgia ortopédica há 12 dias. "
        "D-dímero e angiotomografia ainda não realizados."
    )
))
```

Em cenários clínicos, a resposta começa com um rótulo de decisão:

```
DESFECHO: VERIFICAR_EXAMES
...
```

### Rótulos de desfecho

O modelo emite um destes três rótulos na primeira linha ao receber um cenário clínico. É o que permite o roteamento determinístico no LangGraph.

| Rótulo | Quando ocorre |
|---|---|
| `VERIFICAR_EXAMES` | Faltam resultados essenciais; o assistente recusa conduta definitiva |
| `SUGERIR_CONDUTA` | Dados suficientes; descreve o que o protocolo prevê, com ressalva de validação |
| `EMITIR_ALERTA` | Há sinal de gravidade; sinaliza urgência à equipe |

Extração do rótulo:

```python
import re

def extrair_desfecho(resposta: str) -> str:
    m = re.match(r'^DESFECHO:\s*(\w+)', resposta.strip())
    return m.group(1) if m else "SUGERIR_CONDUTA"   # fallback conservador
```

O fallback aponta para `SUGERIR_CONDUTA`, que sempre carrega o guardrail de validação humana. Se o parsing falhar, o sistema degrada para o caminho que exige revisão médica — nunca para um que a dispense.

### Desempenho esperado

| Ambiente | Tempo por resposta |
|---|---|
| GPU (L4/T4) | 1–3 segundos |
| CPU | 10–30 segundos |

Sem GPU, o modelo funciona mas fica lento. Para a demonstração em vídeo, vale testar o ambiente antes ou usar quantização adicional (GGUF via llama.cpp/Ollama).

---

## Como reproduzir o treino

Não é necessário para usar o modelo — os adapters já estão publicados. Siga apenas se quiser retreinar.

1. Abra `notebooks/02_finetuning.ipynb` no Google Colab
2. Configure GPU: **Ambiente de execução → Alterar tipo de ambiente → L4 GPU** (ou T4)
3. Crie um token de escrita em [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) e cadastre nos Secrets do Colab como `HF_TOKEN`
4. Ajuste `HUB_REPO` na seção 5 para o seu usuário do Hugging Face
5. Execute as células em ordem

**Se estiver numa T4**, altere na seção 10: `bf16=True` → `fp16=True`. A T4 não tem suporte nativo a bfloat16.

O treino leva poucos minutos com 81 exemplos.

### Regerar o dataset (opcional)

O dataset já está versionado em `data/dataset_medico.jsonl`. O notebook `01_gerar_dataset.ipynb` documenta como foi produzido e só precisa ser executado para gerar um conjunto novo — nesse caso, é necessária uma API key da Anthropic cadastrada nos Secrets do Colab como `ANTHROPIC_API_KEY`.

---

## Limitações conhecidas

Declaradas explicitamente por rigor metodológico:

- **Exatidão regulatória não validada.** Protocolos, códigos, listas da Portaria 344 e posologias são inventados e não foram revisados por profissional de saúde.
- **Anonimização por regex tem casos de borda.** Durante o desenvolvimento, um nome com quatro iniciais escapou de um padrão escrito para três — detectado e corrigido por verificação automática. Em produção, ferramentas dedicadas (ex.: Microsoft Presidio) seriam o caminho.
- **Volume dimensiona uma demonstração.** 95 exemplos comprovam o pipeline; não produzem um modelo de produção.
- **O fine-tuning ensina forma, não conteúdo.** Com LoRA neste volume, o modelo aprende a citar fonte, pedir validação e emitir rótulos de decisão. O conhecimento clínico factual vem do RAG.
- **Conjunto de avaliação pequeno.** 14 exemplos indicam tendência, mas não permitem afirmações de significância estatística.
- **Avaliação por padrão textual.** As métricas medem presença de padrões (citação de fonte, ressalva, rótulo), não a correção clínica das respostas.
- **A fronteira entre "informativo" e "conduta" tem zona cinzenta.** Alguns exemplos ficam no limite — descrever a suspensão de medicamentos no pré-operatório é informativo ou conduta? O critério do guardrail não é perfeitamente objetivo.

---

## Requisitos do desafio

| Requisito | Status |
|---|---|
| 1. Fine-tuning com dados médicos internos | ✅ Concluído |
| — preprocessing, anonimização e curadoria | ✅ Concluído |
| — treino do modelo (QLoRA) | ✅ Concluído |
| 2. Assistente médico com LangChain | ⬜ Pendente |
| 3. Segurança e validação (limites, logging, explainability) | 🟡 Guardrails no modelo; logging e explainability pendentes |
| 4. Organização do código e README | 🟡 Em construção |

---

## Equipe

<!-- Preencher com os integrantes do grupo -->

| Nome | RM |
|---|---|
| | |
