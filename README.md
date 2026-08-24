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
│   └── 01_gerar_dataset.ipynb      # pipeline de geração, anonimização e curadoria
└── README.md
```

Estrutura prevista conforme o projeto avança:

```
├── notebooks/
│   └── 02_finetuning.ipynb         # fine-tuning QLoRA
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
| `02_finetuning.ipynb` | Fine-tuning QLoRA do modelo base | GPU (L4 ou T4), conta Hugging Face | Sim |

O notebook de geração é **documentação do processo**, não um passo obrigatório de reprodução. Quem quiser treinar o modelo pode partir direto do `.jsonl` versionado, sem precisar de chave de API.

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

## Limitações conhecidas

Declaradas explicitamente por rigor metodológico:

- **Exatidão regulatória não validada.** Protocolos, códigos, listas da Portaria 344 e posologias são inventados e não foram revisados por profissional de saúde.
- **Anonimização por regex tem casos de borda.** Durante o desenvolvimento, um nome com quatro iniciais escapou de um padrão escrito para três — detectado e corrigido por verificação automática. Em produção, ferramentas dedicadas (ex.: Microsoft Presidio) seriam o caminho.
- **Volume dimensiona uma demonstração.** 95 exemplos comprovam o pipeline; não produzem um modelo de produção.
- **O fine-tuning ensina forma, não conteúdo.** Com LoRA neste volume, o modelo aprende a citar fonte, pedir validação e emitir rótulos de decisão. O conhecimento clínico factual vem do RAG.
- **A fronteira entre "informativo" e "conduta" tem zona cinzenta.** Alguns exemplos ficam no limite — descrever a suspensão de medicamentos no pré-operatório é informativo ou conduta? O critério do guardrail não é perfeitamente objetivo.

---

## Requisitos do desafio

| Requisito | Status |
|---|---|
| 1. Fine-tuning com dados médicos internos | 🟡 Dados preparados; treino pendente |
| — preprocessing, anonimização e curadoria | ✅ Concluído |
| 2. Assistente médico com LangChain | ⬜ Pendente |
| 3. Segurança e validação (limites, logging, explainability) | ⬜ Pendente |
| 4. Organização do código e README | 🟡 Em construção |

---

## Como reproduzir

Instruções detalhadas serão adicionadas conforme cada etapa é concluída.

### Fine-tuning

```bash
# em construção
```

### Assistente

```bash
# em construção
```

---

## Equipe

<!-- Preencher com os integrantes do grupo -->

| Nome | RM |
|---|---|
| | |
