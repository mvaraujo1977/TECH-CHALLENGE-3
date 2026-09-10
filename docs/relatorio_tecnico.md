# Relatório Técnico

**Tech Challenge — Fase 3**
Assistente virtual médico de apoio à decisão clínica

---

> ⚠️ Todos os dados deste projeto são sintéticos. Protocolos, códigos internos,
> doses e pacientes são fictícios, não foram revisados por profissional de saúde
> e não devem ser usados para decisões clínicas reais.

---

## Sumário

1. [Objetivo e escopo](#1-objetivo-e-escopo)
2. [Preparação dos dados](#2-preparação-dos-dados)
3. [Fine-tuning do modelo](#3-fine-tuning-do-modelo)
4. [O assistente médico](#4-o-assistente-médico)
5. [Diagrama do fluxo](#5-diagrama-do-fluxo)
6. [Segurança e validação](#6-segurança-e-validação)
7. [Avaliação e análise dos resultados](#7-avaliação-e-análise-dos-resultados)
8. [Limitações](#8-limitações)
9. [Conclusão](#9-conclusão)

---

## 1. Objetivo e escopo

O desafio propõe a construção de um assistente virtual médico treinado com dados
próprios de um hospital, capaz de auxiliar em condutas clínicas, responder
dúvidas de médicos e sugerir procedimentos com base em protocolos internos — com
fluxos de decisão automatizados e seguros coordenados via LangChain.

O sistema entregue recebe uma pergunta clínica, opcionalmente vinculada a um
paciente, e produz uma resposta que recupera os protocolos pertinentes,
contextualiza com os dados do prontuário, roteia para um de três desfechos
clínicos, cita as fontes consultadas, exige validação humana e registra a
consulta em log auditável.

### Ambiente de desenvolvimento

| | |
|---|---|
| Linguagem | Python 3.13 |
| Orquestração | LangChain 1.4, LangGraph 1.2 |
| Modelo | Transformers 5.16, PEFT 0.20 |
| Vector store | Chroma 1.5 |
| Gestão de pacotes | `uv` |
| Treino e demonstração | Google Colab (GPU T4/L4) |
| Desenvolvimento | Local, CPU |

O código foi escrito para funcionar nos dois ambientes: `src/config.py` resolve
caminhos relativos à raiz do projeto, e `src/llm/modelo.py` detecta a presença
de GPU em tempo de execução, escolhendo entre quantização 4-bit e carga em
precisão plena.

---

## 2. Preparação dos dados

O enunciado pede preparação com técnicas de *preprocessing*, anonimização e
curadoria. Três bases foram construídas, com papéis distintos.

| Base | Papel | Volume |
|---|---|---|
| `dataset_medico.jsonl` | Treino do modelo | 95 exemplos |
| `protocolos/` | Conhecimento recuperável (RAG) | 14 documentos, 36 chunks |
| `prontuarios.json` | Dados de paciente em execução | 8 pacientes |

A distinção é relevante para interpretar os resultados: o dataset de treino
ensina **comportamento**; o conhecimento factual vem dos protocolos, recuperados
em tempo de execução.

### 2.1 Geração do dataset de fine-tuning

Os datasets sugeridos pelo enunciado (PubMedQA, MedQuAD) contêm perguntas e
respostas derivadas de literatura médica pública. O desafio, porém, pede dados
"próprios do hospital" e "protocolos internos" — categoria que aqueles conjuntos
não representam. Optou-se por gerar dados sintéticos simulando documentos
internos, o que adere melhor ao enunciado.

A geração foi feita com um LLM (Claude Sonnet), a partir de um prompt com regras
explícitas: citar sempre uma fonte no formato `PROT-0XX`, incluir ressalva de
validação humana quando houver sugestão de conduta, omiti-la em perguntas
puramente informativas, e responder em no máximo cinco linhas.

Cinco categorias, com pesos definidos pela utilidade para o projeto:

| Categoria | Exemplos | Conteúdo |
|---|---:|---|
| `cenarios_clinicos` | 25 | Casos com dados de paciente exigindo decisão |
| `protocolos` | 20 | Protocolos internos (sepse, TEP, AVC, pré-operatório) |
| `faq_medicos` | 20 | Dúvidas sobre condutas e procedimentos |
| `laudos` | 15 | Estrutura de laudos |
| `receitas` | 15 | Regras de prescrição, incluindo controlados |

A categoria `cenarios_clinicos` recebeu o maior peso porque é a que treina o
comportamento de decisão — cada exemplo inclui dados de paciente no campo
`input` e uma resposta iniciada por um rótulo de desfecho.

### 2.2 Preprocessing

Pipeline aplicado em ordem fixa:

```
gerar → deduplicar → normalizar desfecho → limpar meta-vazamento
      → ANONIMIZAR → verificar → salvar
```

A ordem não é arbitrária. A anonimização vem por último porque uma nova rodada
de geração intercalada reintroduziria dados não tratados — o que ocorreu durante
o desenvolvimento e foi detectado pela verificação automática.

**Geração em lotes de 5.** Lotes maiores produziam truncamento por limite de
tokens: o último exemplo de cada resposta saía cortado, gerando JSON inválido
que o parser descartava silenciosamente.

**Deduplicação e descarte de truncados.** Verificação de instruções repetidas e
de respostas que não terminam em pontuação final.

**Normalização do rótulo de desfecho.** O modelo gerador produziu três formatos
distintos (`Desfecho: X.`, `X — `, `X: `). Todos foram normalizados para
`DESFECHO: <RÓTULO>` na primeira linha, com quebra de linha, para tornar o
parsing determinístico.

**Limpeza de meta-vazamento.** Sete exemplos continham a instrução do prompt
vazada para dentro do conteúdo — construções como "Identificação (fictícia, sem
dados reais)" dentro de um modelo de laudo. Treinar com isso ensinaria o modelo
a escrever ressalvas de meta em respostas de produção.

### 2.3 Anonimização

Duas camadas independentes:

1. **Restrição no prompt de geração** — instrução explícita para não produzir
   nomes, CPFs ou datas identificáveis.
2. **Regex de pós-processamento** — CPF, data, cartão nacional de saúde (15
   dígitos), iniciais em sequência e nomes próprios após marcador de pessoa.

A segunda camada existe porque a primeira não é confiável: o modelo produziu
iniciais de paciente (`R.M.T.S.`) apesar da instrução.

Um caso de borda encontrado durante o desenvolvimento vale registro. O regex
inicial cobria três iniciais consecutivas; um nome com quatro produziu o resíduo
`[INICIAIS_REMOVIDAS]S.` em 20 exemplos. Foi detectado pela verificação
automática de resíduos e corrigido com um padrão de quantidade variável.

Também se descartou um padrão para inicial única (`\b[A-Z]\.\b`): ele destruía
texto clínico legítimo, transformando "E. coli" em "[INICIAL_REMOVIDA] coli".

### 2.4 Curadoria

Revisão humana por amostragem em cada categoria, mais métricas automáticas.

| Verificação | Resultado |
|---|---|
| Citação de fonte | 95/95 |
| Ressalva de validação | 57/95 |
| — nos exemplos que sugerem conduta | 100% |
| Exemplos truncados | 0 |
| Instruções duplicadas | 0 |
| Resíduos de anonimização | 0 |

A cobertura de 60% na ressalva é **intencional**. O guardrail aparece onde há
sugestão de conduta clínica e é omitido em perguntas informativas (estrutura de
laudo, campos de receita). Uma ressalva presente em 100% das respostas seria
ruído: o objetivo é que o modelo aprenda *quando* a validação é necessária, não
a repetir a frase por reflexo.

Distribuição dos três desfechos nos cenários clínicos: 9 `VERIFICAR_EXAMES`,
8 `EMITIR_ALERTA`, 8 `SUGERIR_CONDUTA` — equilíbrio necessário para que cada
caminho do grafo tenha massa de treino equivalente.

---

## 3. Fine-tuning do modelo

### 3.1 Escolha do modelo base

O enunciado sugere LLaMA ou Falcon, admitindo outros. A primeira escolha foi
`meta-llama/Llama-3.2-3B-Instruct`, que retornou `403 GatedRepoError`: o modelo
exige aprovação individual da Meta, com prazo indeterminado.

Optou-se por `Qwen/Qwen2.5-3B-Instruct`:

- **Sem gate de licença**, o que preserva a reprodutibilidade do projeto para o
  restante do grupo e para a banca
- **Mesma faixa de parâmetros** (3B), cabendo em T4/L4 com quantização 4-bit
- **Template de chat bem definido**, aplicado via `apply_chat_template()`

A troca exige alterar uma linha (`MODELO_BASE` em `src/config.py`).

### 3.2 Técnica: QLoRA

Fine-tuning completo de um modelo de 3B exige memória além da disponível em GPU
gratuita. Adotou-se QLoRA: o modelo base é carregado em 4-bit (NF4, com dupla
quantização) e apenas as matrizes LoRA são treinadas.

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `r` | 16 | Com 81 exemplos, ranks maiores tendem a overfitting |
| `alpha` | 32 | Razão 1:2 com `r`, usual |
| `dropout` | 0.05 | Regularização leve |
| `target_modules` | atenção + MLP | Incluir MLP melhora aderência a formato |
| Épocas | 4 | Mais que o usual, compensando dataset pequeno |
| Learning rate | 2e-4, cosine | Padrão para LoRA |
| Batch efetivo | 8 | 2 × acumulação de gradiente 4 |
| `max_seq_length` | 1024 | Maior exemplo tem ~355 tokens |

`load_best_model_at_end` recupera o checkpoint com melhor `eval_loss`,
protegendo contra degradação nas últimas épocas.

### 3.3 Formatação dos exemplos

Os exemplos foram formatados com `tokenizer.apply_chat_template()`, e não com
marcadores inventados. Cada modelo tem tokens especiais próprios definidos no
pré-treino; usar formato divergente degrada o resultado.

O `SYSTEM_PROMPT` usado no treino é o mesmo usado em inferência — divergência
entre os dois desfaz o comportamento aprendido.

### 3.4 Divisão treino/avaliação

Split **estratificado por categoria**, não aleatório: com 95 exemplos, uma
divisão simples poderia deixar o conjunto de avaliação sem nenhum cenário
clínico, que é justamente o comportamento mais importante de medir.

| Categoria | Treino | Avaliação |
|---|---:|---:|
| `cenarios_clinicos` | 21 | 4 |
| `protocolos` | 17 | 3 |
| `faq_medicos` | 17 | 3 |
| `laudos` | 13 | 2 |
| `receitas` | 13 | 2 |
| **Total** | **81** | **14** |

### 3.5 Baseline

Antes de aplicar os adapters, as respostas do modelo base foram registradas para
comparação. Sem baseline, não há como afirmar que o fine-tuning alterou o
comportamento.

A curva de perda está em `docs/resultados/curva_loss.png` e as respostas do
conjunto de avaliação em `docs/resultados/avaliacao_modelo.json`.

### 3.6 Publicação

Os adapters LoRA foram publicados em
[`mvaraujo1977/assistente-medico-lora`](https://huggingface.co/mvaraujo1977/assistente-medico-lora),
repositório público. Apenas os pesos do LoRA são versionados no Hub (~131 MB);
o modelo base é baixado separadamente.

O repositório foi mantido público deliberadamente: privado, exigiria token de
acesso de quem quisesse reproduzir o projeto.

---

## 4. O assistente médico

### 4.1 Arquitetura em três camadas

| Camada | O que garante | Onde vive |
|---|---|---|
| Modelo fine-tuned | Formato e comportamento | Adapters LoRA |
| RAG (LangChain) | Conteúdo factual dos protocolos | Vector store Chroma |
| Grafo (LangGraph) | Decisão de fluxo auditável | Regra determinística |

A separação responde a uma constatação: fine-tuning com LoRA sobre 95 exemplos
não insere conhecimento clínico nos pesos. O que ele ensina é forma — citar
fonte, estruturar a resposta, incluir a ressalva. O conhecimento factual precisa
vir de fora, recuperado no momento da consulta.

### 4.2 Recuperação (RAG)

Os 14 protocolos são documentos markdown com frontmatter YAML (`codigo`,
`titulo`, `versao`, `setor`, `revisao`). O frontmatter alimenta os metadados dos
chunks, viabilizando a citação rastreável exigida pelo requisito de
explainability.

**Chunking.** `RecursiveCharacterTextSplitter` com separadores em ordem de
preferência: cabeçalho markdown, parágrafo, linha, caractere. Isso evita cortar
tabela ou lista numerada ao meio, o que produziria chunk sem sentido. Resultado:
36 chunks de 241 a 795 caracteres.

**Embeddings.** `BAAI/bge-m3`, multilíngue, executado localmente. A escolha por
modelo local em vez de API é coerente com a premissa do cenário: dados
hospitalares não saem da instituição.

**Retriever com escopo.** Detalhado na seção 7.2.

### 4.3 Consulta à base estruturada

O módulo `src/rag/prontuarios.py` atende ao requisito de "consultas em base de
dados estruturadas". Em produção seria um banco relacional ou uma API do sistema
hospitalar; aqui é um JSON com a mesma interface de consulta.

Funções expostas: listar pacientes, buscar por id, separar exames disponíveis de
pendentes, detectar sinais de gravidade e serializar o prontuário para o prompt.

A serialização merece nota: o prontuário é convertido em texto corrido com
sinais vitais e exames em linguagem natural, **no mesmo formato que o modelo viu
durante o treino**. Passar JSON cru degradaria a resposta, porque divergiria do
padrão aprendido.

### 4.4 Fluxo de decisão (LangGraph)

Sete nós, dos quais três são os desfechos exigidos pelo enunciado.

| Nó | Função |
|---|---|
| `carregar_paciente` | Busca o prontuário na base estruturada |
| `recuperar_protocolos` | RAG com escopo do prontuário |
| `consultar_modelo` | Gera o texto da resposta |
| `decidir_desfecho` | Aplica a regra determinística |
| `verificar_exames` | Lista pendentes, recusa conduta definitiva |
| `sugerir_conduta` | Descreve o previsto, com ressalva |
| `emitir_alerta` | Sinaliza urgência e achados de alarme |
| `finalizar` | Garante guardrail, anexa fontes |

O estado é um `TypedDict` (`EstadoClinico`), e cada nó devolve apenas as chaves
que alterou — o LangGraph compõe o merge.

**Injeção de dependência.** O retriever e a função de geração são passados de
fora, não importados dentro dos nós. Isso permitiu exercitar o grafo inteiro com
dublês, sem carregar embeddings nem o modelo de 3B — decisão que viabilizou
desenvolvimento e teste em máquina sem GPU.

---

## 5. Diagrama do fluxo

![Diagrama do grafo](resultados/diagrama_grafo.png)

```
START
  ↓
carregar_paciente        consulta à base estruturada de prontuários
  ↓
recuperar_protocolos     RAG sobre os protocolos internos
  ↓
consultar_modelo         LLM fine-tuned gera o texto
  ↓
decidir_desfecho         regra determinística sobre o prontuário
  ↓
[roteamento condicional]
  ├──> verificar_exames
  ├──> sugerir_conduta
  └──> emitir_alerta
          ↓
       finalizar         guardrail + citação de fontes
          ↓
        END
```

O diagrama é gerado a partir do grafo compilado
(`grafo.get_graph().draw_mermaid_png()`), não desenhado separadamente — logo não
pode divergir da implementação.

---

## 6. Segurança e validação

### 6.1 Limites de atuação

O assistente nunca prescreve diretamente. A ressalva de validação humana é
garantida em duas camadas:

1. **Fine-tuning** — o modelo aprendeu a fechar respostas de conduta com a
   ressalva. Na execução medida, fez isso em 7 de 8 casos.
2. **Código** — `garantir_guardrail()` insere a frase quando ausente, e o
   registro de auditoria marca a intervenção.

A segunda camada não é redundância. Em execução com regime numérico diferente
(bfloat16 em CPU), a taxa espontânea caiu para 1 de 2.

**Um defeito encontrado em teste vale registro**, porque ilustra a fragilidade
de verificação por palavra-chave. A checagem original era feita sobre o texto
final montado, que inclui um bloco de ações recomendadas. A ação "Acionar
imediatamente o médico responsável" contém as mesmas palavras da ressalva de
validação, então o detector concluía que o aviso estava presente e não o
inseria — produzindo respostas com sugestão de conduta sem o aviso de segurança.
A correção foi avaliar o guardrail apenas sobre o texto do modelo, isolado das
ações.

### 6.2 Logging para auditoria

Cada consulta grava um registro em JSONL, append-only. Campos: identificador,
timestamp UTC, pergunta, paciente, fontes recuperadas com versão e índices de
chunk, desfecho, motivo da decisão, sinais de gravidade, exames pendentes,
rótulo emitido pelo modelo, concordância com a regra, se o guardrail foi
inserido, caminho percorrido no grafo, duração e identificação do modelo.

O formato JSONL permite auditar com ferramentas de linha de comando e carregar
em pandas sem parsing customizado. `Auditoria.estatisticas()` agrega as métricas.

O campo `guardrail_adicionado` é notável: ele mede quantas vezes o modelo falhou
no requisito de segurança e precisou de intervenção do código.

### 6.3 Explainability

A citação vem dos **documentos efetivamente recuperados pelo retriever**, não do
código que o modelo escreveu no texto.

Essa distinção decorre de uma observação sobre o dataset: durante a geração, o
mesmo código foi associado a temas diferentes em lotes distintos — `PROT-012`
aparece como tromboembolismo, checklist de alta e controle glicêmico. O
fine-tuning ensinou o *formato* de citar uma fonte, não um mapeamento
código→conteúdo. Os códigos que o modelo produz não são ponteiros confiáveis.

Cada fonte citada traz código, título, versão e os índices de chunk usados,
permitindo rastrear até o arquivo e a seção.

---

## 7. Avaliação e análise dos resultados

Execução dos 8 pacientes em GPU T4, modelo em 4-bit.

### 7.1 Métricas gerais

| Métrica | Resultado |
|---|---|
| Citação de fonte na resposta | 8/8 |
| Fonte recuperada pertinente ao caso | 8/8 |
| Ressalva de validação na resposta final | 8/8 |
| — espontânea do modelo | 7/8 |
| — inserida por código | 1/8 |
| Rótulo de decisão válido emitido pelo modelo | 2/8 |
| — clinicamente correto | 0/8 |
| Registros de auditoria completos | 8/8 |

Tempo médio: ~30 s por consulta em T4, contra ~500 s em CPU.

### 7.2 Avaliação da recuperação

A camada de RAG foi medida isoladamente, sem o LLM, comparando os protocolos
recuperados com o campo `protocolos_relacionados` de cada prontuário.

A primeira implementação usava corte por score de similaridade. A medição mostrou
que **não existe limiar que separe protocolo pertinente de irrelevante**:

| Conjunto | n | mediana |
|---|---:|---:|
| Chunks em `protocolos_relacionados` | 18 | 0.558 |
| Chunks fora | 62 | 0.550 |

Diferença de 0.008 entre as medianas, com sobreposição quase total das
distribuições. Com o corte em 0.35, 13 dos 16 chunks corretos eram descartados e
12 incorretos passavam. O filtro operava como redutor de volume, não de
pertinência.

Um achado técnico contribuiu para o diagnóstico: a coleção Chroma havia sido
criada sem especificar `hnsw:space`, adotando L2 por padrão, e o LangChain
converte a distância euclidiana numa faixa comprimida. O score não era
similaridade de cosseno, o que tornava incomparável qualquer limiar publicado
para o `bge-m3`. A coleção foi recriada com métrica de cosseno.

**Decisão.** Quando o prontuário indica quais protocolos se aplicam, a busca é
restrita a eles, sem corte de relevância. O corte só vale para busca livre.

| | Filtro por similaridade | Escopo curado |
|---|---:|---:|
| Acertos | 3/11 | 11/11 |
| Protocolo de outra condição | 5 | 0 |
| Faltantes | 8 | 0 |
| Pacientes com escopo completo | 2/8 | 8/8 |

Dois dos chunks devolvidos ficam abaixo do corte (0.4202 e 0.4200, no paciente
com sepse). Se o corte valesse dentro do escopo, esse caso ficaria sem protocolo
e a busca livre traria tromboembolismo em seu lugar — a contaminação que a
mudança pretendia eliminar.

### 7.3 Avaliação do roteamento

O modelo continua emitindo o rótulo `DESFECHO:`, mas ele não decide o fluxo.
Comparar os dois mede quão confiável seria delegar a decisão ao LLM.

Resultado: **nenhum dos 8 casos produziu rótulo utilizável.**

| Paciente | Primeira linha da resposta |
|---|---|
| PAC-002 | `DESFECHO: VERIFICAR` |
| PAC-006 | `DESFECHO: VERIFICAR CONTA` |
| PAC-007, PAC-008 | `DESFECHO: AVALIAR` |
| PAC-004, PAC-005 | `DESFECHO: SUGERIR_CONDUTA` (válido, clinicamente errado) |
| PAC-001, PAC-003 | nenhum rótulo |

Os dois rótulos sintaticamente válidos são os mais preocupantes: classificaram
AVC em janela terapêutica (NIHSS 8) e cetoacidose diabética grave (pH 7,18) como
conduta de rotina. Um rótulo válido e errado é mais perigoso que nenhum rótulo,
porque passaria por qualquer validação de formato.

Em execução anterior, em CPU com bfloat16, o modelo produzira `SUGERIR CONDUÇÃO`
— corrupção de token distinta, mesma taxa de acerto: zero.

**Decisão.** A decisão de fluxo é tomada por código:

```
sinais de gravidade  → EMITIR_ALERTA
exames pendentes     → VERIFICAR_EXAMES
nenhum dos dois      → SUGERIR_CONDUTA
```

Gravidade tem precedência sobre exames pendentes: paciente instável precisa de
alerta imediato, e o nó de alerta lista os pendentes de todo modo.

Nos dois casos em que o modelo classificou urgência como rotina, a regra
corrigiu. Se o roteamento dependesse do LLM, ambos seguiriam o caminho de menor
urgência.

### 7.4 Detecção de gravidade

A heurística inicial lia apenas sinais vitais e deixou passar um IAMCSST
confirmado — supradesnivelamento de ST em três derivações e troponina 3,8 ng/mL
(referência < 0,04) — porque pressão, saturação e frequência estavam normais. A
gravidade estava no eletrocardiograma, não nos sinais vitais.

A detecção foi ampliada para três fontes: sinais vitais com limiares
conservadores, resultados de exames (achados críticos em texto livre e limiares
numéricos de troponina e lactato) e protocolos de urgência associados ao paciente
no prontuário.

A ampliação introduziu um falso positivo que vale registro. A busca por achados
em texto livre é feita por substring, e o laudo "Sem hemorragia ou isquemia
aguda" gerava alerta de achado hemorrágico — invertendo o significado do
documento. Corrigido com verificação de termos de negação numa janela anterior ao
achado, validada em 8 casos.

A solução é frágil por natureza: análise de negação em texto clínico livre é
problema aberto, e uma formulação não prevista volta a produzir o erro.

### 7.5 Análise crítica: o que o fine-tuning entrega

Os resultados separam com clareza duas coisas.

**Entrega forma.** Em 8/8 respostas o formato está correto: cita fonte no padrão
esperado, enumera condutas, fecha com a ressalva. O comportamento treinado se
manifestou.

**Não entrega substância.** Em 3 dos 8 casos o conteúdo clínico está errado.

**Caso 1 — inversão do protocolo recuperado.** No paciente com sepse, com o
`PROT-001` corretamente recuperado e presente no contexto, o modelo escreveu:

> "O protocolo prevê antibioticoterapia empírica **antes** da coleta de
> hemocultura."

O protocolo determina o oposto, e o texto estava no contexto fornecido. A própria
resposta se contradiz no item seguinte, instruindo a coletar antes.

Isso é qualitativamente diferente de falha de recuperação: a fonte está correta e
o modelo distorce o que ela diz. Nenhuma melhoria no RAG corrige esse erro.

**Caso 2 — posologia inventada.** No paciente em pós-operatório:

> "A dose inicial pode ser 100 mg enoxaparina dupla via (20 mg cada viço) por 24
> horas, seguida de dose única de 40 mg dupla via."

A dose não consta do `PROT-011`, que trata de indicação e contraindicação sem
especificar posologia. "Viço" é corrupção de token.

**Caso 3 — classificação de gravidade equivocada**, já descrito em 7.3.

Em todos os três o erro vem acompanhado de citação formalmente correta, o que o
torna mais difícil de detectar, não menos.

### 7.6 Reposicionamento das camadas

| Camada | O que garante | O que não garante |
|---|---|---|
| Fine-tuning | Formato, tom, presença da ressalva | Correção do conteúdo |
| RAG | Que a fonte certa esteja disponível | Que o modelo a use corretamente |
| Grafo determinístico | Decisão de fluxo auditável | Correção do texto gerado |
| Validação humana | — | É a única camada que cobre o conteúdo |

A exigência de validação humana, que o enunciado apresenta como requisito de
segurança, não é formalidade de conformidade. Na execução medida, é a única
camada capaz de interceptar a inversão do `PROT-001` antes que chegue ao
paciente.

---

## 8. Limitações

**Escala.** 8 pacientes, 14 protocolos, 95 exemplos de treino. Os números
indicam tendência, não significância estatística. Uma taxa de 3/8 de erro
clínico tem intervalo de confiança largo demais para comparação com qualquer
referência.

**Dados sintéticos.** Protocolos, prontuários, doses e códigos são fictícios,
gerados por LLM e revisados por amostragem, sem validação clínica. A avaliação
mede o comportamento do sistema sobre esses dados, não desempenho clínico.

**Curadoria assumida correta.** O escopo de recuperação depende do campo
`protocolos_relacionados`, atribuído manualmente na construção da base. Num
sistema real, essa atribuição precisaria ser validada, e um erro nela propagaria
para a recuperação sem sinal de alerta.

**Avaliação por padrão textual.** As métricas de guardrail e citação verificam
presença de padrão por palavra-chave. Medem se a ressalva está no texto, não se é
adequada ao conteúdo; medem se há código citado, não se a citação sustenta a
afirmação feita.

**Correção clínica não medida sistematicamente.** Os três erros de conteúdo foram
encontrados por leitura das respostas, não por método. Não há métrica automática
de fidelidade ao protocolo recuperado, e construí-la exigiria anotação por
profissional de saúde.

**Baseline incompleto.** A comparação com o modelo base sem adapters ficou como
célula opcional no notebook de demonstração e não foi executada na rodada final.
Sem ela, a atribuição das melhorias de formato ao fine-tuning é inferência, não
medição.

**Anonimização por regex.** Solução com casos de borda conhecidos, como o
episódio das quatro iniciais. Em produção, ferramentas dedicadas (Microsoft
Presidio, por exemplo) seriam o caminho.

---

## 9. Conclusão

O sistema atende aos quatro requisitos do desafio: fine-tuning com dados
preparados, anonimizados e curados; assistente com LangChain integrando a LLM
customizada, consultando base estruturada e contextualizando com dados do
paciente; segurança com limites de atuação, logging e explainability; e projeto
modularizado com documentação.

O resultado técnico mais relevante, porém, não é o funcionamento do pipeline. É a
medição de **onde o LLM é confiável e onde não é**, e o redesenho da arquitetura
a partir disso.

Duas responsabilidades foram movidas do modelo para código determinístico, cada
uma justificada por medição:

- **Roteamento**, porque o modelo falhou em 8/8 casos e, nos dois em que produziu
  rótulo válido, classificou urgência como rotina
- **Escopo de recuperação**, porque similaridade de texto não separava protocolo
  pertinente de irrelevante, com diferença de 0.008 entre as medianas

O que o LLM faz bem neste sistema é gerar texto estruturado e citar fonte no
formato correto — 8/8 nas duas métricas. O que ele não faz é garantir a correção
do conteúdo clínico, e três casos em oito documentam isso com precisão.

Para um sistema de apoio à decisão em saúde, essa é a conclusão útil: o valor não
está em automatizar a decisão, mas em organizar e apresentar informação
rastreável, deixando a decisão com o profissional. A camada de validação humana
não é uma concessão regulatória sobre um sistema que funcionaria sem ela — é a
camada que faz o sistema seguro.

---

## Referências

- Enunciado do Tech Challenge — Fase 3, Pós-Tech
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023)
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks* (2020)
- Documentação LangChain e LangGraph
- Chen et al., *BGE M3-Embedding* (2024)

## Artefatos

| Artefato | Localização |
|---|---|
| Código-fonte | `src/` |
| Pipeline de fine-tuning | `notebooks/02_finetuning.ipynb` |
| Geração e curadoria do dataset | `notebooks/01_gerar_dataset.ipynb` |
| Demonstração de ponta a ponta | `notebooks/03_demo_assistente.ipynb` |
| Dataset anonimizado | `data/dataset_medico.jsonl` |
| Curva de perda do treino | `docs/resultados/curva_loss.png` |
| Avaliação do fine-tuning | `docs/resultados/avaliacao_modelo.json` |
| Validação da recuperação | `docs/resultados/validacao_rag.md` |
| Log da execução completa | `docs/resultados/demo.jsonl` |
| Diagrama do fluxo | `docs/resultados/diagrama_grafo.png` |
| Análise e limitações | `docs/analise_e_limitacoes.md` |
| Modelo publicado | https://huggingface.co/mvaraujo1977/assistente-medico-lora |
| Repositório | https://github.com/mvaraujo1977/TECH-CHALLENGE-3 |
