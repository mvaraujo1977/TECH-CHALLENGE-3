# Validação da recuperação (RAG)

Medição da camada de recuperação isolada, sem carregar o LLM. Avalia se os
protocolos que chegam ao contexto do modelo são os que o prontuário indica
como aplicáveis ao paciente.

- **Data:** 2026-09-09
- **Embeddings:** `BAAI/bge-m3`, normalizados
- **Índice:** Chroma, 36 chunks de 14 protocolos, `hnsw:space=cosine`
- **Parâmetros:** `TOP_K=4`, `LIMITE_RELEVANCIA=0.45`
- **Referência:** campo `protocolos_relacionados` de cada prontuário
- **Consulta:** queixa + antecedentes (sem a pergunta do usuário)

A validação exercita os nós reais do grafo (`carregar_paciente` e
`recuperar_protocolos`), não uma reimplementação, de modo que a tabela reflete
o caminho de produção.

## Antes e depois

O estado "antes" corresponde ao índice em L2 com relevância derivada da
transformação euclidiana, corte de 0.35 aplicado também dentro do escopo, e
fallback que repetia a busca sem escopo quando nada passava do corte.

| Métrica | Antes | Depois |
|---|---|---|
| Acertos | 3 / 11 | **11 / 11** |
| Extras (protocolo de outra condição) | 5 | **0** |
| Faltantes | 8 | **0** |
| Pacientes com escopo completo | 2 / 8 | **8 / 8** |
| Pacientes sem nenhum extra | 5 / 8 | **8 / 8** |

Os três casos que devolviam protocolo de outra condição foram corrigidos:

| Paciente | Quadro | Antes | Depois |
|---|---|---|---|
| PAC-005 | Cetoacidose | PROT-007 (TEP) | PROT-009 |
| PAC-007 | Cefaleia súbita | PROT-031, PROT-012, PROT-003 | PROT-019 |
| PAC-008 | Sepse | PROT-007 (TEP) | PROT-001 |

Os dois que devolviam contexto vazio (PAC-003 e PAC-006) passaram a trazer os
dois protocolos curados de cada um.

## Por paciente

| Paciente | Esperados | Devolvidos | Acertos | Extras | Faltantes | Scores |
|---|---|---|---|---|---|---|
| PAC-001 | PROT-007, PROT-011 | PROT-007, PROT-011 | 2 | 0 | 0 | 0.619 – 0.514 |
| PAC-002 | PROT-002 | PROT-002 | 1 | 0 | 0 | 0.653 – 0.529 |
| PAC-003 | PROT-012, PROT-011 | PROT-012, PROT-011 | 2 | 0 | 0 | 0.546 – 0.488 |
| PAC-004 | PROT-003 | PROT-003 | 1 | 0 | 0 | 0.607 – 0.475 |
| PAC-005 | PROT-009 | PROT-009 | 1 | 0 | 0 | 0.592 – 0.540 |
| PAC-006 | PROT-001, PROT-025 | PROT-025, PROT-001 | 2 | 0 | 0 | 0.568 – 0.479 |
| PAC-007 | PROT-019 | PROT-019 | 1 | 0 | 0 | 0.621 – 0.574 |
| PAC-008 | PROT-001 | PROT-001 | 1 | 0 | 0 | 0.459 – 0.420 |

O número de trechos devolvidos varia de 2 a 4: dentro do escopo, alguns
protocolos têm menos chunks que o `TOP_K`.

## Distribuição de scores (cosseno)

| Conjunto | n | mín | mediana | máx |
|---|---|---|---|---|
| Trechos efetivamente devolvidos | 26 | 0.4200 | 0.5384 | 0.6525 |
| Busca livre, dentro do escopo | 18 | 0.4586 | 0.5580 | 0.6525 |
| Busca livre, fora do escopo | 62 | 0.4384 | 0.5499 | 0.6196 |

Declarar a métrica como cosseno moveu a faixa de 0.21–0.46 para 0.35–0.65 e
tornou o valor interpretável. **A separação entre as classes, porém, continua
desprezível: 0.5580 contra 0.5499 de mediana, com faixas sobrepostas.**

A conclusão prática é que o score do bge-m3 não distingue, neste corpus, o
protocolo pertinente do impertinente. Nenhum limiar resolve: qualquer valor
alto o bastante para barrar os irrelevantes também derruba os corretos. Quem
faz o trabalho de pertinência é o escopo curado do prontuário; o corte fica
como piso contra material claramente alheio, aplicado só à busca livre.

Evidência direta: **2 dos 26 trechos devolvidos estão abaixo de 0.45** — os
dois do PAC-008, a 0.4202 e 0.4200, com o melhor trecho do caso a 0.4586.
Se o corte valesse dentro do escopo, o protocolo de sepse seria descartado
justamente no paciente com choque séptico.

## Escolha da consulta de embedding

Três variantes medidas sobre os mesmos 8 pacientes, avaliando onde o protocolo
esperado cai no ranking da busca livre (k=10):

| Consulta | MRR | 1º lugar | Top-4 |
|---|---|---|---|
| pergunta + queixa (anterior) | 0.461 | 2 / 8 | 6 / 8 |
| queixa | 0.581 | 3 / 8 | 6 / 8 |
| **queixa + antecedentes** | **0.667** | **4 / 8** | **7 / 8** |

A pergunta do usuário é quase constante entre consultas ("qual a conduta para
este paciente?") e dilui o sinal clínico. Os antecedentes acrescentam contexto
que os protocolos citam explicitamente. O custo é perder a intenção de uma
pergunta específica na recuperação — aceitável porque o escopo já restringe o
conjunto, e a pergunta continua presente no prompt do modelo.

## Limitações

- Amostra de 8 pacientes e 14 protocolos; os números são indicativos.
- A referência é o `protocolos_relacionados` do prontuário, curadoria humana
  que a medição assume correta.
- O corte de 0.45 foi calibrado com consultas derivadas de pacientes, usadas
  como proxy da busca livre, que não tem paciente por definição.
- `LIMITE_RELEVANCIA` depende da métrica do índice: reindexar em outra métrica
  invalida o valor.
