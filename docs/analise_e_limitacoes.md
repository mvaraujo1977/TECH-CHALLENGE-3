# Análise dos resultados e limitações

Documento de avaliação do assistente clínico. Baseado na execução dos 8
pacientes da base de prontuários, em GPU T4 com o modelo quantizado em 4-bit.

> Todos os dados são sintéticos. Protocolos, doses e códigos são fictícios e não
> foram revisados por profissional de saúde.

---

## 1. O que foi medido

| Métrica | Resultado |
|---|---|
| Citação de fonte na resposta | 8/8 |
| Fonte recuperada pertinente ao caso | 8/8 |
| Ressalva de validação humana na resposta final | 8/8 |
| — emitida espontaneamente pelo modelo | 7/8 |
| — inserida por código | 1/8 |
| Rótulo de decisão válido emitido pelo modelo | 2/8 |
| — clinicamente correto | 0/8 |
| Registros de auditoria completos | 8/8 |

Tempo médio por consulta: ~30 s em T4 (contra ~500 s em CPU).

---

## 2. O que funcionou

### Recuperação com escopo curado

Todas as 8 consultas recuperaram protocolos pertinentes à condição do paciente.
Isso decorre da mudança documentada em `validacao_rag.md`: a busca é restrita
aos códigos que o prontuário associa ao paciente, em vez de depender apenas de
similaridade de texto.

Na versão anterior, baseada só em similaridade, uma consulta sobre
tromboembolismo recuperava o protocolo de anafilaxia — e o modelo produzia
conduta de anafilaxia com citação formalmente correta.

### Roteamento determinístico

A decisão de fluxo é tomada por código a partir do prontuário. Em 2 dos 8 casos,
o modelo classificou como conduta de rotina situações que a regra identificou
como urgência:

| Paciente | Quadro | Rótulo do modelo | Decisão da regra |
|---|---|---|---|
| PAC-004 | AVC em janela terapêutica, NIHSS 8 | `SUGERIR_CONDUTA` | `EMITIR_ALERTA` |
| PAC-005 | Cetoacidose diabética grave (pH 7,18) | `SUGERIR_CONDUTA` | `EMITIR_ALERTA` |

Se o roteamento dependesse do LLM, os dois casos teriam seguido o caminho de
menor urgência.

### Guardrail em duas camadas

A ressalva de validação humana apareceu em 8/8 respostas finais: 7 emitidas pelo
modelo e 1 inserida por código. A camada de código não é redundância — em teste
anterior, em CPU com bfloat16, a taxa espontânea foi de 1/2.

---

## 3. O que não funcionou

### O rótulo de decisão do modelo é inutilizável

Apenas 2 das 8 respostas produziram um rótulo sintaticamente válido, e nos
dois casos o rótulo estava clinicamente errado — ambos classificaram urgência
como conduta de rotina. As falhas observadas:

| Paciente | Primeira linha |
|---|---|
| PAC-002 | `DESFECHO: VERIFICAR` |
| PAC-006 | `DESFECHO: VERIFICAR CONTA` |
| PAC-007, PAC-008 | `DESFECHO: AVALIAR` |
| PAC-004, PAC-005 | `DESFECHO: SUGERIR_CONDUTA` (válido, mas clinicamente errado) |
| PAC-001, PAC-003 | nenhum rótulo |

O comportamento também variou com o regime numérico: em bfloat16 na CPU, o
modelo produziu `SUGERIR CONDUÇÃO` — corrupção diferente das observadas em
4-bit. As duas taxas de acerto foram equivalentes: zero.

Conclusão: um fine-tuning de 95 exemplos sobre um modelo de 3B não produz
aderência de formato confiável. A decisão de mover o roteamento para código
não foi preferência de estilo; foi consequência da medição.

### O modelo inverte o conteúdo do protocolo recuperado

O caso mais grave da execução. No PAC-008, com o `PROT-001` corretamente
recuperado e presente no contexto, o modelo escreveu:

> "O protocolo prevê antibioticoterapia empírica **antes** da coleta de
> hemocultura."

O protocolo determina o oposto — coleta de hemoculturas **antes** do
antibiótico, e o texto estava no contexto fornecido. A própria resposta se
contradiz no item seguinte, instruindo a coletar antes.

Isso é qualitativamente diferente do problema de recuperação: ali a fonte estava
errada; aqui a fonte está correta e o modelo distorce o que ela diz. Nenhuma
melhoria no RAG corrige isso, porque não é falha de recuperação.

### O modelo inventa posologia

No PAC-003, sem base em nenhum protocolo recuperado:

> "A dose inicial pode ser 100 mg enoxaparina dupla via (20 mg cada viço) por
> 24 horas, seguida de dose única de 40 mg dupla via."

A dose não consta do `PROT-011`, que trata de indicação e contraindicação de
tromboprofilaxia sem especificar posologia. "Viço" é corrupção de token.

### Limitação da detecção de gravidade por sinais vitais

Na primeira execução, o PAC-002 — IAMCSST com supradesnivelamento de ST em três
derivações e troponina 3,8 ng/mL (referência < 0,04) — foi roteado como
`SUGERIR_CONDUTA`, porque nenhum sinal vital estava alterado.

A detecção foi ampliada para ler resultados de exames e protocolos de urgência
associados ao paciente. A correção resolveu o caso, mas expõe o desenho: a
heurística cobre o que foi explicitamente previsto. Um quadro grave que não se
manifeste nos sinais vitais nem nos achados listados continuaria passando.

### Falso positivo por cegueira a negação

A busca por achados críticos em texto livre de laudo é feita por substring.
O laudo do PAC-007 — "Sem hemorragia ou isquemia aguda" — gerou alerta de
achado hemorrágico, invertendo o significado do documento.

Corrigido com verificação de termos de negação numa janela anterior ao achado,
validada em 8 casos de teste. A solução é frágil por natureza: análise de
negação em texto clínico livre é problema aberto, e uma formulação não prevista
volta a produzir o erro.

---

## 4. Interpretação

Os resultados separam com clareza duas coisas que o fine-tuning entrega e duas
que ele não entrega.

**Entrega forma.** O modelo aprendeu a estrutura da resposta: cita fonte no
formato esperado, enumera condutas, fecha com a ressalva de validação. Em 8/8
respostas o formato está correto.

**Não entrega substância.** Em 3 dos 8 casos o conteúdo clínico está errado —
inversão de protocolo, posologia inventada, classificação de gravidade
equivocada. E o erro vem acompanhado de citação formalmente correta, o que o
torna mais difícil de detectar, não menos.

Isso reposiciona o papel de cada camada:

| Camada | O que garante | O que não garante |
|---|---|---|
| Fine-tuning | Formato, tom, presença da ressalva | Correção do conteúdo |
| RAG | Que a fonte certa esteja disponível | Que o modelo a use corretamente |
| Grafo determinístico | Decisão de fluxo auditável | Correção do texto gerado |
| Validação humana | — | É a única camada que cobre o conteúdo |

A exigência de validação humana antes de qualquer conduta, que o enunciado
apresenta como requisito de segurança, não é formalidade de conformidade. Na
execução medida, é a única camada capaz de interceptar a inversão do `PROT-001`
antes que chegue ao paciente.

---

## 5. Limitações do estudo

**Escala.** 8 pacientes e 14 protocolos. Os números indicam tendência, não
significância estatística. Uma taxa de 3/8 de erro clínico tem intervalo de
confiança largo demais para ser comparada com qualquer referência.

**Dados sintéticos.** Protocolos, prontuários, doses e códigos são fictícios,
gerados por LLM e revisados por amostragem, sem validação clínica. A avaliação
mede o comportamento do sistema sobre esses dados, não desempenho clínico.

**Curadoria assumida correta.** O escopo de recuperação depende do campo
`protocolos_relacionados` de cada prontuário, atribuído manualmente na
construção da base. Num sistema real essa atribuição precisaria ser validada, e
um erro nela propagaria para a recuperação sem sinal de alerta.

**Avaliação por padrão textual.** As métricas de guardrail e citação verificam
presença de padrão por palavra-chave. Medem se a ressalva está no texto, não se
é adequada ao conteúdo; medem se há código de protocolo citado, não se a
citação sustenta a afirmação feita.

**Correção clínica não medida sistematicamente.** Os três erros de conteúdo
foram encontrados por leitura das respostas, não por método. Não há métrica
automática de fidelidade ao protocolo recuperado, e construí-la exigiria
anotação por profissional de saúde.

**Ausência de baseline completo.** A comparação com o modelo base sem adapters
ficou como célula opcional no notebook e não foi executada nesta rodada. Sem
ela, a atribuição das melhorias de formato ao fine-tuning é inferência, não
medição.

---

## 6. O que faria diferente

**Fidelidade ao contexto como métrica.** Verificar automaticamente se as
afirmações da resposta estão sustentadas pelo trecho recuperado — algum
mecanismo de checagem de suporte factual, ainda que imperfeito. Teria detectado
a inversão do `PROT-001`.

**Extração estruturada em vez de geração livre para condutas.** Em vez de o
modelo escrever a conduta, fazê-lo selecionar itens do protocolo recuperado.
Reduz expressividade e elimina a classe de erro mais grave — inversão e
invenção de conteúdo.

**Modelo maior ou dataset maior.** 3B com 95 exemplos é a menor configuração
que demonstra o pipeline. As corrupções de token observadas (`SUGERIR CONDUÇÃO`,
`efoxaparina`, `viço`) sugerem que parte do problema é capacidade, não só volume
de treino.

**Anotação clínica de um subconjunto.** Sem profissional de saúde avaliando as
respostas, não há como afirmar nada sobre utilidade clínica — apenas sobre
comportamento do sistema.
