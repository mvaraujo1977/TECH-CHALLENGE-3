# Análise dos resultados e limitações

Documento de avaliação do assistente clínico, baseado em duas execuções
completas dos 8 pacientes da base de prontuários, em GPU T4 com o modelo
quantizado em 4-bit, mais uma execução do modelo base sem adapters para
comparação.

> Todos os dados são sintéticos. Protocolos, doses e códigos são fictícios e não
> foram revisados por profissional de saúde.

---

## 1. O que foi medido

Duas execuções completas foram realizadas. A segunda ocorreu após a ampliação
da detecção de gravidade descrita na seção 4, e é a referência para os números
abaixo.

| Métrica | Execução 1 | Execução 2 |
|---|---|---|
| Citação de fonte na resposta | 8/8 | 8/8 |
| Fonte recuperada pertinente ao caso | 8/8 | 8/8 |
| Ressalva de validação na resposta final | 8/8 | 8/8 |
| — espontânea do modelo | 7/8 | 8/8 |
| — inserida por código | 1/8 | 0/8 |
| Rótulo de decisão sintaticamente válido | 2/8 | 4/8 |
| — clinicamente correto | 0/2 | 0/4 |
| Erros de conteúdo clínico identificados | 3/8 | 1/8 |
| Registros de auditoria completos | 8/8 | 8/8 |

Tempo médio por consulta: ~30 s em T4, contra ~500 s em CPU.

Os registros da execução 2 estão em `docs/resultados/demo.jsonl`.

### Comparação com o modelo base

Para atribuir os efeitos observados ao fine-tuning, o mesmo pipeline foi
executado com o modelo base (`Qwen2.5-3B-Instruct` sem adapters LoRA), sobre
dois pacientes.

| Comportamento | Modelo base | Fine-tuned |
|---|---|---|
| Ressalva de validação espontânea | 0/2 | 8/8 |
| Emissão de rótulo `DESFECHO:` | 0/2 | 4/8 (nenhum correto) |
| Citação de protocolo recuperado | 2/2 | 8/8 |
| Formato enumerado e estruturado | 2/2 | 8/8 |

A comparação separa com precisão o que cada camada entrega:

- **A ressalva de validação é efeito do fine-tuning.** O modelo base não a
  produziu em nenhum caso; o fine-tuned, em todos.
- **A citação de protocolo é efeito do RAG, não do fine-tuning.** O modelo base
  cita `PROT-001` e `PROT-007` corretamente, porque o contexto está no prompt.
- **A emissão do rótulo de decisão é efeito parcial e inútil do fine-tuning.**
  O base nunca emite; o fine-tuned emite, mas com rótulo inválido ou
  clinicamente errado.

---

## 2. O que funcionou

### Recuperação com escopo curado

Todas as consulta das duas execuções recuperaram protocolos pertinentes à
condição do paciente. Decorre da mudança documentada em `validacao_rag.md`: a
busca é restrita aos códigos que o prontuário associa ao paciente, em vez de
depender apenas de similaridade de texto.

Na versão anterior, baseada só em similaridade, uma consulta sobre
tromboembolismo recuperava o protocolo de anafilaxia — e o modelo produzia
conduta de anafilaxia com citação formalmente correta.

### Roteamento determinístico

A decisão de fluxo é tomada por código a partir do prontuário. Na execução 2,
**4 dos 8 casos** tiveram o modelo classificando como conduta de rotina
situações que a regra identificou como urgência:

| Paciente | Quadro | Rótulo do modelo | Decisão da regra |
|---|---|---|---|
| PAC-004 | AVC em janela terapêutica, NIHSS 8 | `SUGERIR_CONDUTA` | `EMITIR_ALERTA` |
| PAC-005 | Cetoacidose diabética grave (pH 7,18) | `SUGERIR_CONDUTA` | `EMITIR_ALERTA` |
| PAC-006 | Pé diabético infectado, febre, protocolo de sepse | `SUGERIR_CONDUTA` | `EMITIR_ALERTA` |
| PAC-008 | Choque séptico, 8 sinais de gravidade | `SUGERIR_CONDUTA` | `EMITIR_ALERTA` |

Em todos os quatro, o rótulo era sintaticamente válido — passaria por qualquer
validação de formato. Se o roteamento dependesse do LLM, os quatro seguiriam o
caminho de menor urgência.

Nota sobre a piora entre execuções: a execução 1 registrou 2 discordâncias, a
execução 2 registrou 4. A causa não é degradação do modelo, e sim a ampliação
da detecção de gravidade, que aumentou o número de casos corretamente
classificados como urgência pela regra — e portanto o número de divergências
expostas.

### Guardrail em duas camadas

A ressalva de validação apareceu em 8/8 respostas finais nas duas execuções. Na
execução 2, todas espontâneas; na execução 1, uma precisou de inserção por
código.

A camada de código não é redundância. Em execução anterior, em CPU com
bfloat16, a taxa espontânea foi de 1/2. A taxa varia com o regime numérico, e é
a camada de código que torna a garantia independente disso.

Vale registrar uma variação de forma: nos pacientes 003 e 007 da execução 2, o
modelo embutiu a ressalva **dentro** da enumeração em vez de fechar a resposta
com ela. A verificação por palavra-chave detectou, mas a posição não é estável.

---

## 3. O que não funcionou

### O rótulo de decisão do modelo é inutilizável

Nas duas execuções somadas, 16 respostas produziram 6 rótulos sintaticamente
válidos e **zero clinicamente corretos**.

Falhas observadas na execução 2:

| Paciente | Primeira linha |
|---|---|
| PAC-002 | `DESFECHO: VERIFICAR ANÁLISE` |
| PAC-003 | `DESFECHO: VERIFICAR VALIDAÇÃO DO MÉDICO RESPONSÁVEL` |
| PAC-007 | `DESFECHO: AVALIAR` |
| PAC-004, 005, 006, 008 | `DESFECHO: SUGERIR_CONDUTA` (válido, clinicamente errado) |
| PAC-001 | nenhum rótulo |

Na execução 1, os rótulos inválidos foram outros: `VERIFICAR`, `VERIFICAR
CONTA`, `AVALIAR`. Em CPU com bfloat16, o modelo produziu `SUGERIR CONDUÇÃO`.

Os rótulos inválidos não se repetem entre execuções — cada rodada inventa
variações próprias. Isso indica que o problema não é um modo de falha
específico a ser corrigido, mas ausência de aderência confiável ao formato.

Conclusão: um fine-tuning de 95 exemplos sobre um modelo de 3B não produz
aderência de formato confiável. A decisão de mover o roteamento para código não
foi preferência de estilo; foi consequência da medição.

### O modelo distorce o conteúdo do protocolo recuperado

O erro mais grave do sistema, e o mais difícil de tratar.

**Execução 1, PAC-008** — com o `PROT-001` corretamente recuperado e presente
no contexto:

> "O protocolo prevê antibioticoterapia empírica **antes** da coleta de
> hemocultura."

O protocolo determina o oposto, e o texto estava no contexto fornecido. A
própria resposta se contradizia no item seguinte.

**Execução 2, PAC-006** — o mesmo tipo de erro, em outro paciente:

> "1. **Suspender** antibioticoterapia empírica conforme PROT-001 antes de
> coletar hemocultura.
> 2. Coletar hemocultura antes do antibiótico conforme PROT-001.
> 3. Coletar lactato sérico **antes da hemocultura** conforme PROT-001."

Três defeitos: "suspender" onde o protocolo manda iniciar; contradição direta
entre os itens 1 e 2; e uma ordem de precedência entre lactato e hemocultura
que o protocolo não estabelece.

**O achado relevante é a natureza estocástica do erro.** O PAC-008, que errou
na execução 1, acertou na execução 2 ("Coleta de hemocultura antes do
antibiótico conforme Protocolo PROT-001"). O PAC-006, que não apresentou o erro
na execução 1, apresentou na 2.

Com `temperatura=0.3`, o mesmo prompt produz respostas diferentes entre
execuções, e o erro de fidelidade ao contexto **muda de lugar**. Isso é pior que
um erro reprodutível: não há caso específico a corrigir, e a taxa de ~1 em 8 se
distribui de forma imprevisível.

Reduzir a temperatura a zero tornaria as respostas determinísticas, mas não há
evidência de que eliminaria o erro — apenas de que o fixaria num lugar.

### Posologia inventada

Execução 1, PAC-003, sem base em nenhum protocolo recuperado:

> "A dose inicial pode ser 100 mg enoxaparina dupla via (20 mg cada viço) por 24
> horas, seguida de dose única de 40 mg dupla via."

A dose não consta do `PROT-011`, que trata de indicação e contraindicação sem
especificar posologia. "Viço" é corrupção de token.

Na execução 2 o mesmo paciente não apresentou o erro — outra manifestação da
variabilidade descrita acima.

### Corrupção de tokens

Ao longo das execuções: `SUGERIR CONDUÇÃO`, `efoxaparina` (por enoxaparina),
`viço`, `PROTO-019 PROT-019`, `monitorizar contínua`. São palavras quase
corretas, com trocas plausíveis.

Sugere que parte do problema é capacidade do modelo de 3B, não apenas volume de
treino.

---

## 4. Correções aplicadas durante a avaliação

Dois defeitos do próprio sistema foram encontrados pela avaliação e corrigidos.
Registrados aqui porque a fragilidade das soluções é relevante.

### Detecção de gravidade limitada aos sinais vitais

Na execução 1, o PAC-002 — IAMCSST com supradesnivelamento de ST em três
derivações e troponina 3,8 ng/mL (referência < 0,04) — foi roteado como
`SUGERIR_CONDUTA`, porque nenhum sinal vital estava alterado. A gravidade
estava no eletrocardiograma.

A detecção foi ampliada para ler três fontes: sinais vitais, resultados de
exames (achados críticos em texto livre e limiares numéricos de troponina e
lactato) e protocolos de urgência associados ao paciente no prontuário.

Na execução 2 o PAC-002 dispara alerta por três vias independentes.

A limitação de desenho permanece: a heurística cobre o que foi explicitamente
previsto. Um quadro grave que não se manifeste nos sinais vitais nem nos
achados listados continuaria passando.

### Falso positivo por cegueira a negação

A busca por achados críticos em texto livre de laudo é feita por substring. O
laudo do PAC-007 — "Sem hemorragia ou isquemia aguda" — gerava alerta de achado
hemorrágico, invertendo o significado do documento.

Corrigido com verificação de termos de negação numa janela anterior ao achado,
validada em 8 casos de teste.

A solução é frágil por natureza: análise de negação em texto clínico livre é
problema aberto, e uma formulação não prevista volta a produzir o erro.

### Falso positivo na verificação do guardrail

A verificação era feita sobre o texto final montado, que inclui o bloco de ações
recomendadas. A ação "Acionar imediatamente o médico responsável" contém as
mesmas palavras da ressalva de validação, então o detector concluía que o aviso
estava presente e não o inseria — produzindo respostas com sugestão de conduta
sem aviso de segurança.

Corrigido avaliando o guardrail apenas sobre o texto do modelo, isolado das
ações. Coberto por teste de regressão.

---

## 5. Interpretação

Os resultados, agora com baseline, separam com clareza o que cada camada
entrega.

**O fine-tuning entrega forma.** A comparação com o modelo base é direta: a
ressalva de validação passou de 0/2 para 8/8. O formato enumerado e a citação
estruturada aparecem nos dois, mas o comportamento de segurança é atribuível ao
treino.

**O fine-tuning não entrega substância.** Em 16 respostas, houve 4 erros de
conteúdo clínico — inversão de protocolo, posologia inventada, classificação de
gravidade equivocada. E o erro vem acompanhado de citação formalmente correta,
o que o torna mais difícil de detectar, não menos.

**A citação de fonte é efeito do RAG.** O modelo base cita protocolos
corretamente quando eles estão no contexto. O fine-tuning ensinou o formato da
citação, não a capacidade de citar.

**O rótulo de decisão é o único comportamento treinado que falhou
completamente.** Zero acertos clínicos em 6 rótulos válidos, ao longo de duas
execuções.

### Reposicionamento das camadas

| Camada | O que garante | O que não garante |
|---|---|---|
| Fine-tuning | Presença da ressalva de validação | Correção do conteúdo |
| RAG | Que a fonte certa esteja no contexto | Que o modelo a use corretamente |
| Grafo determinístico | Decisão de fluxo auditável | Correção do texto gerado |
| Validação humana | — | É a única camada que cobre o conteúdo |

A exigência de validação humana antes de qualquer conduta, que o enunciado
apresenta como requisito de segurança, não é formalidade de conformidade. Nas
execuções medidas, é a única camada capaz de interceptar as distorções do
`PROT-001` antes que cheguem ao paciente.

---

## 6. Limitações do estudo

**Escala.** 8 pacientes, 14 protocolos, duas execuções. Os números indicam
tendência, não significância estatística. Uma taxa de 4 erros clínicos em 16
respostas tem intervalo de confiança largo demais para comparação com qualquer
referência.

**Baseline parcial.** A comparação com o modelo base cobriu 2 dos 8 pacientes,
não os 8. A atribuição do guardrail ao fine-tuning se sustenta (0/2 contra 8/8 é
uma diferença grande), mas as demais comparações são indicativas.

**Dados sintéticos.** Protocolos, prontuários, doses e códigos são fictícios,
gerados por LLM e revisados por amostragem, sem validação clínica. A avaliação
mede o comportamento do sistema sobre esses dados, não desempenho clínico.

**Curadoria assumida correta.** O escopo de recuperação depende do campo
`protocolos_relacionados`, atribuído manualmente na construção da base. Num
sistema real, essa atribuição precisaria ser validada, e um erro nela propagaria
sem sinal de alerta.

**Avaliação por padrão textual.** As métricas de guardrail e citação verificam
presença de padrão por palavra-chave. Medem se a ressalva está no texto, não se
é adequada ao conteúdo nem se está na posição correta; medem se há código citado,
não se a citação sustenta a afirmação feita.

**Correção clínica não medida sistematicamente.** Os erros de conteúdo foram
encontrados por leitura das respostas, não por método. Não há métrica automática
de fidelidade ao protocolo recuperado, e construí-la exigiria anotação por
profissional de saúde.

**Variabilidade não caracterizada.** Duas execuções mostraram que o erro de
fidelidade muda de paciente entre rodadas. Caracterizar a distribuição desse
erro exigiria dezenas de execuções, o que não foi feito.

---

## 7. O que faria diferente

**Fidelidade ao contexto como métrica automática.** Verificar se as afirmações
da resposta estão sustentadas pelo trecho recuperado. Teria detectado as duas
distorções do `PROT-001` sem depender de leitura manual, e permitiria medir a
taxa ao longo de muitas execuções.

**Extração estruturada em vez de geração livre para condutas.** Em vez de o
modelo escrever a conduta, fazê-lo selecionar itens do protocolo recuperado.
Reduz expressividade e elimina a classe de erro mais grave — inversão e invenção
de conteúdo.

**Temperatura zero com avaliação de múltiplas execuções.** A temperatura de 0.3
introduz variabilidade que dificulta reprodução e medição. Fixá-la em zero
tornaria os erros reprodutíveis e localizáveis, ainda que não os eliminasse.

**Modelo maior ou dataset maior.** 3B com 95 exemplos é a menor configuração que
demonstra o pipeline. As corrupções de token observadas sugerem que parte do
problema é capacidade, não só volume de treino.

**Anotação clínica de um subconjunto.** Sem profissional de saúde avaliando as
respostas, não há como afirmar nada sobre utilidade clínica — apenas sobre
comportamento do sistema.
