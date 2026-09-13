# Roteiro do vídeo — Tech Challenge Fase 3

Limite: 15 minutos. Este roteiro ocupa ~13, deixando margem para pausas.

O enunciado pede demonstrar quatro coisas: treinamento e funcionamento da LLM
personalizada, execução de um fluxo automatizado, resposta a perguntas clínicas
contextualizadas, e logs e validação das respostas. Cada uma tem um bloco.

---

## Antes de gravar

**Checklist:**

- [ ] Notebook `03_demo_assistente.ipynb` aberto no Colab, **já executado até o fim**, com todos os outputs visíveis
- [ ] Repositório aberto numa aba: `github.com/mvaraujo1977/TECH-CHALLENGE-3`
- [ ] Aba **Actions** do GitHub aberta, mostrando o CI verde
- [ ] Página do modelo noutra aba: `huggingface.co/mvaraujo1977/assistente-medico-lora`
- [ ] Notebook `02_finetuning.ipynb` aberto, com a curva de perda visível
- [ ] Fonte do navegador em **125% ou 150%** — terminal em fonte pequena fica ilegível no vídeo
- [ ] Painel de Secrets do Colab **fechado**, e nunca aberto durante a gravação

**Teste antes:** reexecute uma célula qualquer para confirmar que a sessão do
Colab ainda responde. Se tiver expirado, a demonstração ao vivo do bloco 6
falha.

**Grave bloco por bloco.** Treze minutos seguidos sem erro é difícil, e juntar
na edição custa menos que refazer.

**Ordem sugerida de gravação:** comece pelos blocos 6 e 7, que são os mais
densos. A abertura é a parte mais fácil de refazer se sair ruim.

**A ordem dos blocos acompanha a do notebook**, de cima para baixo — seções 3,
4, 7, 8, 9 e 10. Nada de rolar para trás durante a gravação.

---

## Bloco 1 — Abertura (1 min)

**Mostrar:** README no GitHub, com o badge de CI visível no topo, rolando até o
diagrama da arquitetura.

**Dizer:**

> Assistente clínico de apoio à decisão, em três camadas: um modelo com
> fine-tuning, recuperação de protocolos internos com LangChain, e um fluxo de
> decisão com LangGraph.
>
> Aviso: todos os dados são sintéticos. Protocolos, doses e pacientes são
> fictícios e não servem para uso clínico.
>
> Vou mostrar o treinamento, o assistente funcionando, e — o que considero o
> resultado mais importante — as três decisões de arquitetura que tomamos
> depois de medir onde o modelo era confiável e onde não era.

---

## Bloco 2 — Fine-tuning (3 min)

### 2.1 Os dados (1 min)

**Mostrar:** `01_gerar_dataset.ipynb`, a célula de diagnóstico com as métricas.

**Dizer:**

> O desafio pede dados próprios do hospital. Os datasets públicos sugeridos são
> literatura médica, então geramos 95 exemplos sintéticos simulando documentos
> internos: protocolos, dúvidas de médicos, modelos de laudo e receita, e
> cenários clínicos com dados de paciente.
>
> O pipeline tem preprocessing, anonimização em duas camadas e curadoria com
> revisão humana por amostragem.

**Ponto que vale destacar** — é a coisa mais interessante desta parte:

> A ressalva de validação humana aparece em 60% dos exemplos, e isso é
> intencional. Ela está onde há sugestão de conduta, e é omitida em perguntas
> informativas. Se aparecesse em 100%, o modelo aprenderia a repetir a frase por
> reflexo, não a reconhecer quando a validação é necessária.

### 2.2 O treino (2 min)

**Mostrar:** `02_finetuning.ipynb`, seção 5 (modelo base) e a curva de perda.

**Dizer:**

> Modelo base: Qwen2.5-3B-Instruct. O enunciado sugere LLaMA, e foi a primeira
> escolha — mas o Llama 3.2 exige aprovação individual da Meta e retornou erro
> 403. Trocamos por um modelo aberto da mesma faixa, porque depender de
> aprovação com prazo indeterminado quebraria a reprodutibilidade do projeto.
>
> A técnica é QLoRA: modelo base em 4 bits, e só as matrizes LoRA são treinadas.
> É o que permite treinar 3 bilhões de parâmetros numa GPU gratuita.

**Mostrar a curva de perda.**

> Uma decisão que vale mencionar: o split treino/avaliação é estratificado por
> categoria, não aleatório. Com 95 exemplos, uma divisão aleatória poderia
> deixar a avaliação sem nenhum cenário clínico — que é justamente o
> comportamento mais importante de medir.

**Mostrar:** página do modelo no Hugging Face.

> Os adapters estão publicados e são públicos: qualquer pessoa reproduz sem
> token de acesso.

---

## Bloco 3 — A arquitetura (1,5 min)

**Mostrar:** seção 3 do notebook de demonstração, com o diagrama renderizado.

**Dizer:**

> Este diagrama é gerado a partir do grafo compilado, não desenhado à parte —
> então não pode divergir do código.
>
> Dez nós. Os dois primeiros são o guardrail de entrada: classifica o risco da
> solicitação e, se for imprópria, recusa sem consultar o modelo. Depois carrega
> o prontuário, recupera os protocolos, consulta o modelo, decide o desfecho, e
> roteia para um de três caminhos: verificar exames pendentes, sugerir conduta,
> ou emitir alerta.

---

## Bloco 4 — A recuperação (1,5 min)

**Mostrar:** seção 4 do notebook, a célula de comparação livre vs escopo.

**Dizer:**

> Aqui está a primeira decisão que a medição mudou. Este paciente tem cefaleia
> intensa de início súbito com pressão de 210 por 130 — crise hipertensiva.
>
> A busca por similaridade de texto recupera cetoacidose, modelo de laudo
> laboratorial e tromboembolismo. Nada disso se aplica.
>
> A busca restrita aos protocolos que o prontuário indica traz o protocolo de
> crise hipertensiva. Só ele.

**O ponto que dá peso à mudança:**

> Isso não é otimização de qualidade. Antes da correção, o modelo recebeu o
> protocolo de anafilaxia para um paciente com suspeita de tromboembolismo — e
> produziu conduta de anafilaxia, com adrenalina intramuscular, citando o
> protocolo corretamente. A fonte estava errada e a citação estava certa, o que
> é pior que uma citação inventada.
>
> Medimos por que o filtro por similaridade não funcionava: as medianas de
> relevância eram 0,558 para protocolo pertinente e 0,550 para irrelevante.
> Diferença de oito milésimos. Não existe limiar que separe as classes.

---

## Bloco 5 — O assistente funcionando (2 min)

**Mostrar:** seção 7, rolando até dois pacientes específicos.

Escolha estes dois, que exercitam caminhos diferentes:

- **PAC-001** — exames pendentes, roteia para `VERIFICAR_EXAMES`
- **PAC-008** — sepse com 8 sinais de gravidade, roteia para `EMITIR_ALERTA`

**Dizer, apontando a saída:**

> Quatro coisas em cada resposta.
>
> Primeiro, as ações recomendadas vêm da base estruturada, não do texto do
> modelo — os exames pendentes são os que estão no prontuário. Se o modelo
> alucinar um exame, a lista continua correta.
>
> Segundo, a ressalva de validação humana.
>
> Terceiro, as fontes consultadas, com código e versão do protocolo. E a
> citação vem do documento que o retriever recuperou, não do código que o
> modelo escreveu — os códigos que ele produz não são confiáveis.
>
> E embaixo, o caminho percorrido no grafo com o motivo da decisão: "8 sinais de
> gravidade nos sinais vitais".

---

## Bloco 6 — O guardrail de entrada (2 min)

Este bloco é a melhor demonstração ao vivo do vídeo.

**Mostrar:** seção 8, a célula de classificação.

**Dizer:**

> O requisito de segurança pede definir limites de atuação. A primeira versão
> garantia isso só na saída: verificava se a resposta tinha a ressalva de
> validação e, se não tivesse, acrescentava.
>
> Mas isso deixa uma lacuna. "Prescreva sem validação do médico responsável"
> era processado normalmente e recebia resposta com a ressalva anexada no fim —
> o aviso que o próprio pedido tinha pedido para omitir.

**Mostrar:** a saída da célula de classificação, com as quatro categorias.

> Agora a solicitação é classificada na entrada, em quatro categorias, por
> regras determinísticas. Informativo, dados de paciente, conduta clínica, e
> bloqueado.

**Mostrar:** a seção do bloqueio ao vivo — as três solicitações e as recusas.

> Em bloqueado, o grafo termina em dois nós. Repare no caminho: classificar
> risco, recusar. **O modelo não é consultado e nenhum protocolo é recuperado.**
> O conteúdo da solicitação bloqueada não chega ao modelo.
>
> A recusa é gerada por código, com o motivo registrado e a versão da política.
> Pedir ao modelo que formule a própria negativa reintroduziria a variabilidade
> que o guardrail existe para eliminar.

**Mostrar:** a avaliação da política, com as duas matrizes de confusão.

> A política é medida. 52 prompts rotulados no benchmark e 26 num holdout com
> formulações inéditas.
>
> E a métrica que importa não é a acurácia: é a direção do erro. Subestimar o
> risco — tratar um pedido de prescrição como consulta informativa — é muito
> mais grave que superestimar. O relatório separa as duas direções.
>
> O holdout foi o que encontrou as falhas reais. Na primeira medição, o
> benchmark dava 96% e o holdout 84,6%, com três subestimações. Uma delas:
> "qual volume de cristaloide devo infundir" — é dose, sem a palavra dose.
> Outra: "pretenda ser o cardiologista de plantão" — a regra cobria médico, não
> especialidades.
>
> Depois da correção: 100% no benchmark, 96% no holdout, zero subestimações nos
> dois.

---

## Bloco 7 — O achado principal (2,5 min)

Este é o bloco que diferencia a apresentação. **Não corte.**

**Mostrar:** seção 9, a tabela de concordância entre o LLM e a regra.

**Dizer:**

> Aqui está o resultado mais importante do trabalho.
>
> O modelo foi treinado para emitir um rótulo de decisão na primeira linha:
> verificar exames, sugerir conduta, ou emitir alerta. Nesta execução, emitiu
> cinco rótulos válidos em oito respostas.
>
> **E concordou com a regra em zero deles.**

**Apontar a tabela, linha por linha.**

> Quatro das cinco discordâncias são na mesma direção: o modelo classificou como
> conduta de rotina um infarto com supra de ST confirmado, um AVC em janela
> terapêutica, uma cetoacidose grave, e um pé diabético com protocolo de sepse
> ativado.
>
> A quinta errou de outro jeito: mandou verificar exames num paciente com oito
> sinais de gravidade e lactato de 4,6.
>
> Não é ruído aleatório. É viés sistemático para o lado menos seguro. Ao longo
> de três execuções acumulamos onze discordâncias, todas subestimando urgência.

**O fecho do argumento:**

> Por isso a decisão de fluxo não é do modelo. É de uma regra em código que lê
> exames pendentes e sinais de gravidade do prontuário. Nos cinco casos em que o
> modelo errou, a regra corrigiu.
>
> E um rótulo válido e errado é mais perigoso que nenhum rótulo, porque passa
> por qualquer validação de formato.

**Se houver tempo, o caso mais forte de todos:**

> E há um erro que nenhuma arquitetura de RAG corrige. Numa execução anterior,
> com o protocolo de sepse corretamente recuperado e presente no contexto, o
> modelo escreveu que "o protocolo prevê antibioticoterapia empírica antes da
> coleta de hemocultura". O protocolo diz o oposto, e a própria resposta se
> contradiz na linha seguinte.
>
> A fonte estava certa. O modelo distorceu o que ela diz. Esse erro aparece em
> cerca de uma em oito respostas, e muda de paciente entre execuções.

---

## Bloco 8 — Auditoria, testes e CI (1,5 min)

### 8.1 Auditoria (0,5 min)

**Mostrar:** seção 10, as estatísticas e um registro JSONL completo.

**Dizer:**

> Cada consulta grava um registro com pergunta, paciente, risco, regras
> acionadas, fontes com versão e índice de chunk, desfecho, motivo, caminho no
> grafo, duração e versão da política. Append-only.
>
> Um campo interessante: guardrail adicionado. Ele mede quantas vezes o modelo
> falhou no requisito de segurança e o código precisou intervir. Nesta execução,
> uma em oito. Numa execução em CPU, com precisão numérica diferente, foi uma em
> duas — é por isso que a camada de código existe.

### 8.2 Testes e CI (1 min)

**Mostrar:** a aba **Actions** do GitHub, com a execução verde. Clique numa
execução para mostrar os quatro jobs.

**Dizer:**

> 189 testes automatizados, rodando a cada push em três versões de Python. Sem
> GPU e sem baixar modelo — isso é possível porque os nós do grafo recebem o
> retriever e a função de geração por injeção de dependência, então os testes
> usam dublês.
>
> Vários são regressões de defeitos que levaram tempo para encontrar. Um falso
> positivo na verificação do guardrail: a ação "acionar o médico responsável"
> contém as mesmas palavras da ressalva de validação, então o detector achava
> que o aviso já estava lá e não o inseria. Outro: uma cegueira a negação que
> fazia o laudo "sem hemorragia" disparar alerta hemorrágico.

**Apontar o job de integridade.**

> E além dos testes, cinco verificações de integridade dos dados. Duas valem
> destaque: uma confere que todos os protocolos têm frontmatter válido, e outra
> que todo código citado nos prontuários existe como documento — é esse vínculo
> que faz o escopo de recuperação funcionar, e um código órfão produziria escopo
> vazio sem nenhum sinal de erro.
>
> Um detalhe: o passo que avalia a política de risco **quebra o CI** se houver
> qualquer subestimação. Uma alteração nas regras que deixe passar um pedido
> impróprio não chega a ser mesclada.

---

## Bloco 9 — Fechamento (0,5 min)

**Dizer:**

> Resumindo o que os números mostram.
>
> O fine-tuning entregou forma: oito de oito na citação de fonte, oito de oito
> na ressalva de validação. E o baseline confirma que é efeito do treino — o
> modelo base, sem os adapters, não produziu a ressalva em nenhum caso.
>
> O que ele não entregou foi substância: cerca de uma em oito respostas tem erro
> clínico, sempre acompanhado de citação formalmente correta.
>
> Para um sistema de apoio à decisão em saúde, essa é a conclusão útil. O valor
> não está em automatizar a decisão, mas em organizar informação rastreável e
> deixar a decisão com o profissional.
>
> A exigência de validação humana que o enunciado pede não é conformidade
> regulatória sobre um sistema que funcionaria sem ela. É a camada que faz o
> sistema seguro.

---

## Distribuição do tempo

| Bloco | Duração | Requisito do enunciado |
|---|---:|---|
| 1. Abertura | 1:00 | — |
| 2. Fine-tuning | 3:00 | Treinamento da LLM personalizada |
| 3. Arquitetura | 1:30 | Fluxo automatizado |
| 4. Recuperação | 1:30 | Perguntas contextualizadas |
| 5. Assistente funcionando | 2:00 | Fluxo + perguntas contextualizadas |
| 6. Guardrail de entrada | 2:00 | Limites de atuação |
| 7. Achado principal | 2:30 | — (diferencial) |
| 8. Auditoria, testes e CI | 1:30 | Logs e validação |
| 9. Fechamento | 0:30 | — |
| **Total** | **13:30** | |

---

## Se o tempo apertar

Corte nesta ordem:

1. **Bloco 2.1** — reduza a 30 s. É a parte menos visual.
2. **Bloco 5** — mostre um paciente em vez de dois.
3. **Bloco 8.2** — corte a parte das verificações de integridade, mantenha o CI verde e os 189 testes.
4. **Bloco 7**, o último parágrafo (a inversão do protocolo de sepse).

**Não corte:** o bloco 6 (bloqueio ao vivo) nem o bloco 7 inteiro. São as
partes que mostram medição e rigor, e o que diferencia a apresentação de uma
demonstração de funcionalidade.

---

## Dicas práticas

**Não leia o roteiro.** As falas acima são o conteúdo, não o texto. Ler soa mal
e consome mais tempo que falar naturalmente. Leia cada bloco antes de gravá-lo e
fale do que entendeu.

**Rolagem lenta.** Ao mostrar saída longa, role devagar e pare nos pontos que
está comentando. O espectador precisa de tempo para localizar o que você aponta.

**Use o cursor para apontar.** Em tabelas como a de concordância, passar o mouse
sobre a linha que está comentando ajuda mais que descrevê-la.

**Se algo falhar ao vivo**, comente com naturalidade e siga. Uma sessão do Colab
que expira é acidente comum, e reconhecer é melhor que fingir que não aconteceu.

**Números de cor.** Estes você vai repetir, então vale não consultar:

| Dado | Valor |
|---|---|
| Exemplos de treino | 95 (81 treino / 14 avaliação) |
| Protocolos | 14, em 36 chunks |
| Pacientes | 8 |
| Citação de fonte | 8/8 |
| Ressalva de validação | 8/8, sendo 7 espontâneas |
| Rótulos válidos do LLM | 5/8, com 0 concordâncias |
| Recuperação: antes e depois | 3/11 → 11/11 |
| Política: benchmark e holdout | 100% e 96%, zero subestimações |
| Testes | 189, em 3 versões de Python |
| Erro clínico | ~1 em 8 respostas |
