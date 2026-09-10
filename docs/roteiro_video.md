# Roteiro do vídeo — Tech Challenge Fase 3

Limite: 15 minutos. O roteiro abaixo ocupa ~13, deixando margem.

O enunciado pede demonstrar quatro coisas: treinamento e funcionamento da LLM
personalizada, execução de um fluxo automatizado, resposta a perguntas clínicas
contextualizadas, e logs e validação das respostas. Cada uma tem um bloco.

---

## Antes de gravar

**Deixe pronto e aberto em abas:**

1. `notebooks/03_demo_assistente.ipynb` no Colab, **já executado até a seção 6**
   (modelo carregado). Isso poupa ~5 minutos de instalação e download na
   gravação.
2. Repositório no GitHub
3. Página do modelo no Hugging Face
4. `notebooks/02_finetuning.ipynb` no Colab, com os outputs do treino visíveis
5. `docs/resultados/curva_loss.png` aberto

**Teste antes:** rode a seção 7 uma vez para confirmar que o modelo responde. Se
a sessão do Colab tiver expirado, a demonstração ao vivo falha.

**Cuidado:** não mostre as chaves de API nos Secrets do Colab em nenhum momento.

---

## Bloco 1 — Abertura e contexto (1 min)

**Mostrar:** README no GitHub.

**Dizer:**

> Assistente clínico de apoio à decisão, construído em três camadas: um modelo
> com fine-tuning, recuperação de protocolos internos com LangChain, e um fluxo
> de decisão com LangGraph.
>
> Aviso importante: todos os dados são sintéticos. Protocolos, doses e pacientes
> são fictícios e não servem para uso clínico.
>
> Vou mostrar o treinamento do modelo, o assistente funcionando de ponta a
> ponta, e — o que considero o resultado mais interessante — as duas decisões de
> arquitetura que tomamos depois de medir onde o modelo era confiável e onde não
> era.

Role o README até a seção de arquitetura, mostre o diagrama do grafo.

---

## Bloco 2 — Fine-tuning (3 min)

### 2.1 Os dados (1 min)

**Mostrar:** `notebooks/01_gerar_dataset.ipynb`, seção de diagnóstico.

**Dizer:**

> O desafio pede dados próprios do hospital. Os datasets públicos sugeridos são
> literatura médica, então geramos 95 exemplos sintéticos simulando documentos
> internos: protocolos, dúvidas de médicos, modelos de laudo e receita, e
> cenários clínicos com dados de paciente.
>
> O pipeline tem preprocessing, anonimização em duas camadas e curadoria com
> revisão humana por amostragem.

Mostre a saída da célula de diagnóstico: 95 exemplos, distribuição por
categoria, 100% com citação de fonte.

**Ponto que vale destacar:**

> A ressalva de validação humana aparece em 60% dos exemplos, e isso é
> intencional. Ela está onde há sugestão de conduta e é omitida em perguntas
> informativas. Se aparecesse em 100%, o modelo aprenderia a repetir a frase por
> reflexo, não a reconhecer quando a validação é necessária.

### 2.2 O treino (2 min)

**Mostrar:** `notebooks/02_finetuning.ipynb`, seções 5, 9 e 10.

**Dizer:**

> Modelo base: Qwen2.5-3B-Instruct. O enunciado sugere LLaMA, e foi a primeira
> escolha — mas o Llama 3.2 exige aprovação individual da Meta e retornou erro
> 403. Trocamos por um modelo aberto da mesma faixa de parâmetros, porque
> depender de uma aprovação com prazo indeterminado quebraria a reprodutibilidade
> do projeto.
>
> A técnica é QLoRA: modelo base em 4 bits, e só as matrizes LoRA são treinadas.
> É o que permite treinar um modelo de 3 bilhões de parâmetros numa GPU gratuita.

Mostre a curva de perda (`curva_loss.png`).

> Uma decisão que vale mencionar: o split treino/avaliação é estratificado por
> categoria, não aleatório. Com 95 exemplos, uma divisão aleatória poderia deixar
> a avaliação sem nenhum cenário clínico — que é justamente o comportamento mais
> importante de medir.

Mostre a página do modelo no Hugging Face.

> Os adapters estão publicados e são públicos, então qualquer pessoa reproduz
> sem token de acesso.

---

## Bloco 3 — O assistente funcionando (4 min)

### 3.1 O fluxo (1 min)

**Mostrar:** seção 3 do notebook de demonstração, com o diagrama renderizado.

**Dizer:**

> Este diagrama é gerado a partir do grafo compilado, não desenhado à parte —
> então não pode divergir do código.
>
> O fluxo carrega o prontuário, recupera os protocolos, consulta o modelo, decide
> o desfecho e roteia para um de três caminhos: verificar exames pendentes,
> sugerir conduta, ou emitir alerta.

### 3.2 A recuperação (1,5 min)

**Mostrar:** seção 4 do notebook, célula de comparação livre vs escopo.

**Dizer:**

> Aqui está a primeira decisão que a medição mudou. Este paciente tem cefaleia
> intensa de início súbito com pressão de 210 por 130 — crise hipertensiva.
>
> A busca por similaridade de texto recupera cetoacidose, modelo de laudo
> laboratorial e tromboembolismo. Nada disso se aplica.
>
> A busca restrita aos protocolos que o prontuário indica traz o protocolo de
> crise hipertensiva. Só ele.

**O ponto importante:**

> Isso não é otimização de qualidade. Antes da correção, o modelo recebeu o
> protocolo de anafilaxia para um paciente com suspeita de tromboembolismo — e
> produziu conduta de anafilaxia, com adrenalina intramuscular, citando o
> protocolo corretamente. A fonte estava errada e a citação estava certa, o que
> é pior que uma citação inventada.

### 3.3 Execução ao vivo (1,5 min)

**Mostrar:** seção 7, rodando 2 ou 3 pacientes.

Escolha estes:

- **PAC-001** — exames pendentes, roteia para `VERIFICAR_EXAMES`
- **PAC-008** — sepse com 8 sinais de gravidade, roteia para `EMITIR_ALERTA`

**Dizer, apontando a saída:**

> Repare em três coisas na resposta. Primeiro, as ações recomendadas vêm da base
> estruturada, não do texto do modelo — os exames pendentes são os que estão no
> prontuário. Segundo, a ressalva de validação humana. Terceiro, as fontes
> consultadas, com código e versão do protocolo.
>
> E embaixo, o caminho percorrido no grafo, com o motivo da decisão: "2 exames
> sem resultado disponível".

---

## Bloco 4 — Segurança e validação (2,5 min)

**Mostrar:** seção 8 do notebook.

**Dizer:**

> O requisito de segurança pede três coisas: limites de atuação, logging e
> explainability.
>
> Explainability: 8 de 8 respostas citam fonte, e a citação vem dos documentos
> que o retriever devolveu — não do código que o modelo escreveu no texto. Essa
> distinção importa porque os códigos que o modelo produz não são confiáveis.
>
> Limites de atuação: a ressalva de validação aparece em 8 de 8 respostas
> finais. Sete o modelo emitiu sozinho; uma o código inseriu.

**Ponto que vale contar** (é um bug real, e mostra rigor):

> A verificação do guardrail tinha um defeito que só apareceu rodando o fluxo
> completo. Ela era feita sobre o texto final, que inclui as ações recomendadas.
> A ação "acionar imediatamente o médico responsável" contém as mesmas palavras
> da ressalva de validação — então o detector achava que o aviso já estava lá e
> não inseria. Resultado: resposta com sugestão de conduta saindo sem aviso de
> segurança. A correção foi verificar apenas o texto do modelo, isolado das
> ações.

**Mostrar:** seção 9, o registro JSONL e as estatísticas.

> Cada consulta grava um registro com pergunta, paciente, fontes com versão e
> índice de chunk, desfecho, motivo, caminho no grafo, duração. Append-only.
>
> Um campo interessante: `guardrail_adicionado`. Ele mede quantas vezes o modelo
> falhou no requisito de segurança e o código precisou intervir.

---

## Bloco 5 — O achado principal (2 min)

Este é o bloco que diferencia a apresentação. Não corte se o tempo apertar.

**Mostrar:** seção 8, tabela de concordância LLM vs regra.

**Dizer:**

> Aqui está o resultado mais importante do trabalho.
>
> O modelo foi treinado para emitir um rótulo de decisão na primeira linha:
> verificar exames, sugerir conduta ou emitir alerta. Em 8 execuções, nenhuma
> produziu um rótulo utilizável. Ele inventou "VERIFICAR", "VERIFICAR CONTA",
> "AVALIAR". Em outra execução, com precisão numérica diferente, produziu
> "SUGERIR CONDUÇÃO".
>
> E os dois casos em que o rótulo era sintaticamente válido são os mais
> preocupantes: ele classificou um AVC em janela terapêutica e uma cetoacidose
> diabética grave como conduta de rotina. Um rótulo válido e errado é mais
> perigoso que nenhum rótulo, porque passa por qualquer validação de formato.
>
> Por isso a decisão de fluxo não é do modelo. É de uma regra em código que lê
> exames pendentes e sinais de gravidade do prontuário. Nos dois casos em que o
> modelo errou, a regra corrigiu.

**Se houver tempo, o caso mais forte:**

> E há um erro que nenhuma arquitetura de RAG corrige. No paciente com sepse, com
> o protocolo correto recuperado e presente no contexto, o modelo escreveu que "o
> protocolo prevê antibioticoterapia empírica antes da coleta de hemocultura". O
> protocolo diz o oposto, e a própria resposta se contradiz na linha seguinte.
>
> A fonte estava certa. O modelo distorceu o que ela diz.

---

## Bloco 6 — Fechamento (0,5 min)

**Dizer:**

> Resumindo o que os números mostram: o fine-tuning entregou forma — 8 de 8 no
> formato, na citação de fonte, na ressalva de validação. E não entregou
> substância: 3 de 8 respostas têm erro clínico, sempre acompanhado de citação
> formalmente correta.
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
| 3. Assistente funcionando | 4:00 | Fluxo automatizado + perguntas contextualizadas |
| 4. Segurança e validação | 2:30 | Logs e validação das respostas |
| 5. Achado principal | 2:00 | — (diferencial) |
| 6. Fechamento | 0:30 | — |
| **Total** | **13:00** | |

---

## Se o tempo apertar

Corte nesta ordem:

1. Bloco 2.1 (geração de dados) — reduza a 30 s, é a parte menos visual
2. Bloco 3.3 — rode 2 pacientes em vez de 3
3. Bloco 5, segundo parágrafo (a inversão do protocolo)

**Não corte:** o bloco 5 inteiro, nem o defeito do guardrail no bloco 4. São as
partes que mostram medição e rigor, e o que diferencia a apresentação de uma
demonstração de funcionalidade.

---

## Dicas práticas

**Zoom.** Aumente a fonte do navegador para 125% ou 150% antes de gravar. Saída
de terminal em fonte pequena fica ilegível em vídeo comprimido.

**Não leia o roteiro.** As falas acima são o conteúdo, não o texto. Ler soa mal e
consome mais tempo que falar naturalmente.

**Rolagem lenta.** Ao mostrar código ou saída longa, role devagar e pare nos
pontos que está comentando.

**Uma tomada por bloco.** Gravar 13 minutos seguidos sem erro é difícil. Grave
bloco por bloco e junte na edição.

**Se algo falhar ao vivo**, comente com naturalidade e siga. Uma sessão do Colab
que expira no meio é acidente comum, e reconhecer é melhor que fingir que não
aconteceu.
