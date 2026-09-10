# Bases de dados

> ⚠️ **Todos os dados são sintéticos**, criados para fins acadêmicos. Protocolos, códigos,
> doses, prazos e pacientes são fictícios e **não devem ser usados clinicamente**.

Três bases distintas, com papéis diferentes na arquitetura:

| Base | Papel | Consumida por |
|---|---|---|
| `dataset_medico.jsonl` | Treino do modelo (comportamento e formato) | Fine-tuning — já concluído |
| `protocolos/` | Conhecimento factual recuperável | RAG (LangChain) |
| `prontuarios.json` | Dados do paciente em tempo de execução | Nós do grafo (LangGraph) |

## `protocolos/` — base do RAG

14 documentos markdown com frontmatter YAML (`codigo`, `titulo`, `versao`, `setor`, `revisao`).
O frontmatter alimenta os metadados dos chunks no vector store, permitindo citar a fonte
exata na resposta (requisito de explainability).

| Código | Tema |
|---|---|
| PROT-001 | Sepse e choque séptico |
| PROT-002 | Dor torácica e síndrome coronariana aguda |
| PROT-003 | AVC agudo (Código AVC) |
| PROT-004 | Avaliação pré-operatória eletiva |
| PROT-007 | Tromboembolismo pulmonar |
| PROT-009 | Cetoacidose diabética |
| PROT-011 | Tromboprofilaxia |
| PROT-012 | Controle glicêmico do internado |
| PROT-019 | Crise hipertensiva |
| PROT-023 | Prescrição de medicamentos controlados |
| PROT-025 | Precauções e isolamento (CCIH) |
| PROT-031 | Anafilaxia em adultos |
| LAUDO-IMG-01 | Modelo de laudo de imagem |
| LAUDO-LAB-05 | Modelo de laudo laboratorial |

### Nota sobre os códigos

Os códigos citados pelo **modelo fine-tuned** são inconsistentes: durante a geração do dataset,
o mesmo código foi associado a temas diferentes em lotes distintos (`PROT-012` aparece como TEP,
checklist de alta e controle glicêmico).

Isso é esperado — o fine-tuning ensinou o *formato* de citar uma fonte, não um mapeamento
código→conteúdo. **A explainability do sistema não usa o código citado pelo modelo**, e sim o
documento efetivamente recuperado pelo retriever, que é rastreável.

## `prontuarios.json` — base estruturada

8 pacientes fictícios, cada um com identificação, admissão, antecedentes, alergias,
medicamentos, sinais vitais e lista de exames com status (`resultado_disponivel`,
`pendente`, `coletado_aguardando`).

Os pacientes foram construídos para exercitar os três desfechos do grafo de decisão:

| Paciente | Perfil | Desfecho esperado |
|---|---|---|
| PAC-001 | TEP suspeito, D-dímero e angiotomografia pendentes | `VERIFICAR_EXAMES` |
| PAC-002 | IAMCSST inferior confirmado por ECG e troponina | `EMITIR_ALERTA` |
| PAC-003 | Hiperglicemia pós-operatória, exames completos | `SUGERIR_CONDUTA` |
| PAC-004 | AVC em janela terapêutica, TC pendente | `EMITIR_ALERTA` |
| PAC-005 | Cetoacidose grave, exames completos | `SUGERIR_CONDUTA` |
| PAC-006 | Pé diabético infectado, culturas pendentes | `VERIFICAR_EXAMES` |
| PAC-007 | Emergência hipertensiva, exames completos | `SUGERIR_CONDUTA` |
| PAC-008 | Sepse com hipotensão e lactato elevado | `EMITIR_ALERTA` |

O campo `protocolos_relacionados` liga cada paciente aos documentos pertinentes, útil para
validar se o retriever está recuperando o protocolo correto.
