"""Smoke test do fluxo completo, com o modelo real.

Roda dois casos que exercitam ramos diferentes do grafo:

    PAC-001  exames pendentes, sem sinal de gravidade  -> verificar_exames
    PAC-008  sinais de gravidade + exame pendente      -> emitir_alerta

O objetivo não é avaliar qualidade clínica, e sim responder quatro perguntas
sobre o comportamento do sistema montado:

    1. o modelo emite o rótulo "DESFECHO:" quando recebe dados de paciente?
    2. ele cita o protocolo recuperado pelo RAG ou inventa um código?
    3. o guardrail sai do modelo ou precisou ser inserido por código?
    4. o registro em logs/*.jsonl sai no formato esperado?

Uso:

    python scripts/demo.py              # os dois casos no mesmo processo
    python scripts/demo.py PAC-001      # só um caso

Rodar um caso por processo é a saída em máquina com pouca RAM: o modelo de 3B
em bfloat16 ocupa ~6 GB, e cada processo começa com a memória limpa.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import config  # noqa: E402
from src.graph.fluxo import criar_assistente  # noqa: E402
from src.llm import modelo as m  # noqa: E402

# bfloat16 na CPU: concessão de capacidade, não de desempenho — o 3B em float32
# não cabe na RAM desta máquina. O nome tem de ser o atributo do torch
# ("bfloat16"), não o apelido "bf16", que `carregar_modelo` rejeita.
DTYPE_CPU = "bfloat16"

# Reduzido em relação aos 400 do config: em CPU cada token custa caro, e as
# quatro perguntas acima se respondem com a primeira parte da resposta.
MAX_NEW_TOKENS = 200

CASOS = [
    ("PAC-001", "Qual a conduta para este paciente?"),
    ("PAC-008", "Qual a conduta para este paciente?"),
]

# Códigos de protocolo como aparecem na base: PROT-007. O segmento de letras no
# meio é opcional para tolerar variações do tipo PROT-CARD-001.
PADRAO_CODIGO = re.compile(r"\b[A-Z]{3,}(?:-[A-Z]{2,})*-\d+\b")

SEPARADOR = "=" * 70


def texto_do_modelo(resposta_final: str) -> str:
    """Isola o trecho que veio do modelo, antes dos blocos montados por código.

    `finalizar` concatena: texto do modelo, ações, guardrail e fontes. Cortar no
    primeiro bloco montado devolve apenas o que o LLM escreveu — que é o que
    precisa ser inspecionado para as perguntas 1, 2 e 3.
    """
    return resposta_final.split("**Ações recomendadas:**")[0].strip()


def relatar(id_paciente: str, pergunta: str, assistente) -> dict:
    print(f"\n{SEPARADOR}\n{id_paciente} — {pergunta}\n{SEPARADOR}")

    resposta, registro = assistente.consultar(pergunta, id_paciente=id_paciente)
    bruto = texto_do_modelo(resposta)

    codigos_rag = [f["codigo"] for f in registro.fontes]
    codigos_citados = sorted(set(PADRAO_CODIGO.findall(bruto)))
    inventados = [c for c in codigos_citados if c not in codigos_rag]

    print("\n--- resposta final ---")
    print(resposta)

    print("\n--- diagnóstico ---")
    print(f"  desfecho             : {registro.desfecho}")
    print(f"  veio do modelo       : {registro.desfecho_do_modelo}")
    print(f"  primeira linha       : {bruto.splitlines()[0] if bruto else '(vazio)'!r}")
    print(f"  caminho no grafo     : {' -> '.join(registro.caminho_no_grafo)}")
    print(f"  trechos recuperados  : {registro.trechos_recuperados}")
    print(f"  fontes do RAG        : {codigos_rag}")
    print(f"  códigos no texto     : {codigos_citados or '(nenhum)'}")
    print(f"  códigos inventados   : {inventados or '(nenhum)'}")
    print(f"  guardrail no modelo  : {m.tem_guardrail(bruto)}")
    print(f"  guardrail inserido   : {registro.guardrail_adicionado}")
    print(f"  erro                 : {registro.erro}")
    print(f"  duração              : {registro.duracao_s}s")

    return {
        "id": id_paciente,
        "desfecho": registro.desfecho,
        "desfecho_do_modelo": registro.desfecho_do_modelo,
        "codigos_rag": codigos_rag,
        "codigos_citados": codigos_citados,
        "inventados": inventados,
        "guardrail_no_modelo": m.tem_guardrail(bruto),
        "guardrail_inserido": registro.guardrail_adicionado,
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    casos = CASOS
    if argv:
        pedidos = {a.strip().upper() for a in argv}
        casos = [c for c in CASOS if c[0] in pedidos]
        if not casos:
            disponiveis = ", ".join(c[0] for c in CASOS)
            print(f"Nenhum caso corresponde a {sorted(pedidos)}. Casos: {disponiveis}")
            return 2

    print(SEPARADOR)
    print("SMOKE TEST — fluxo completo com modelo real")
    print(SEPARADOR)
    print(f"  modelo base   : {config.MODELO_BASE}")
    print(f"  adapter       : {config.ADAPTER_LORA}")
    print(f"  dtype_cpu     : {DTYPE_CPU}")
    print(f"  max_new_tokens: {MAX_NEW_TOKENS} (config: {config.MAX_NEW_TOKENS})")
    print("\nCarregando embeddings, índice e modelo (pode demorar)...", flush=True)

    assistente = criar_assistente(
        forcar_cpu=True,
        dtype_cpu=DTYPE_CPU,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(f"Pronto: {assistente.nome_modelo}", flush=True)

    resultados = [relatar(pid, pergunta, assistente) for pid, pergunta in casos]

    print(f"\n{SEPARADOR}\nRESUMO\n{SEPARADOR}")
    for r in resultados:
        print(
            f"  {r['id']}: {r['desfecho']:16} "
            f"rótulo_do_modelo={r['desfecho_do_modelo']!s:5} "
            f"guardrail_do_modelo={r['guardrail_no_modelo']!s:5} "
            f"códigos_inventados={len(r['inventados'])}"
        )

    print(f"\n  arquivo de log: {assistente.auditoria.arquivo}")
    print(f"  estatísticas  : {assistente.auditoria.estatisticas()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
