"""Avaliação da política de risco.

Pergunta que este módulo responde: *o classificador roteia corretamente as
solicitações entre INFORMATIVO, DADOS_PACIENTE, CONDUTA_CLINICA e BLOQUEADO?*

Acurácia global não basta. O que importa clinicamente é a **assimetria dos
erros**: tratar um pedido de prescrição como consulta informativa é muito mais
grave que o inverso. O relatório separa as duas direções:

``subestimacao``   previu risco **menor** que o esperado. Grave: a solicitação
                   segue por um caminho com menos salvaguardas.
``superestimacao`` previu risco **maior**. Custoso em usabilidade — uma
                   consulta legítima recebe tratamento restritivo — mas seguro.

O conjunto de holdout existe para medir generalização. Suas formulações são
deliberadamente diferentes das do benchmark principal, então acurácia alta nele
indica que as regras capturam a intenção, e não apenas as palavras que foram
usadas para escrevê-las.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src import config
from src.seguranca.politica import VERSAO_POLITICA, Risco, classificar

ORDEM = [r.value for r in Risco]

SEVERIDADE = {nome: indice for indice, nome in enumerate(ORDEM)}

BENCHMARK_PADRAO = config.DIR_DADOS / "benchmarks" / "seguranca.jsonl"
HOLDOUT_PADRAO = config.DIR_DADOS / "benchmarks" / "seguranca_holdout.jsonl"


@dataclass(frozen=True)
class Item:
    id: str
    pergunta: str
    esperado: str


@dataclass
class Relatorio:
    """Resultado agregado da avaliação."""

    conjunto: str
    n: int
    acuracia: float
    por_categoria: dict[str, dict[str, Any]]
    confusao: dict[str, dict[str, int]]
    subestimacao: int
    superestimacao: int
    erros: list[dict[str, str]] = field(default_factory=list)
    versao_politica: str = VERSAO_POLITICA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def matriz(self) -> str:
        """Matriz de confusão em texto. Linhas = esperado, colunas = previsto."""
        largura = 17
        cabecalho = "esperado \\ previsto".ljust(22)
        cabecalho += "".join(nome[:15].rjust(largura) for nome in ORDEM)
        linhas = [cabecalho, "-" * len(cabecalho)]

        for esperado in ORDEM:
            linha = esperado.ljust(22)
            for previsto in ORDEM:
                valor = self.confusao[esperado][previsto]
                marca = str(valor) if valor else "·"
                linha += marca.rjust(largura)
            linhas.append(linha)

        return "\n".join(linhas)

    def resumo(self) -> str:
        partes = [
            f"{self.conjunto}: {self.acuracia:.1%} ({self.n} prompts)",
            f"subestimação: {self.subestimacao}",
            f"superestimação: {self.superestimacao}",
        ]
        return " | ".join(partes)


def carregar(caminho: Path) -> list[Item]:
    if not caminho.exists():
        raise FileNotFoundError(f"Benchmark não encontrado: {caminho}")

    itens = []
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        dados = json.loads(linha)
        itens.append(Item(dados["id"], dados["pergunta"], dados["esperado"]))
    return itens


def avaliar(itens: list[Item], conjunto: str = "benchmark") -> Relatorio:
    """Classifica cada item e agrega os resultados."""
    confusao = {e: {p: 0 for p in ORDEM} for e in ORDEM}
    por_categoria = {c: {"total": 0, "corretos": 0} for c in ORDEM}
    erros: list[dict[str, str]] = []
    subestimacao = superestimacao = 0

    for item in itens:
        previsto = classificar(item.pergunta).categoria.value

        confusao[item.esperado][previsto] += 1
        por_categoria[item.esperado]["total"] += 1

        if previsto == item.esperado:
            por_categoria[item.esperado]["corretos"] += 1
            continue

        direcao = SEVERIDADE[previsto] - SEVERIDADE[item.esperado]
        if direcao < 0:
            subestimacao += 1
        else:
            superestimacao += 1

        erros.append({
            "id": item.id,
            "pergunta": item.pergunta,
            "esperado": item.esperado,
            "previsto": previsto,
            "direcao": "subestimação" if direcao < 0 else "superestimação",
        })

    for categoria, dados in por_categoria.items():
        total = dados["total"]
        dados["acuracia"] = round(dados["corretos"] / total, 4) if total else None

    acertos = sum(d["corretos"] for d in por_categoria.values())

    return Relatorio(
        conjunto=conjunto,
        n=len(itens),
        acuracia=round(acertos / len(itens), 4) if itens else 0.0,
        por_categoria=por_categoria,
        confusao=confusao,
        subestimacao=subestimacao,
        superestimacao=superestimacao,
        erros=erros,
    )


def avaliar_tudo(
    benchmark: Path | None = None,
    holdout: Path | None = None,
) -> dict[str, Relatorio]:
    """Avalia benchmark e holdout, devolvendo os dois relatórios."""
    resultados = {}

    caminho_bm = benchmark or BENCHMARK_PADRAO
    if caminho_bm.exists():
        resultados["benchmark"] = avaliar(carregar(caminho_bm), "benchmark")

    caminho_hd = holdout or HOLDOUT_PADRAO
    if caminho_hd.exists():
        resultados["holdout"] = avaliar(carregar(caminho_hd), "holdout")

    return resultados


def salvar(relatorios: dict[str, Relatorio], diretorio: Path | None = None) -> list[Path]:
    """Grava os relatórios em JSON e a matriz de confusão em texto."""
    destino = Path(diretorio or (config.RAIZ / "docs" / "resultados"))
    destino.mkdir(parents=True, exist_ok=True)

    gravados = []
    for nome, relatorio in relatorios.items():
        arquivo = destino / f"seguranca_{nome}.json"
        arquivo.write_text(
            json.dumps(relatorio.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        gravados.append(arquivo)

        matriz = destino / f"seguranca_{nome}_confusao.txt"
        matriz.write_text(
            f"{relatorio.resumo()}\npolítica {relatorio.versao_politica}\n\n"
            f"{relatorio.matriz()}\n",
            encoding="utf-8",
        )
        gravados.append(matriz)

    return gravados


def main() -> int:
    """Executa a avaliação e imprime o resultado. Sem GPU, sem modelo."""
    relatorios = avaliar_tudo()

    if not relatorios:
        print("Nenhum benchmark encontrado.")
        return 1

    for nome, relatorio in relatorios.items():
        print("=" * 72)
        print(f"POLÍTICA DE RISCO — {nome.upper()}  (versão {relatorio.versao_politica})")
        print("=" * 72)
        print(f"\nAcurácia: {relatorio.acuracia:.1%} em {relatorio.n} prompts\n")

        for categoria, dados in relatorio.por_categoria.items():
            if not dados["total"]:
                continue
            print(f"  {categoria:16s} {dados['corretos']:2d}/{dados['total']:2d}"
                  f"  ({dados['acuracia']:.0%})")

        print(f"\n  subestimação  {relatorio.subestimacao:2d}   (previu risco MENOR — grave)")
        print(f"  superestimação {relatorio.superestimacao:2d}   (previu risco MAIOR — seguro)")

        print("\n" + relatorio.matriz())

        if relatorio.erros:
            print("\nErros:")
            for erro in relatorio.erros:
                print(f"  [{erro['id']}] {erro['direcao']}: "
                      f"{erro['esperado']} → {erro['previsto']}")
                print(f"      {erro['pergunta']}")
        print()

    gravados = salvar(relatorios)
    print(f"Relatórios gravados: {len(gravados)} arquivo(s) em docs/resultados/")

    # Subestimação é o erro que importa: retorna falha se houver alguma.
    total_subestimacao = sum(r.subestimacao for r in relatorios.values())
    if total_subestimacao:
        print(f"\n⚠️  {total_subestimacao} subestimação(ões) de risco.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
