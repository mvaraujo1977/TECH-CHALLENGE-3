"""Registro estruturado das consultas, para rastreamento e auditoria.

Atende ao requisito 3 do desafio: "implementar logging detalhado para
rastreamento e auditoria" e "garantir explainability das respostas".

Cada consulta gera um registro em JSONL contendo a pergunta, o paciente
consultado, os trechos recuperados pelo RAG, o desfecho, a resposta e as
intervenções de segurança aplicadas. O formato JSONL permite auditar com
ferramentas de linha de comando e carregar em pandas sem parsing customizado.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src import config


@dataclass
class RegistroConsulta:
    """Uma consulta completa ao assistente, do input à resposta final."""

    # Identificação
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    momento: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    # Entrada
    pergunta: str = ""
    id_paciente: str | None = None

    # Recuperação (explainability)
    fontes: list[dict] = field(default_factory=list)
    trechos_recuperados: int = 0

    # Guardrail de entrada
    risco: str = ""
    regras_de_risco: list[str] = field(default_factory=list)
    versao_politica: str = ""

    # Decisão determinística (a que vale)
    desfecho: str = ""
    motivo_desfecho: str = ""
    sinais_gravidade: list[str] = field(default_factory=list)
    exames_pendentes: list[str] = field(default_factory=list)

    # Rótulo do LLM — registrado para medir concordância, não decide o fluxo
    desfecho_do_modelo: str | None = None
    concorda_com_modelo: bool | None = None

    # Saída
    resposta: str = ""
    guardrail_adicionado: bool = False

    # Execução
    caminho_no_grafo: list[str] = field(default_factory=list)
    duracao_s: float | None = None
    modelo: str = ""
    erro: str | None = None

    def para_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    def resumo(self) -> str:
        """Uma linha legível, para acompanhar a execução ao vivo."""
        partes = [f"[{self.id}]", self.desfecho]
        if self.risco and self.risco != "BLOQUEADO":
            partes.append(f"risco {self.risco}")
        if self.id_paciente:
            partes.append(self.id_paciente)
        if self.fontes:
            partes.append("fontes: " + ", ".join(f["codigo"] for f in self.fontes))
        if self.guardrail_adicionado:
            partes.append("guardrail+")
        if self.desfecho_do_modelo is None:
            partes.append("LLM sem rótulo")
        elif self.concorda_com_modelo is False:
            partes.append(f"LLM discordou ({self.desfecho_do_modelo})")
        if self.duracao_s is not None:
            partes.append(f"{self.duracao_s:.1f}s")
        return " | ".join(partes)


class Auditoria:
    """Escreve registros em JSONL, um por linha.

    Append-only por design: registros de auditoria não devem ser sobrescritos
    nem editados. Cada execução acrescenta ao arquivo do dia.
    """

    def __init__(self, diretorio: Path | None = None, nome: str | None = None):
        self.diretorio = Path(diretorio or config.DIR_LOGS)
        self.diretorio.mkdir(parents=True, exist_ok=True)

        if nome is None:
            hoje = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            nome = f"consultas-{hoje}.jsonl"

        self.arquivo = self.diretorio / nome

    def registrar(self, registro: RegistroConsulta) -> None:
        with self.arquivo.open("a", encoding="utf-8") as f:
            f.write(registro.para_json() + "\n")

    def ler(self, limite: int | None = None) -> list[dict]:
        """Lê os registros gravados, do mais recente para o mais antigo."""
        if not self.arquivo.exists():
            return []

        linhas = [
            json.loads(linha)
            for linha in self.arquivo.read_text(encoding="utf-8").splitlines()
            if linha.strip()
        ]
        linhas.reverse()
        return linhas[:limite] if limite else linhas

    def estatisticas(self) -> dict:
        """Métricas agregadas — material para o relatório e a demonstração."""
        registros = self.ler()
        if not registros:
            return {"total": 0}

        total = len(registros)
        por_desfecho: dict[str, int] = {}
        for r in registros:
            chave = r.get("desfecho", "?")
            por_desfecho[chave] = por_desfecho.get(chave, 0) + 1

        com_fonte = sum(1 for r in registros if r.get("fontes"))
        guardrail_add = sum(1 for r in registros if r.get("guardrail_adicionado"))
        erros = sum(1 for r in registros if r.get("erro"))

        # Concordância entre o rótulo do LLM e a decisão determinística.
        # `rotulo_valido` mede quantas vezes o modelo emitiu um rótulo
        # reconhecível; `concordancia` mede, entre esses, quantos coincidiram.
        rotulo_valido = sum(1 for r in registros if r.get("desfecho_do_modelo"))
        concordaram = sum(1 for r in registros if r.get("concorda_com_modelo") is True)

        # Distribuição do guardrail de entrada. `bloqueadas` conta as
        # solicitações recusadas antes de chegar ao modelo.
        por_risco: dict[str, int] = {}
        for r in registros:
            chave = r.get("risco") or "(sem classificação)"
            por_risco[chave] = por_risco.get(chave, 0) + 1
        bloqueadas = sum(1 for r in registros if r.get("risco") == "BLOQUEADO")

        duracoes = [r["duracao_s"] for r in registros if r.get("duracao_s")]

        return {
            "total": total,
            "por_desfecho": por_desfecho,
            "por_risco": por_risco,
            "bloqueadas_na_entrada": f"{bloqueadas}/{total}",
            "com_citacao_de_fonte": f"{com_fonte}/{total}",
            "guardrail_adicionado_por_codigo": f"{guardrail_add}/{total}",
            "llm_emitiu_rotulo_valido": f"{rotulo_valido}/{total}",
            "llm_concordou_com_a_regra": (
                f"{concordaram}/{rotulo_valido}" if rotulo_valido else "n/a"
            ),
            "erros": erros,
            "duracao_media_s": round(sum(duracoes) / len(duracoes), 2) if duracoes else None,
        }
