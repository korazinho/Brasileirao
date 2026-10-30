"""REST API do motor econométrico original para a interface Lovable.

Inicie localmente com: python -m uvicorn main:app --reload --port 8000
O app publica SOMENTE endpoints de leitura e cenários hipotéticos.
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import json
import math
import time
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from motor import LABELS, download_public, prepare, simulate, synthetic_demo

TZ = ZoneInfo("America/Sao_Paulo")
PUBLIC_TTL_SECONDS = 300
YEAR = 2026

app = FastAPI(
    title="Brasileirão Analytics — API Econométrica",
    version="2.1.0",
    description=("Regressões de Poisson e Monte Carlo para Séries A e B. "
                 "Dados da fonte pública dependem da publicação de resultados finais. "
                 "Modo demonstração é inteiramente sintético."),
)
# A API é pública e de leitura. Não usar este CORS aberto caso se adicionem
# endpoints privados ou autenticação no futuro.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

MODEL_OPTIONS = ("Automático", *LABELS.keys())


def now_brasilia() -> datetime:
    return datetime.now(TZ)


def _date_cut(asof: str | None) -> pd.Timestamp:
    now = now_brasilia().replace(tzinfo=None)
    # Arredondar o corte atual a intervalos de 5 minutos permite cache do ajuste.
    # Isto NÃO torna os placares ao vivo nem antecipa publicação de resultados.
    if asof is None:
        return pd.Timestamp(now).floor("5min")
    try:
        date = datetime.strptime(asof, "%Y-%m-%d").date()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="corte deve ter formato YYYY-MM-DD") from exc
    if date > now.date():
        raise HTTPException(status_code=422, detail="Não é permitido corte no futuro")
    if date == now.date():
        return pd.Timestamp(now).floor("5min")
    return pd.Timestamp(date) + pd.Timedelta(hours=23, minutes=59, seconds=59)


def _validate_params(serie: str, modelo: str, sims: int):
    if serie not in ('A', 'B'):
        raise HTTPException(status_code=422, detail="serie deve ser A ou B")
    if modelo not in MODEL_OPTIONS:
        raise HTTPException(status_code=422, detail=f"modelo deve ser um de {MODEL_OPTIONS}")
    if sims < 500 or sims > 20000:
        raise HTTPException(status_code=422, detail="simulacoes deve ficar entre 500 e 20000")


@lru_cache(maxsize=12)
def _source(serie: str, demo: bool, refresh_bucket: int):
    if demo:
        return synthetic_demo()
    return download_public(serie)


@lru_cache(maxsize=32)
def _bundle(serie: str, demo: bool, modelo: str, corte: str, refresh_bucket: int):
    h, f = _source(serie, demo, refresh_bucket)
    return prepare(h, f, pd.Timestamp(corte), modelo, serie=serie)


def _build(serie: str, demo: bool, modelo: str, simulacoes: int, semente: int, corte: str | None):
    _validate_params(serie, modelo, simulacoes)
    cut = _date_cut(corte)
    bucket = 0 if demo else int(time.time() // PUBLIC_TTL_SECONDS)
    try:
        bundle = _bundle(serie, demo, modelo, cut.isoformat(), bucket)
    except HTTPException:
        raise
    except Exception as exc:
        # Não retornar traceback ou caminhos internos no serviço público.
        raise HTTPException(status_code=503, detail={
            "mensagem": "Não foi possível carregar/estimar os dados desta série.",
            "orientacao": "Confira a fonte pública e o formato dos resultados. Tente novamente mais tarde.",
            "erro": str(exc)[:240],
        }) from exc
    try:
        projected = simulate(bundle, n=simulacoes, seed=semente)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha na simulação: {str(exc)[:200]}") from exc
    return bundle, projected


def _records(df: pd.DataFrame) -> list[dict]:
    """Converte NaN/NaT e types do NumPy em JSON válido sem Infinity."""
    return json.loads(df.to_json(orient="records", date_format="iso", date_unit="s"))


def _payload(bundle: dict, projected: pd.DataFrame, *, demo: bool,
             sims: int, serie: str, seed: int) -> dict:
    current = bundle["standings"].copy()
    current["posicao_atual"] = range(1, len(current) + 1)
    standings = projected.merge(current[["time", "posicao_atual"]], on="time", validate="one_to_one")
    # Posição projetada = ordem dos pontos finais médios, com média de posição
    # como desempate. Não equivale a chance modal de posição final.
    forecasts = bundle["forecasts"].copy()
    forecasts.insert(0, "jogo_indice", range(len(forecasts)))
    return {
        "metadata": {
            "temporada": bundle["year"], "serie": serie,
            "fonte": "Demonstração fictícia" if demo else "Fontes públicas",
            "ficticio": bool(demo),
            "modelo_selecionado": bundle["selected"],
            "corte": bundle["cut"].isoformat(),
            "ultima_consulta_api": now_brasilia().isoformat(),
            "atualizacao_aproximada_segundos": PUBLIC_TTL_SECONDS,
            "ao_vivo": False,
            "jogos_concluidos": int(bundle["n_played"]),
            "jogos_restantes": int(len(forecasts)),
            "simulacoes": sims,
            "semente": seed,
            "aviso": "Não é placar ao vivo: depende dos resultados finais divulgados pelas fontes.",
            "limitacao": ("Série B pública: modelo exploratório sem validação histórica externa."
                          if serie == 'B' and not demo else None),
        },
        "classificacao_atual": _records(current),
        "classificacao_projetada": _records(standings),
        "partidas_restantes": _records(forecasts),
        "primeiro_turno": _records(bundle["first"]),
        "coeficientes": _records(bundle["coefs"]),
        "correlacoes_exploratorias": _records(bundle["correlations"]),
        "validacao": _records(bundle["score"]),
    }


@app.get("/")
def root():
    return {"projeto": "Brasileirão Analytics", "status": "online", "documentacao": "/docs",
            "health": "/health", "projecoes": "/api/v1/projecoes"}


@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "ok", "motor": "Python / Poisson / Monte Carlo", "servico": "brasileirao-analytics-api",
            "dados_carregados": False, "observacao": "Health não verifica a disponibilidade das fontes públicas."}


@app.get("/api/v1/series")
def series():
    return {"temporada": YEAR, "series": [{"id": "A", "nome": "Série A"},
                                            {"id": "B", "nome": "Série B"}],
            "modelos": [{"id": option, "nome": LABELS.get(option, "Automático · menor Brier histórico")}
                        for option in MODEL_OPTIONS]}


@app.get("/api/v1/projecoes")
def projections(
    serie: Literal['A', 'B'] = 'A',
    simulacoes: int = Query(2000, ge=500, le=20000),
    semente: int = Query(2026, ge=0, le=999999),
    modelo: str = 'Automático',
    corte: str | None = None,
    demonstracao: bool = False,
):
    bundle, predicted = _build(serie, demonstracao, modelo, simulacoes, semente, corte)
    return _payload(bundle, predicted, demo=demonstracao, sims=simulacoes,
                    serie=serie, seed=semente)


class ScenarioRequest(BaseModel):
    serie: Literal['A', 'B'] = 'A'
    modelo: str = 'Automático'
    corte: str | None = None
    demonstracao: bool = False
    simulacoes: int = Field(default=2000, ge=500, le=20000)
    semente: int = Field(default=2026, ge=0, le=999999)
    jogo_indice: int = Field(ge=0)
    resultado: Literal['Mandante', 'Empate', 'Visitante']


@app.post("/api/v1/cenarios")
def scenarios(req: ScenarioRequest):
    bundle, baseline = _build(req.serie, req.demonstracao, req.modelo,
                              req.simulacoes, req.semente, req.corte)
    forecasts = bundle["forecasts"]
    if req.jogo_indice >= len(forecasts):
        raise HTTPException(status_code=422, detail="jogo_indice não corresponde a jogo restante")
    scenario = simulate(bundle, n=req.simulacoes, seed=req.semente,
                        override=req.jogo_indice, outcome=req.resultado)
    ix = req.jogo_indice
    match = forecasts.iloc[ix]
    b = baseline.set_index("time")
    s = scenario.set_index("time")
    cols = ['prob_titulo', 'prob_z4', 'pontos_previstos']
    if req.serie == 'B':
        cols += ['prob_acesso', 'prob_playoffs', 'prob_acesso_direto']
    deltas = scenario[['time']].copy()
    for col in cols:
        deltas[f'delta_{col}'] = deltas['time'].map((s[col]-b[col]).round(2).to_dict())
    return {
        "metadata": {"serie": req.serie, "ficticio": req.demonstracao,
                     "modelo_selecionado": bundle['selected'], "corte": bundle['cut'].isoformat(),
                     "simulacoes": req.simulacoes, "jogo_indice": ix, "resultado_forcado": req.resultado,
                     "partida": {"mandante": match.mandante, "visitante": match.visitante,
                                 "rodada": int(match.rodada) if pd.notna(match.rodada) else None}},
        "classificacao_cenario": _records(scenario),
        "deltas": _records(deltas),
    }
