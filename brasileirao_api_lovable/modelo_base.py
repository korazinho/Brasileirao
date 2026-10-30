#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regressao de gols (Poisson) + simulacao do Brasileirao Serie A 2026.

Fontes:
* football-data.co.uk/new/BRA.csv (partidas 2012-2026), direitos da fonte;
* brasileiraoemrodadas.com.br/docs/serie_a_2026/ (calendario/placares).

O script baixa fontes publicas quando executado COM acesso a Internet. Nao usa
probabilidades de terceiros como alvo. Nao incorpora resultados futuros ao corte.
Saidas: probabilidades_2026.csv, partidas_projetadas_2026.csv,
        coeficientes_2026.csv, primeiro_turno_2026.csv,
        diagnostico_validacao.csv, grafico_probabilidades_2026.png.

pip install pandas numpy scipy scikit-learn requests lxml matplotlib
python brasileirao_econometrico_2026.py
python brasileirao_econometrico_2026.py --corte '2026-10-30 10:13' --simulacoes 20000
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from collections import defaultdict
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr, skellam
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

URL_HIST = 'https://www.football-data.co.uk/new/BRA.csv'
URL_2026 = 'https://brasileiraoemrodadas.com.br/docs/serie_a_2026/'
URL_SERIE_B_2026 = 'https://brasileiraoemrodadas.com.br/docs/serie_b_2026/'
COLUMNS = ['temporada','data','rodada','mandante','visitante','gols_m','gols_v']


def norm(v: object) -> str:
    s = str(v).lower().strip()
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    s = re.sub(r'[^a-z0-9]+', ' ', s).strip()
    aliases = {
       'athletico paranaense':'athletico pr', 'atletico paranaense':'athletico pr',
       'atletico pr':'athletico pr','athletico':'athletico pr',
       'atletico mineiro':'atletico mg', 'atletico mg':'atletico mg',
       'bragantino':'rb bragantino', 'red bull bragantino':'rb bragantino',
       'vasco da gama':'vasco','botafogo rj':'botafogo',
       'flamengo rj':'flamengo','chapecoense sc':'chapecoense',
       'america mineiro':'america mg','america mg':'america mg',
       'atletico goianiense':'atletico go','atletico go':'atletico go',
       'avai sc':'avai', 'coritiba saf':'coritiba',
       'gremio fbpa':'gremio', 'sao paulo fc':'sao paulo',
    }
    return aliases.get(s, s)


def download_hist(session: requests.Session, cache: Path) -> pd.DataFrame:
    if not cache.exists():
        r = session.get(URL_HIST, timeout=45)
        r.raise_for_status()
        cache.write_bytes(r.content)
    data = pd.read_csv(cache, encoding='cp1252', low_memory=False)
    data.columns = data.columns.str.strip()
    names = {s.lower(): s for s in data.columns}
    expected = ['Season','Date','Home','Away','HG','AG']
    if not all(x.lower() in names for x in expected):
        raise ValueError('Formato inesperado no BRA.csv: '+', '.join(data.columns[:20]))
    sel = pd.DataFrame({
        'temporada': pd.to_numeric(data[names['season']], errors='coerce'),
        'data': pd.to_datetime(data[names['date']], dayfirst=True, errors='coerce'),
        'mandante': data[names['home']].map(norm),
        'visitante': data[names['away']].map(norm),
        'gols_m': pd.to_numeric(data[names['hg']], errors='coerce'),
        'gols_v': pd.to_numeric(data[names['ag']], errors='coerce'),
    })
    if 'league' in names:
        league = data[names['league']].astype(str).str.lower().str.strip()
        serie_a = league.str.contains(r'serie\s*a|brazil\s*1|brasileir[aã]o\s*(?:serie\s*)?a', regex=True)
        if not serie_a.any():
            raise ValueError('Nome da primeira divisao nao reconhecido. Leagues: '+str(league.unique()[:20]))
        sel = sel.loc[serie_a].copy()
    sel = sel[sel.temporada.between(2017,2026)].copy()
    sel['rodada'] = np.nan  # historico pode nao informar rodada; usamos jogos previos.
    sel = sel.dropna(subset=['data','mandante','visitante'])
    sel = sel.drop_duplicates(['temporada','mandante','visitante'],keep='last')
    return sel[COLUMNS].sort_values(['temporada','data','mandante']).reset_index(drop=True)


def download_fixture(session: requests.Session, cache: Path, serie: str = "A") -> pd.DataFrame:
    if not cache.exists():
        r=session.get(URL_2026 if serie == "A" else URL_SERIE_B_2026, timeout=45)
        r.raise_for_status()
        cache.write_text(r.text,encoding='utf-8')
    raw_tables=pd.read_html(StringIO(cache.read_text(encoding='utf-8')))
    options=[t for t in raw_tables if {'Mandante','Visitante','Placar','Data'}.issubset(set(map(str,t.columns)))]
    if not options:
        raise ValueError('Nao encontrei tabela de jogos no HTML de 2026. Confira o site ou passe cache atualizado.')
    raw=max(options,key=len).copy()
    # A fonte insere um cabeçalho de seção para cada uma das 38 rodadas.
    # pd.read_html() expande <td colspan="7">1ª rodada</td> em sete
    # colunas, contando esse cabeçalho como uma linha adicional (418 = 380+38).
    # Retire SOMENTE os cabeçalhos reconhecidos, sem apagar partidas inválidas.
    labels = raw['Rodada'].astype(str).str.strip()
    section_headers = labels.str.fullmatch(r'\d{1,2}[ªº]?\s+rodada', case=False, na=False)
    raw = raw.loc[~section_headers].copy()
    # Jogos adiados podem estar com data/hora em branco na fonte (p. ex.
    # Chapecoense x Vasco, 21ª rodada). Conservar a partida com data NaT,
    # desde que não exista placar, em vez de inventar uma data ou eliminá-la.
    raw_date = raw['Data'].astype('string').str.strip()
    blank_dates = raw_date.isna() | raw_date.fillna('').eq('')
    valid_dates = raw_date.str.fullmatch(r'\d{1,2}/\d{1,2}/\d{4}').fillna(False)
    score_text = raw['Placar'].astype(str)
    has_score = score_text.str.contains(r'\d+\s*[xX×–—-]\s*\d+', regex=True, na=False)
    invalid_dates = ~(valid_dates | (blank_dates & ~has_score))
    if invalid_dates.any():
        examples = raw.loc[invalid_dates, ['Rodada', 'Data', 'Mandante', 'Visitante']].head(3)
        raise ValueError('Linhas com data inválida no calendário: '+examples.to_string(index=False))
    fixtures=[]
    for _,r in raw.iterrows():
        placar=str(r['Placar'])
        m=re.search(r'(\d+)\s*[xX×–—-]\s*(\d+)',placar)
        rodada_match=re.search(r'(\d+)',str(r.get('Rodada','')))
        hora = str(r.get('Hora', '23:59')).strip()
        if not re.match(r'^\d{1,2}:\d{2}$', hora):
            hora = '23:59'
        # Data não anunciada é desconhecida, não uma data fictícia.
        dt = (pd.NaT if bool(blank_dates.loc[_]) else
              pd.to_datetime(f"{r['Data']} {hora}", format='%d/%m/%Y %H:%M', errors='coerce'))
        if pd.isna(dt) and not bool(blank_dates.loc[_]):
            raise ValueError(f'Data inválida para {r["Mandante"]} x {r["Visitante"]}: {r["Data"]}')
        fixtures.append({
            'temporada':2026,
            'data':dt,
            'rodada':int(rodada_match.group(1)) if rodada_match else np.nan,
            'mandante':norm(r['Mandante']), 'visitante':norm(r['Visitante']),
            'gols_m':float(m.group(1)) if m else np.nan,
            'gols_v':float(m.group(2)) if m else np.nan,
        })
    fx=pd.DataFrame(fixtures)[COLUMNS]
    if len(fx)!=380 or fx[['mandante','visitante']].duplicated().any():
        raise ValueError(f'Calendario inconsistente: {len(fx)} partidas; esperado 380 sem duplas repetidas.')
    return fx


def compose_2026(fx: pd.DataFrame, bra: pd.DataFrame, corte: pd.Timestamp) -> pd.DataFrame:
    current=bra[(bra.temporada==2026)&bra.gols_m.notna()&bra.gols_v.notna()]
    # BRA pode ter placares mais recentes que a pagina com calendario.
    merged=fx.merge(current[['mandante','visitante','data','gols_m','gols_v']].rename(
        columns={'data':'bra_data','gols_m':'bra_m','gols_v':'bra_v'}),
        on=['mandante','visitante'],how='left',validate='one_to_one')
    conflict=merged[(merged.gols_m.notna()) & (merged.bra_m.notna()) &
                    ((merged.gols_m!=merged.bra_m)|(merged.gols_v!=merged.bra_v))]
    if len(conflict):
        raise ValueError('Placar divergente entre fontes; revise: '+str(conflict[['mandante','visitante']].head().values.tolist()))
    # A fonte de resultados so vale se a data dela nao exceder o corte.
    merged.loc[merged.bra_data>corte,['bra_m','bra_v']]=np.nan
    merged['gols_m']=merged['gols_m'].fillna(merged['bra_m'])
    merged['gols_v']=merged['gols_v'].fillna(merged['bra_v'])
    merged=merged.drop(columns=['bra_m','bra_v','bra_data'])
    # data limite: jogos de HOJE so podem ser conhecidos depois do horario de inicio.
    # Como o calendario pode ter remarcacoes, nao presumir jogo finalizado se data > corte.
    merged.loc[merged.data>corte,['gols_m','gols_v']]=np.nan
    return merged[COLUMNS].sort_values(['data','mandante']).reset_index(drop=True)


def ppg(games: list, kind: str = 'all', last: int|None = None) -> float:
    base = [g for g in games if kind=='all' or g['venue']==kind]
    if last is not None:
        base=base[-last:]
    # suavizacao: 3 partidas de informacao inicial com 1,33 pontos/jogo
    return (sum(g['points'] for g in base)+1.33*3)/(len(base)+3)


def rate(games: list, which: str, last: int|None = None) -> float:
    subset=games[-last:] if last is not None else games
    return (sum(g[which] for g in subset)+1.33*3)/(len(subset)+3)


FEATURES = [
    'mando', 'forca_turno_1', 'forca_turno_adv',
    'ppg_temporada', 'ppg_oponente','forma_5','forma_5_adv',
    'ppg_no_mando','ppg_adv_no_mando','gols_recentes','gols_sofridos_adv',
    'forca_calendario','forca_calendario_adv',
    'adversario_z4','adversario_titulo','proprio_z4','reta_final',
]


def features_for(own: str, opp: str, is_home: bool, hist: dict, clubs: list[str]) -> dict:
    a,b=hist[own],hist[opp]
    n=max(1, len(a)); nb=max(1,len(b))
    # classificacao medida somente ANTES da partida, sem vazamento do resultado.
    rankings=sorted(clubs,key=lambda k:(-ppg(hist[k]),-len(hist[k]),k))
    ra=rankings.index(own)+1; rb=rankings.index(opp)+1
    late=1 if min(n,nb)>=24 else 0
    # Forca da tabela enfrentada ate HOJE, pela campanha previa dos oponentes.
    # Sem usar placares posteriores ao confronto que estamos prevendo.
    def dificuldade(partidas):
        opps=[g.get('opponent') for g in partidas if g.get('opponent') in hist]
        return float(np.mean([ppg(hist[t]) for t in opps])) if opps else 1.33
    # Pontos do primeiro turno sao congelados na 19a partida de cada equipe.
    own_first=ppg(a[:19]); opp_first=ppg(b[:19])
    venue='home' if is_home else 'away'
    awayvenue='away' if is_home else 'home'
    values={
        'mando':int(is_home),
        'forca_turno_1':own_first, 'forca_turno_adv':opp_first,
        'ppg_temporada':ppg(a), 'ppg_oponente':ppg(b),
        'forma_5':ppg(a,last=5), 'forma_5_adv':ppg(b,last=5),
        'ppg_no_mando':ppg(a,venue),
        'ppg_adv_no_mando':ppg(b,awayvenue),
        'gols_recentes':rate(a,'gf',last=5),
        'gols_sofridos_adv':rate(b,'ga',last=5),
        'forca_calendario':dificuldade(a),
        'forca_calendario_adv':dificuldade(b),
        'adversario_z4':int(rb>=17 and late),
        'adversario_titulo':int(rb<=3 and late),
        'proprio_z4':int(ra>=17 and late),
        'reta_final':late,
    }
    return values


def initial_hist(clubs: list[str]):
    return {t:[] for t in clubs}


def update_hist(hist, match):
    home,away=match['mandante'],match['visitante']
    gh,ga=int(match['gols_m']),int(match['gols_v'])
    hist[home].append({'venue':'home','opponent':away,'points':3 if gh>ga else 1 if gh==ga else 0,'gf':gh,'ga':ga})
    hist[away].append({'venue':'away','opponent':home,'points':3 if ga>gh else 1 if gh==ga else 0,'gf':ga,'ga':gh})


def season_records(df: pd.DataFrame):
    clubs=sorted(set(df.mandante)|set(df.visitante))
    hist=initial_hist(clubs)
    observations=[]
    observed=df[df.gols_m.notna() & df.gols_v.notna()].sort_values(['data','mandante'])
    # Cada data e um bloco: partidas do mesmo dia nao enxergam resultados umas das outras.
    for _, same_day in observed.groupby(observed.data.dt.normalize(),sort=True):
        for _,r in same_day.iterrows():
            hm,am=r.mandante,r.visitante
            row_base={'temporada':int(r.temporada),'data':r.data}
            observations.append({**row_base,'alvo_gols':int(r.gols_m),**features_for(hm,am,True,hist,clubs)})
            observations.append({**row_base,'alvo_gols':int(r.gols_v),**features_for(am,hm,False,hist,clubs)})
        for _,r in same_day.iterrows():
            update_hist(hist,r)
    return pd.DataFrame(observations),hist,clubs


def table_from_played(df:pd.DataFrame,clubs):
    standing={t:{'time':t,'pontos':0,'vitorias':0,'gols_pro':0,'gols_contra':0,'jogos':0} for t in clubs}
    for _,r in df[df.gols_m.notna() & df.gols_v.notna()].iterrows():
        h,a=r.mandante,r.visitante
        gh,ga=int(r.gols_m),int(r.gols_v)
        standing[h]['jogos']+=1; standing[a]['jogos']+=1
        standing[h]['gols_pro']+=gh; standing[h]['gols_contra']+=ga
        standing[a]['gols_pro']+=ga; standing[a]['gols_contra']+=gh
        if gh>ga:
            standing[h]['pontos']+=3; standing[h]['vitorias']+=1
        elif gh<ga:
            standing[a]['pontos']+=3; standing[a]['vitorias']+=1
        else:
            standing[h]['pontos']+=1; standing[a]['pontos']+=1
    tab=pd.DataFrame(standing.values())
    tab['saldo']=tab.gols_pro-tab.gols_contra
    return tab.sort_values(['pontos','vitorias','saldo','gols_pro'],ascending=False).reset_index(drop=True)


MODELS = {
    'A_forca': ['mando','ppg_temporada','ppg_oponente','gols_recentes','gols_sofridos_adv'],
    'B_forma_casa_fora': ['mando','ppg_temporada','ppg_oponente','gols_recentes',
                          'gols_sofridos_adv','forca_turno_1','forca_turno_adv',
                          'forma_5','forma_5_adv','ppg_no_mando','ppg_adv_no_mando',
                          'forca_calendario','forca_calendario_adv'],
    'C_contexto': FEATURES,
}


def fit_model(train, feats):
    model=make_pipeline(StandardScaler(),PoissonRegressor(alpha=0.25,max_iter=2500))
    model.fit(train[feats],train.alvo_gols)
    return model


def evaluate_score(pred: np.ndarray, observed: np.ndarray):
    """Deviance dos gols + Brier multiclasses e log-loss 1x2."""
    gh,ga=observed[::2],observed[1::2]
    mh,ma=pred[::2],pred[1::2]
    prob_h=skellam.sf(0,mh,ma)
    prob_d=skellam.pmf(0,mh,ma)
    prob_a=skellam.cdf(-1,mh,ma)
    probs=np.clip(np.column_stack([prob_h,prob_d,prob_a]),1e-8,1)
    probs=probs/probs.sum(axis=1,keepdims=True)
    outcomes=np.where(gh>ga,0,np.where(gh==ga,1,2))
    truth=np.eye(3)[outcomes]
    return {'mae_gols':mean_absolute_error(observed,pred),
            'poisson_deviance':mean_poisson_deviance(observed,pred),
            'brier_1x2':float(np.mean(np.sum((probs-truth)**2,axis=1))),
            'logloss_1x2':float(-np.mean(np.log(probs[np.arange(len(probs)),outcomes])))}


def validate_longitudinal(train):
    # Toda uma temporada de teste fica fora do treinamento, sem embaralhamento.
    results=[]
    for year in [2024,2025]:
        tr=train[train.temporada<year]
        va=train[train.temporada==year]
        if len(tr)<1000 or len(va)<400:
            continue
        observed=va.alvo_gols.to_numpy()
        baseline=tr.groupby('mando').alvo_gols.mean()
        ref=np.clip(va['mando'].map(baseline).to_numpy(),.05,8)
        scores=evaluate_score(ref,observed)
        results.append({'temporada_teste':year,'modelo':'Referencia_mando','n_jogos':len(va)//2,**scores})
        for name,feats in MODELS.items():
            m=fit_model(tr,feats)
            pred=np.clip(m.predict(va[feats]),0.05,8)
            scores=evaluate_score(pred,observed)
            results.append({'temporada_teste':year,'modelo':name,'n_jogos':len(va)//2,**scores})
    return pd.DataFrame(results)


def run(args):
    out=Path(args.saida);out.mkdir(parents=True,exist_ok=True)
    corte=pd.Timestamp(args.corte) if args.corte else pd.Timestamp(datetime.now(ZoneInfo('America/Sao_Paulo')).replace(tzinfo=None))
    session=requests.Session()
    session.headers.update({'User-Agent':'Mozilla/5.0 (research; brasileirao-econometrico)'})
    if args.atualizar:
        for path in [out/'fonte_BRA_cache.csv',out/'fonte_tabela_2026_cache.html']:
            path.unlink(missing_ok=True)
    bra=download_hist(session,out/'fonte_BRA_cache.csv')
    fx=download_fixture(session,out/'fonte_tabela_2026_cache.html')
    season26=compose_2026(fx,bra,corte)
    played=season26[season26.gols_m.notna()]
    remaining=season26[season26.gols_m.isna()].copy()
    if len(played)<190 or len(played)+len(remaining)!=380:
        raise ValueError('Quantidade inconsistente de jogos. Nao simular sem reconciliar dados.')
    teams=sorted(set(season26.mandante)|set(season26.visitante))
    if len(teams)!=20:
        raise ValueError(f'Esperados 20 clubes, encontrados {len(teams)}: {teams}')
    print(f'Corte: {corte} | disputados: {len(played)} | restantes: {len(remaining)}')

    # O arquivo BRA e usado como base historica; 2026 e substituido pelo calendario
    # conferido partida a partida, para nao faltar um confronto em aberto.
    sample=bra[(bra.temporada>=2018)&(bra.temporada<=2025)&bra.gols_m.notna()&bra.gols_v.notna()]
    records=[]
    for year,d in sample.groupby('temporada'):
        rec,_,_=season_records(d)
        if len(rec): records.append(rec)
    cur_obs,hist,clubs=season_records(season26)
    records.append(cur_obs)
    train=pd.concat(records,ignore_index=True)
    train=train.dropna(subset=FEATURES+['alvo_gols'])
    checks=validate_longitudinal(train[train.temporada<2026])
    checks.to_csv(out/'diagnostico_validacao.csv',index=False)
    # Seleciona a especificacao pelo menor Brier 1x2 FORA DA AMOSTRA.
    if checks.empty:
        selected='B_forma_casa_fora'
    else:
        mean_brier=checks.groupby('modelo').brier_1x2.mean()
        selected=min(MODELS,key=lambda k: mean_brier.get(k,float('inf')))
    feats=MODELS[selected]
    print(f'Modelo econometrico selecionado na validacao: {selected}')
    model=fit_model(train,feats)
    betas=model.named_steps['poissonregressor'].coef_
    assoc=[]
    for i,k in enumerate(feats):
        rho,p=spearmanr(train[k],train.alvo_gols)
        assoc.append({'variavel':k,'coef_regressao_por_desvio_padrao':float(betas[i]),
                      'spearman_exploratorio':float(rho) if np.isfinite(rho) else np.nan,
                      'p_exploratorio':float(p) if np.isfinite(p) else np.nan})
    pd.DataFrame(assoc).to_csv(out/'coeficientes_2026.csv',index=False)
    # Mesmo se a validacao preferir um modelo simples, testamos as hipoteses
    # de contexto: coeficientes regularizados NAO sao efeitos causais.
    full=fit_model(train,MODELS['C_contexto'])
    pd.DataFrame({
        'variavel':MODELS['C_contexto'],
        'coef_padronizado_modelo_contextual':full.named_steps['poissonregressor'].coef_,
    }).to_csv(out/'coeficientes_contexto_2026.csv',index=False)
    correlation=[]
    for col in FEATURES:
        rho,p=spearmanr(cur_obs[col],cur_obs.alvo_gols)
        correlation.append({'variavel':col,'spearman_2026':rho,'p_nao_ajustado':p})
    pd.DataFrame(correlation).to_csv(out/'correlacoes_exploratorias_2026.csv',index=False)

    # Tabela depois do primeiro turno oficial: apenas jogos identificados como rodada <=19.
    half=season26[(season26.rodada<=19)&season26.gols_m.notna()]
    first=table_from_played(half,clubs)
    first.to_csv(out/'primeiro_turno_2026.csv',index=False)
    now=table_from_played(played,clubs)
    now.to_csv(out/'classificacao_no_corte_2026.csv',index=False)

    fixtures=[]
    for _,r in remaining.iterrows():
        hx=features_for(r.mandante,r.visitante,True,hist,clubs)
        ax=features_for(r.visitante,r.mandante,False,hist,clubs)
        mu_h=float(np.clip(model.predict(pd.DataFrame([hx])[feats])[0],.15,5.5))
        mu_a=float(np.clip(model.predict(pd.DataFrame([ax])[feats])[0],.15,5.5))
        fixtures.append({'data':r.data,'rodada':r.rodada,'mandante':r.mandante,
                         'visitante':r.visitante,'gols_esperados_m':mu_h,'gols_esperados_v':mu_a})
    upcoming=pd.DataFrame(fixtures)
    # Probabilidades 1x2 pela distribuicao Poisson, sem aproximar usando placar medio.
    from scipy.stats import poisson
    for i,fxr in upcoming.iterrows():
        mat=np.outer(poisson.pmf(np.arange(0,15),fxr.gols_esperados_m),
                     poisson.pmf(np.arange(0,15),fxr.gols_esperados_v))
        mat=mat/mat.sum()
        upcoming.loc[i,'prob_casa']=float(np.tril(mat,-1).sum())
        upcoming.loc[i,'prob_empate']=float(np.trace(mat))
        upcoming.loc[i,'prob_fora']=float(np.triu(mat,1).sum())
    upcoming.to_csv(out/'partidas_projetadas_2026.csv',index=False)

    rng=np.random.default_rng(args.semente)
    n=int(args.simulacoes);c=len(clubs)
    ids={t:i for i,t in enumerate(clubs)}
    base=now.set_index('time').loc[clubs]
    pts=np.tile(base.pontos.to_numpy(dtype=np.int16),(n,1)).copy()
    wins=np.tile(base.vitorias.to_numpy(dtype=np.int16),(n,1)).copy()
    gf=np.tile(base.gols_pro.to_numpy(dtype=np.int16),(n,1)).copy()
    ga=np.tile(base.gols_contra.to_numpy(dtype=np.int16),(n,1)).copy()
    for _,r in upcoming.iterrows():
        h,a=ids[r.mandante],ids[r.visitante]
        x=rng.poisson(r.gols_esperados_m,size=n).astype(np.int16)
        y=rng.poisson(r.gols_esperados_v,size=n).astype(np.int16)
        gf[:,h]+=x; ga[:,h]+=y; gf[:,a]+=y; ga[:,a]+=x
        hh=x>y; aa=x<y; dd=x==y
        pts[:,h]+=(3*hh+dd).astype(np.int16)
        pts[:,a]+=(3*aa+dd).astype(np.int16)
        wins[:,h]+=hh.astype(np.int16);wins[:,a]+=aa.astype(np.int16)
    # Desempates conforme regras principais: pontos, vitorias, saldo, gols pro.
    # Nao inclui confronto direto/cartoes, pela ausencia de dados de sumula.
    noise=rng.uniform(0,0.00001,size=(n,c))
    order=np.lexsort((noise,-gf,-(gf-ga),-wins,-pts),axis=1)
    positions=np.empty_like(order)
    positions[np.arange(n)[:,None],order]=np.arange(1,c+1)[None,:]
    rows=[]
    for t,j in ids.items():
        pp=positions[:,j];p=pts[:,j]
        rows.append({'time':t,'pontos_atuais':int(base.loc[t,'pontos']),
            'jogos_atuais':int(base.loc[t,'jogos']),
            'jogos_restantes':int(((remaining.mandante==t)|(remaining.visitante==t)).sum()),
            'prob_campeao_pct':round(100*np.mean(pp==1),2),
            'prob_rebaixado_pct':round(100*np.mean(pp>=17),2),
            'pontos_medios_finais':round(float(np.mean(p)),2),
            'pontos_p10':int(np.percentile(p,10)),'pontos_p90':int(np.percentile(p,90)),
            'posicao_mediana':int(np.median(pp))})
    result=pd.DataFrame(rows).sort_values('pontos_atuais',ascending=False)
    result.to_csv(out/'probabilidades_2026.csv',index=False)
    print('\nProjecoes:')
    print(result[['time','pontos_atuais','prob_campeao_pct','prob_rebaixado_pct','pontos_medios_finais']].to_string(index=False))
    if not checks.empty:
        print('\nValidacao fora da amostra (2024 e 2025):')
        print(checks.to_string(index=False))
    try:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(11,7))
        plot=result.sort_values('prob_rebaixado_pct',ascending=True)
        ax.barh(plot.time,plot.prob_rebaixado_pct)
        ax.set_xlabel('Probabilidade de rebaixamento (%)')
        ax.set_xlim(0,100)
        ax.set_title(f'Brasileirao 2026 - {n:,} simulacoes | corte {corte:%d/%m/%Y}')
        fig.tight_layout()
        fig.savefig(out/'grafico_probabilidades_2026.png',dpi=160)
        plt.close(fig)
    except ImportError:
        print('matplotlib ausente: grafico nao gerado.')
    print('\nArquivos salvos em:',out.resolve())


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--corte',default=None,help='Data e hora local, ex.: 2026-10-30 10:13')
    p.add_argument('--simulacoes',type=int,default=20000)
    p.add_argument('--semente',type=int,default=2026)
    p.add_argument('--saida',default='saida_brasileirao')
    p.add_argument('--atualizar',action='store_true',help='Rebaixa arquivos de fonte para incorporar novas rodadas')
    args=p.parse_args()
    run(args)
