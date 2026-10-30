"""Motor de regressão Poisson e simulações de campeonatos por rodada.

Todos os indicadores para TREINAMENTO são anteriores ao jogo de destino.
A projeção de calendário usa informações disponíveis no corte escolhido.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from scipy.stats import skellam, spearmanr
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modelo_base import (
    COLUMNS, FEATURES, MODELS, compose_2026, download_fixture,
    download_hist, features_for, fit_model, norm, season_records,
    table_from_played, validate_longitudinal,
)

LABELS = {
    'Referencia_mando': 'Referência · apenas mando',
    'A_forca': 'A · força das equipes',
    'B_forma_casa_fora': 'B · forma e casa/fora',
    'C_contexto': 'C · situação do adversário',
}

@dataclass
class Baseline:
    means: dict

    def predict(self, x):
        return np.array([self.means[int(k)] for k in x['mando']], dtype=float)


def _clean_columns(frame: pd.DataFrame, *, fixture=False, year=2026) -> pd.DataFrame:
    """Aceita o esquema padronizado do projeto, tanto para upload quanto demo."""
    df = frame.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {'data', 'mandante', 'visitante', 'gols_m', 'gols_v'}
    if not required.issubset(df):
        raise ValueError('CSV deve conter data, mandante, visitante, gols_m e gols_v.')
    if 'temporada' not in df:
        df['temporada'] = year
    if 'rodada' not in df:
        df['rodada'] = np.nan
    # Datas em ISO (YYYY-MM-DD) e em DD/MM/YYYY; o parser dá suporte a ambos.
    raw_dates = df['data'].astype(str).str.strip()
    iso = raw_dates.str.match(r'^\d{4}-\d{2}-\d{2}')
    dates = pd.to_datetime(raw_dates.where(iso), errors='coerce')
    dates.loc[~iso] = pd.to_datetime(raw_dates[~iso], dayfirst=True, errors='coerce')
    df['data'] = pd.to_datetime(dates, errors='coerce')
    if (df[['mandante','visitante']].isna().any().any()
            or df['mandante'].astype(str).str.strip().eq('').any()
            or df['visitante'].astype(str).str.strip().eq('').any()):
        raise ValueError('Há nomes de clubes vazios no CSV.')
    df['mandante'] = df['mandante'].map(norm)
    df['visitante'] = df['visitante'].map(norm)
    for c in ['gols_m', 'gols_v', 'rodada', 'temporada']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # Somente no calendário futuro são válidas datas ausentes, quando
    # a partida ainda não tem placar. O modelo conserva essas partidas.
    no_date = df['data'].isna()
    no_score = df['gols_m'].isna() & df['gols_v'].isna()
    if (no_date & (~no_score | (not fixture))).any():
        raise ValueError('Datas inválidas. Use YYYY-MM-DD ou DD/MM/YYYY; '
                         'só jogos futuros sem placar podem ficar sem data.')
    if df[['mandante','visitante']].isna().any().any() or (df.mandante == df.visitante).any():
        raise ValueError('Encontrado confronto com clube ausente ou jogo contra si mesmo.')
    if ((df[['gols_m','gols_v']].notna().sum(axis=1) == 1)).any():
        raise ValueError('Cada partida precisa dos dois placares ou de ambos vazios.')
    if df[['gols_m','gols_v']].dropna().lt(0).any().any():
        raise ValueError('Placares negativos não são válidos.')
    if not df['temporada'].notna().all():
        raise ValueError('Temporada ausente em uma ou mais linhas.')
    df = df[COLUMNS].sort_values(['temporada','data','mandante','visitante']).reset_index(drop=True)
    if fixture:
        if not df.temporada.eq(year).all():
            raise ValueError(f'O calendário deve conter só a temporada {year}.')
        teams = sorted(set(df.mandante)|set(df.visitante))
        if len(teams) != 20 or len(df) != 380:
            raise ValueError(f'Esperados 20 times e 380 jogos, recebidos {len(teams)} times e {len(df)} jogos.')
        if df.duplicated(['mandante','visitante']).any():
            raise ValueError('Há mandante/visitante repetidos; esperada uma partida para cada mando.')
        appearances = pd.concat([df.mandante,df.visitante]).value_counts()
        if not appearances.eq(38).all():
            raise ValueError('Cada equipe precisa ter exatamente 38 partidas.')
    return df


def load_csv_bytes(buffer: bytes, *, fixture: bool, year=2026):
    # Vírgula, ponto-e-vírgula e BOM UTF-8 são comuns em planilhas brasileiras.
    try:
        df = pd.read_csv(BytesIO(buffer), sep=None, engine='python', encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(BytesIO(buffer), sep=None, engine='python', encoding='latin-1')
    if fixture and 'temporada' in df.columns:
        seasons=pd.to_numeric(df['temporada'],errors='coerce').dropna().unique()
        if len(seasons)==1:
            year=int(seasons[0])
    return _clean_columns(df, fixture=fixture, year=year)


def download_public(serie='A'):
    if serie not in ('A', 'B'):
        raise ValueError('Série deve ser A ou B.')
    with tempfile.TemporaryDirectory() as directory:
        p = Path(directory)
        sess = requests.Session()
        sess.headers.update({'User-Agent': 'Mozilla/5.0 (Brasileirao Analytics; pesquisa estatistica)'})
        if serie == 'A':
            historic = download_hist(sess, p/'BRA.csv')
        else:
            # O arquivo BRA.csv é da primeira divisão. NÃO misturamos Série A na
            # amostra da B. Histórico da B pode ser enviado por CSV separadamente.
            historic = pd.DataFrame(columns=COLUMNS)
        fixtures = download_fixture(sess, p/'fixture.html', serie=serie)
    return _clean_columns(historic), _clean_columns(fixtures, fixture=True)


def cut_fixtures(calendar: pd.DataFrame, historic: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    year = int(calendar.temporada.iloc[0])
    h = historic[historic.temporada.eq(year)]
    # Quando o CSV histórico tem placares para a mesma temporada, reconciliar.
    if h.empty:
        out=calendar.copy()
    else:
        out = calendar.merge(
            h[['mandante','visitante','data','gols_m','gols_v']].rename(columns={
                'data':'data_h','gols_m':'gh_h','gols_v':'ga_h'
            }), on=['mandante','visitante'], how='left', validate='one_to_one'
        )
        mismatch=out.gols_m.notna() & out.gh_h.notna() & (
            (out.gols_m != out.gh_h)|(out.gols_v != out.ga_h))
        if mismatch.any():
            raise ValueError('Fontes discordam sobre placar de partida. Confira os CSVs antes de simular.')
        out.loc[out.data_h.gt(asof), ['gh_h','ga_h']] = np.nan
        # Apenas se a partida já aconteceu no corte, podemos recuperar
        # a data do histórico; não inventar data de jogo pendente.
        out['data'] = out['data'].fillna(out['data_h'].where(out.data_h.le(asof)))
        out['gols_m'] = out.gols_m.fillna(out.gh_h)
        out['gols_v'] = out.gols_v.fillna(out.ga_h)
        out=out.drop(columns=['data_h','gh_h','ga_h'])
    out.loc[out.data.gt(asof), ['gols_m','gols_v']] = np.nan
    return out.sort_values(['data','mandante','visitante']).reset_index(drop=True)


def _training_set(history: pd.DataFrame, current: pd.DataFrame):
    y = int(current.temporada.iloc[0])
    all_rec=[]
    for yr, grp in history[(history.temporada<y)&(history.temporada>=2018)].groupby('temporada'):
        full=grp[grp.gols_m.notna() & grp.gols_v.notna()]
        # Não contam temporadas historicamente muito incompletas.
        if len(full) < 270:
            continue
        rec,_,_=season_records(full)
        if not rec.empty: all_rec.append(rec)
    cur_rec, cur_hist, clubs = season_records(current)
    if len(cur_rec)<200:
        raise ValueError('São necessários pelo menos 100 jogos concluídos no campeonato para estimar a primeira versão.')
    all_rec.append(cur_rec)
    train=pd.concat(all_rec, ignore_index=True).dropna(subset=FEATURES+['alvo_gols'])
    return train, cur_rec, cur_hist, clubs


def _select_model(train: pd.DataFrame, selection: str, scores: pd.DataFrame):
    if selection == 'Automático':
        if scores.empty:
            selection='A_forca'
        else:
            selection=scores.groupby('modelo').brier_1x2.mean().idxmin()
    if selection == 'Referencia_mando':
        means=train.groupby('mando').alvo_gols.mean().to_dict()
        return Baseline(means), selection, ['mando']
    return fit_model(train,MODELS[selection]), selection, MODELS[selection]


def prepare(historical: pd.DataFrame, fixtures: pd.DataFrame, cut: pd.Timestamp, choice='Automático', serie='A'):
    if serie not in ('A', 'B'):
        raise ValueError('Série deve ser A ou B.')
    year=int(fixtures.temporada.iloc[0])
    fixtures=_clean_columns(fixtures, fixture=True, year=year)
    historical=_clean_columns(historical)
    current=cut_fixtures(fixtures,historical,pd.Timestamp(cut))
    train, cur_records, team_history, clubs = _training_set(historical,current)
    historic_obs=train[train.temporada.lt(year)]
    score=validate_longitudinal(historic_obs) if len(historic_obs)>500 else pd.DataFrame()
    model,selected,feats=_select_model(train,choice,score)
    played=current[current.gols_m.notna()].copy()
    remaining=current[current.gols_m.isna()].copy()
    standings=table_from_played(played,clubs)
    half=current[(current.rodada<=19)&(current.gols_m.notna())]
    first=table_from_played(half,clubs)
    preds=[]
    for _,r in remaining.iterrows():
        fxh=features_for(r.mandante,r.visitante,True,team_history,clubs)
        fxa=features_for(r.visitante,r.mandante,False,team_history,clubs)
        x=pd.DataFrame([fxh,fxa])
        mus=np.clip(model.predict(x[feats]),.2,5.0)
        h,a=map(float,mus)
        ph=float(skellam.sf(0,h,a))
        pd_=float(skellam.pmf(0,h,a))
        pa=max(0.0,1-ph-pd_)
        preds.append({'data':r.data,'rodada':r.rodada,'mandante':r.mandante,
                      'visitante':r.visitante,'lambda_m':h,'lambda_v':a,
                      'prob_m':ph,'prob_e':pd_,'prob_v':pa})
    forecasts=pd.DataFrame(preds,columns=['data','rodada','mandante','visitante','lambda_m','lambda_v','prob_m','prob_e','prob_v'])
    coefs=pd.DataFrame(columns=['variavel','coeficiente_padronizado'])
    if selected!='Referencia_mando':
        coefs=pd.DataFrame({'variavel':feats,'coeficiente_padronizado':model.named_steps['poissonregressor'].coef_})
    assoc=[]
    for f in FEATURES:
        rho, p = spearmanr(cur_records[f],cur_records.alvo_gols)
        assoc.append({'variavel':f,'spearman':float(rho),'p_exploratorio':float(p)})
    correlations=pd.DataFrame(assoc)
    # Matriz de forças para os dois jogos dos playoffs da Série B.
    # Primeira partida na casa do pior classificado; a segunda na casa do melhor.
    # Calculada no corte, sem conhecer os resultados das simulações.
    playoff_lambdas = None
    if serie == 'B':
        names = sorted(clubs)
        pairs = [(home, away) for home in names for away in names if home != away]
        rows = []
        for home, away in pairs:
            rows.extend([features_for(home, away, True, team_history, clubs),
                         features_for(away, home, False, team_history, clubs)])
        lambdas = np.clip(model.predict(pd.DataFrame(rows)[feats]), .2, 5.0).reshape(-1, 2)
        playoff_lambdas = {pair: tuple(map(float, mu)) for pair, mu in zip(pairs, lambdas)}
    return {'serie':serie, 'playoff_lambdas':playoff_lambdas,
            'year':year,'cut':pd.Timestamp(cut),'current':current,'standings':standings,
            'first':first,'forecasts':forecasts,'score':score,'coefs':coefs,
            'correlations':correlations,'selected':selected,'n_played':len(played),
            'n_historical':len(historic_obs)//2,'n_training':len(train)//2}


def _conditioned_scores(rng, mu_h, mu_a, n, result):
    """Amostra placares da Poisson *condicionados* à vitória/empate/derrota.
    Não fixa arbitrariamente 1×0, 0×0 ou 0×1 para os cenários.
    """
    x=np.empty(n,dtype=np.int16); y=np.empty(n,dtype=np.int16)
    pending=np.arange(n)
    while len(pending):
        count=len(pending)
        a=rng.poisson(mu_h,size=count).astype(np.int16)
        b=rng.poisson(mu_a,size=count).astype(np.int16)
        ok=(a>b) if result=='Mandante' else ((a<b) if result=='Visitante' else (a==b))
        x[pending[ok]]=a[ok]; y[pending[ok]]=b[ok]
        pending=pending[~ok]
    return x,y


def simulate(bundle: dict, n=10000, seed=2026, override: int|None=None, outcome='Modelo'):
    standings=bundle['standings']
    allteams=sorted(standings.time.tolist())
    ids={t:i for i,t in enumerate(allteams)}
    base=standings.set_index('time').loc[allteams]
    c=len(allteams); n=int(n)
    pts=np.broadcast_to(base.pontos.to_numpy(np.int32),(n,c)).copy()
    wins=np.broadcast_to(base.vitorias.to_numpy(np.int32),(n,c)).copy()
    gf=np.broadcast_to(base.gols_pro.to_numpy(np.int32),(n,c)).copy()
    ga=np.broadcast_to(base.gols_contra.to_numpy(np.int32),(n,c)).copy()
    rng=np.random.default_rng(seed)
    for i,r in bundle['forecasts'].iterrows():
        j,k=ids[r.mandante],ids[r.visitante]
        if override is not None and i==override and outcome!='Modelo':
            x,y=_conditioned_scores(rng,r.lambda_m,r.lambda_v,n,outcome)
        else:
            x=rng.poisson(r.lambda_m,size=n).astype(np.int16)
            y=rng.poisson(r.lambda_v,size=n).astype(np.int16)
        gf[:,j]+=x; ga[:,j]+=y; gf[:,k]+=y; ga[:,k]+=x
        hm=x>y; av=x<y; dr=x==y
        pts[:,j]+=(3*hm+dr).astype(np.int32)
        pts[:,k]+=(3*av+dr).astype(np.int32)
        wins[:,j]+=hm.astype(np.int32); wins[:,k]+=av.astype(np.int32)
    # Critérios principais da CBF: pontos, vitórias, saldo, gols marcados.
    # Confronto direto, cartões não simulados; desempate residual aleatório.
    noise=rng.uniform(0,1e-6,size=(n,c))
    ordering=np.lexsort((noise,-gf,-(gf-ga),-wins,-pts),axis=1)
    positions=np.empty_like(ordering)
    positions[np.arange(n)[:,None], ordering] = np.arange(1,c+1)[None,:]
    playoff_counts = np.zeros(c, dtype=np.float64)
    direct_counts = np.zeros(c, dtype=np.float64)
    access_counts = np.zeros(c, dtype=np.float64)
    if bundle.get('serie', 'A') == 'B':
        # 1º e 2º sobem diretamente; 3º x 6º, 4º x 5º disputam
        # duas vagas em ida e volta. Igualdade no agregado: melhor colocado.
        playoff_counts = np.bincount(ordering[:, 2:6].ravel(), minlength=c) / n * 100
        direct_counts = np.bincount(ordering[:, :2].ravel(), minlength=c) / n * 100
        access_counts += direct_counts
        mus = bundle.get('playoff_lambdas')
        if mus is None:
            raise ValueError('Faltam parâmetros dos playoffs da Série B.')
        mh = np.full((c,c), np.nan)
        ma = np.full((c,c), np.nan)
        for (h,a),(x,y) in mus.items():
            mh[ids[h],ids[a]] = x
            ma[ids[h],ids[a]] = y
        for high_rank, low_rank in ((2,5), (3,4)):
            strong = ordering[:,high_rank]
            weak = ordering[:,low_rank]
            # Ida: pior campanha em casa. Volta: melhor campanha em casa.
            leg1_h = rng.poisson(mh[weak,strong])
            leg1_a = rng.poisson(ma[weak,strong])
            leg2_h = rng.poisson(mh[strong,weak])
            leg2_a = rng.poisson(ma[strong,weak])
            goal_strong = leg1_a + leg2_h
            goal_weak = leg1_h + leg2_a
            winner = np.where(goal_strong >= goal_weak, strong, weak)
            access_counts += np.bincount(winner, minlength=c)/n*100
    played=base.jogos
    rows=[]
    for team,j in ids.items():
        pos=positions[:,j];p=pts[:,j]
        rows.append({'time':team,
            'prob_acesso_direto':round(float(direct_counts[j]),2) if bundle.get('serie','A')=='B' else np.nan,
            'prob_playoffs':round(float(playoff_counts[j]),2) if bundle.get('serie','A')=='B' else np.nan,
            'prob_acesso':round(float(access_counts[j]),2) if bundle.get('serie','A')=='B' else np.nan,
            'posicao_mais_provavel':int(np.argmax(np.bincount(pos,minlength=21)[1:])+1),
            'pontos_atuais':int(base.loc[team,'pontos']),
            'jogos':int(played.loc[team]),'prob_titulo':round(100*np.mean(pos==1),2),
            'prob_g4':round(100*np.mean(pos<=4),2),
            'prob_z4':round(100*np.mean(pos>=17),2),
            'pontos_previstos':round(float(np.mean(p)),1),
            'pontos_p10':float(np.percentile(p,10)),
            'pontos_p90':float(np.percentile(p,90)),
            'posicao_media':round(float(np.mean(pos)),1),
            'posicao_mediana':int(np.median(pos)),
            'jogos_restantes':int(38-played.loc[team])})
    out=pd.DataFrame(rows).sort_values(['pontos_previstos','posicao_media'],ascending=[False,True]).reset_index(drop=True)
    out['posicao_projetada'] = np.arange(1, len(out)+1)
    return out


def synthetic_demo(year=2026, cut=None):
    """Calendário simétrico e 7 temporadas de placares FICTÍCIOS.
    É proibido apresentar estas saídas como dados reais do Brasileirão.
    """
    rng=np.random.default_rng(821)
    teams=[f'Clube {i:02d} (FICTÍCIO)' for i in range(1,21)]
    odds=rng.normal(0,.32,size=20)
    circle=list(range(20)); first=[]
    for rr in range(19):
        matches=[]
        for j in range(10):
            a,b=circle[j],circle[-1-j]
            if (j+rr)%2: a,b=b,a
            matches.append((a,b))
        first.append(matches)
        circle=[circle[0],circle[-1],*circle[1:-1]]
    rounds=first+[[(b,a) for a,b in matches] for matches in first]
    result=[]
    for season in range(year-6,year+1):
        for round_no, matches in enumerate(rounds,1):
            for d,(a,b) in enumerate(matches):
                dt=pd.Timestamp(f'{season}-01-25') + pd.Timedelta(days=round_no*8.5+d%2)
                gh=rng.poisson(np.exp(.28+odds[a]-odds[b]))
                ga=rng.poisson(np.exp(-.02+odds[b]-odds[a]))
                if season==year and round_no>32:
                    gh=ga=np.nan
                result.append({'temporada':season,'data':dt,'rodada':round_no,
                    'mandante':teams[a],'visitante':teams[b],'gols_m':gh,'gols_v':ga})
    df=pd.DataFrame(result)
    return (_clean_columns(df[df.temporada<year]),
            _clean_columns(df[df.temporada==year],fixture=True,year=year))
