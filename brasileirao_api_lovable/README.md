# Brasileirão Analytics · API Python para Lovable

Este é o **backend FastAPI** que reutiliza os arquivos `motor.py` e `modelo_base.py` do Brasileirão Analytics 2.0. Ele **não** é a interface Streamlit, e **não** é o site Lovable. Serve para dar à interface web acesso aos cálculos reais de Poisson, Monte Carlo e playoffs.

## 1. Testar no seu computador (Windows)

Extraia o ZIP. Abra o PowerShell **na pasta onde estão `main.py`, `motor.py` e `modelo_base.py`** e execute, um comando por vez:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Abra `http://localhost:8000/docs`. A tela documenta os endpoints. Para conferir o teste seguro com clubes **fictícios**, abra:

`http://localhost:8000/api/v1/projecoes?serie=A&demonstracao=true&simulacoes=500`

> `localhost` só funciona no seu computador. **Não** informe `localhost:8000` na configuração da API do Lovable publicado na internet.

## 2. Publicar a API no Render

1. Crie um repositório no GitHub e envie `main.py`, `motor.py`, `modelo_base.py`, `requirements.txt`, `render.yaml` e `.gitignore`. Envie os arquivos descompactados, não apenas o ZIP.
2. Em https://dashboard.render.com , faça **New > Web Service**, conecte o repositório e escolha **Python 3**.
3. Configure **Build Command:** `pip install -r requirements.txt`.
4. Configure **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`.
5. Para reprodução, escolha uma versão Python suportada, por exemplo `3.11.11` via variável de ambiente `PYTHON_VERSION` se o painel permitir; o arquivo `render.yaml` também contém esta configuração.
6. Publique. O Render fornece um link HTTPS `https://NOME-DO-SEU-SERVICO.onrender.com`. **Não substitua `NOME-DO-SEU-SERVICO` até existir um endereço de verdade.**
7. Acesse seu link `/health` e `/docs`. `GET /health` testa somente o funcionamento da API, **não** valida o acesso às fontes reais.

Conforme o plano e o uso, o serviço pode adormecer por inatividade e levar tempo para responder à próxima requisição. Consulte custos e limites no Render.

## 3. API / contratos para o Lovable

### `GET /health`
Retorna `{"status":"ok", ...}` se o servidor estiver executando. **Não confirma** que os dados públicos estão disponíveis.

### `GET /api/v1/series`
Lista Séries A e B e os identificadores de modelo.

### `GET /api/v1/projecoes`
Exemplo com dados **públicos**, apenas quando o Render tiver acesso à fonte:

`/api/v1/projecoes?serie=A&simulacoes=2000`

Exemplo com dados **inteiramente fictícios**:

`/api/v1/projecoes?serie=B&simulacoes=500&demonstracao=true`

Parâmetros opcionais: `serie` (`A`/`B`), `simulacoes` (`500`–`20000`), `semente` (inteiro), `modelo` (`Automático`, `Referencia_mando`, `A_forca`, `B_forma_casa_fora`, `C_contexto`), `corte` (`YYYY-MM-DD`), `demonstracao` (`true`/`false`, padrão `false`).

Campos principais no JSON de resposta:
- `metadata`: fonte, se é fictícia, série, data de corte, modelo selecionado, jogos concluídos, simulações, aviso de que não é placar ao vivo;
- `classificacao_atual`: pontos, vitórias, saldo de gols, partidas, `posicao_atual`;
- `classificacao_projetada`: `time`, `posicao_projetada`, `pontos_atuais`, `pontos_previstos`, `pontos_p10`, `pontos_p90`, `posicao_mais_provavel`, `prob_titulo`, `prob_z4`, `prob_g4`, `prob_acesso`, `prob_playoffs`, `prob_acesso_direto`, etc.;
- `partidas_restantes`: `jogo_indice`, `mandante`, `visitante`, `lambda_m`, `lambda_v`, `prob_m`, `prob_e`, `prob_v`;
- `primeiro_turno`, `coeficientes`, `correlacoes_exploratorias`, `validacao`.

**As probabilidades são porcentagens de 0 a 100, não frações de 0 a 1.** O JSON retorna valores ausentes como `null`.

### `POST /api/v1/cenarios`
Body JSON de exemplo (partida #0 do calendário **pendente**):

```json
{
  "serie":"B",
  "simulacoes":2000,
  "semente":2026,
  "jogo_indice":0,
  "resultado":"Mandante",
  "demonstracao":true
}
```

`resultado` pode ser `Mandante`, `Empate` ou `Visitante`. Retorna `classificacao_cenario` e `deltas`, em relação à previsão sem resultado imposto. Para localizar `jogo_indice`, consulte `partidas_restantes`.

## 4. Prompt para a interface Lovable

> Conecte o site à minha API Python FastAPI, usando a URL-base HTTPS informada. Teste `GET /health`, depois `GET /api/v1/projecoes?serie=A&simulacoes=2000` e `GET /api/v1/projecoes?serie=B&simulacoes=2000`. Leia `metadata`, `classificacao_atual`, `classificacao_projetada`, `partidas_restantes`, `coeficientes`, `correlacoes_exploratorias` e `validacao`. Exiba os dados reais nas tabelas e nos gráficos, preserve os nomes dos campos e mostre mensagem de erro quando uma requisição falhar. `prob_titulo`, `prob_z4`, `prob_acesso` etc. já estão na escala de 0–100%. Adicione o simulador chamando `POST /api/v1/cenarios` com um `jogo_indice` de `partidas_restantes`. Nunca invente números; mostre um aviso explícito quando `metadata.ficticio` for verdadeiro. Indique a data de corte e que o serviço não faz placar ao vivo. Se a API não responder, exiba estado de erro e botão de tentar novamente.

## 5. Limitações importantes

- **Não há servidor publicado neste pacote.** É preciso hospedar a API e depois inserir a URL real no Lovable.
- A fonte pública deve funcionar no ambiente de hospedagem; ainda **não foi validada** daqui com os resultados reais. A API retorna 503 e explicação em caso de falha de importação/modelagem.
- A Série B da atualização pública **não inclui histórico de temporadas anteriores** e é exploratória, sem validação externa adequada.
- Atualização a cada 5 minutos = nova consulta *quando há requisições*, não gols ao vivo. Os dados externos podem atualizar mais lentamente.
- A API calcula regulação com gols independentes e Monte Carlo; não incorpora cartões, decisões disciplinares, todos os desempates residuais e incerteza paramétrica.
- A API permite CORS aberto apenas porque os endpoints não têm autenticação nem dados privados. Se for acrescentar contas ou dados de usuário, implemente autenticação e CORS restritivo.
- A hospedagem gratuita pode ter suspensão por inatividade, limites de CPU/memória e políticas próprias.

Arquivos:

```
main.py           FastAPI e rotas JSON
modelo_base.py    Engenharia de atributos e leitura de resultados
motor.py          Regressões, simulações e playoffs
requirements.txt  Dependências Python
render.yaml       Exemplo de configuração de serviço Render
README.md         Instruções
```

## Testes realizados

Foi executado localmente `TestClient` (FastAPI) com endpoints `/`, `/health`, `/api/v1/series`, `/api/v1/projecoes` com séries A e B **fictícias** e `/api/v1/cenarios`, confirmando JSON e total de 20 equipes. **Não é teste de deploy real nem confirmação de acesso à fonte pública.**
