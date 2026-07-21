# Tetris AI

Base headless e por turnos para experimentar agentes de IA em Tetris. O nucleo nao importa OpenGL, teclado ou qualquer outro componente grafico. O ambiente usa um ruleset de planejamento versionado, configuracao imutavel, 7-bag deterministico e um contrato de decisao que nao expoe o RNG privado aos agentes.

## Instalacao

Requer Python 3.10 ou superior.

```bash
python -m pip install -e ".[dev]"
```

Para usar o agente de aprendizado por reforco baseado em PyTorch, instale o
extra de RL:

```bash
python -m pip install -e ".[dev,rl]"
```

O agente RL usa somente o `DecisionContext` publico e aprende com as
transicoes retornadas pelo ambiente real. Treine sem abrir o visualizador:

```bash
python -m tetris_ai.cli.train_rl --steps 100000 --checkpoint results/rl_agent.pt
```

O aquecimento aleatorio, a mascara de acoes e o replay sao configurados pelo
comando. O checkpoint gerado pertence ao esquema atual de observacao e acoes;
checkpoints anteriores a essa integracao nao sao compativeis.

## Comandos

Execute os testes:

```bash
pytest
```

Compare o agente aleatorio com o agente baseado em estado, objetivo e busca heuristica:

```bash
python -m tetris_ai.cli.evaluate_agents --episodes 5 --max-pieces 100 --search-depth 3 --search-strategy greedy
```

O CSV e salvo em `results/evaluation.csv` com score, linhas removidas, recompensa, motivo de termino, profundidade efetiva, estrategia de busca e nos expandidos.

## Agente de busca heuristica

`HeuristicSearchAgent` formula um `TetrisSearchProblem` a cada vez que o
plano interno acaba. Um estado de busca contem a matriz do tabuleiro, a peca
corrente, a fila publica de proximas pecas (o unico estado do gerador que o
agente pode conhecer), o hold e a posicao final da ultima peca travada. Cada
acao e uma colocacao final `(rotacao, coluna)`, logo seu custo de caminho e
`g(n) = 1` por peca colocada.

O objetivo e travar a peca atual — e, quando ha lookahead, as pecas visiveis
planejadas — no menor custo de tabuleiro. A heuristica minimizada e:

```text
h(n) = 35.6 * buracos + 0.51 * altura_agregada
```

Use `--search-strategy greedy` para priorizar `h(n)`, ou `--search-strategy
astar` para priorizar `g(n) + h(n)`. Por padrao, cada planejamento possui um
orcamento de 2.000 nos; `--max-nodes-expanded 0` remove esse limite.

## Contrato do ambiente

Cada acao representa uma colocacao final e trava exatamente uma peca. Uma acao com `use_hold=True` faz a troca e coloca a peca resultante na mesma transicao. O espaco possui `8 * width` IDs estaveis e a observacao fornece uma mascara para as combinacoes validas.

```python
from tetris_ai.env import TetrisConfig, TetrisEnv

env = TetrisEnv(config=TetrisConfig(preview_count=5), seed=0)
observation, reset_info = env.reset(0)

while not observation.done:
    context = env.decision_context()
    action = context.legal_actions[0]
    observation, reward, terminated, truncated, info = env.step(action)
```

`terminated=True` significa game over. `truncated=True` indica um limite externo, normalmente `max_pieces`; truncamento nao recebe a penalidade de derrota por padrao. `done` permanece como propriedade de conveniencia equivalente a `terminated or truncated`.

O agente recebe `DecisionContext`, nao a instancia real de `TetrisEnv`. Simulacoes consomem somente `next_pieces`, a fila publica configuravel. Quando ela acaba, o forward model retorna truncamento por `preview_horizon` em vez de consultar o RNG secreto.

Abra uma janela para assistir o agente jogando:

```bash
python -m tetris_ai.cli.watch_agent --agent state-goal --seed 0 --max-pieces 500 --search-depth 3 --search-strategy greedy --delay-ms 80 --min-delay-ms 18 --level-speed-factor 0.85
```

O visualizador usa o estilo da interface original, com tabuleiro a esquerda, painel lateral, `HOLD`, `NEXT` desenhado, score, level, linhas, ultima jogada e nos expandidos pela busca. A animacao acelera a cada level: `delay = max(min_delay, round(delay_base * fator^(level - 1)))`. Esse tempo pertence somente a apresentacao e nao altera as regras, os agentes, a recompensa ou os resultados do ambiente `planning-v1`.

A especificacao formal do ambiente esta em [docs/ENVIRONMENT_SPEC.md](docs/ENVIRONMENT_SPEC.md). A arquitetura, a migracao e as limitacoes estao em [docs/ARCHITECTURE_AND_MIGRATION.md](docs/ARCHITECTURE_AND_MIGRATION.md).
