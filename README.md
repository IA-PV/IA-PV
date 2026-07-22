# Tetris AI

Base headless e por turnos para experimentar agentes de IA em Tetris. O núcleo não importa OpenGL, teclado ou qualquer outro componente gráfico. O ambiente `planning-v2` usa configuração imutável, 7-bag determinístico e um contrato de decisão que não expõe o RNG privado aos agentes. A recompensa oficial é a utilidade fixa de linhas `(0, 1, 3, 5, 8)`; score, level e Back-to-Back são telemetria comparável, não um objetivo oculto.

## Instalacao

Requer Python 3.10 ou superior.

```bash
python -m pip install -e ".[dev]"
```

## Comandos

Execute os testes:

```bash
pytest
```

Avalie o agente baseado em estado, objetivo e busca heuristica:

```bash
python -m tetris_ai.cli.evaluate_agents --episodes 50 --max-pieces 500 --search-depth 3 --search-strategy greedy --max-nodes-expanded 2000 --workers 0
```

`--workers 0` distribui partidas independentes entre todos os processadores
logicos, preservando um para o sistema operacional. Use `--workers 1` para a
execucao serial ou um numero explicito, como `--workers 4`, para limitar o uso
de CPU e memoria.

Use `--search-strategy greedy` para priorizar `h(n)`, ou `--search-strategy
astar` para priorizar `g(n) + h(n)`. Por padrao, cada planejamento possui um
orcamento de 2.000 nos; `--max-nodes-expanded 0` remove esse limite.

Cada execucao gera pastas imutaveis em `reports/<agente>/<run_id>/`, contendo o
CSV por episodio, resumo estatistico, configuracoes, metadados de
reprodutibilidade e graficos Matplotlib em SVG e PNG 300 DPI. A comparacao
mostra observacoes individuais, media, intervalo de confianca de 95% e uma
analise pareada por semente. Comparacoes tambem geram
`reports/comparisons/<run_id>/`. O formato completo esta em
[docs/EXPERIMENT_REPORTS.md](docs/EXPERIMENT_REPORTS.md).

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

## Agente baseado em algoritmo genetico

O terceiro agente evolui os pesos de uma politica linear sobre dez atributos
normalizados e usa lookahead limitado por beam search. Cada geracao enfrenta um lote
novo de sementes, e campeoes geracionais sao comparados em sementes separadas de
validacao. Selecao por torneio, crossover aritmetico, mutacao gaussiana e elitismo
produzem as proximas geracoes.

Treine e salve o cromossomo, os hiperparametros e a curva de evolucao:

```bash
python -m tetris_ai.cli.train_genetic_agent --population-size 16 --generations 10 --episodes-per-individual 4 --validation-episodes 12 --max-pieces 200 --lookahead-depth 2 --lookahead-beam-width 2 --seed 0 --workers 0
```

Compare os tres agentes em sementes separadas das usadas no treino:

```bash
python -m tetris_ai.cli.evaluate_agents --episodes 50 --seed 1000000 --max-pieces 500 --search-depth 3 --beam-width 8 --genetic-model reports/genetic_agent/<run_id>/model.json --workers 0
```

Depois de congelar políticas e hiperparâmetros, repita como teste de estresse em
1.000 peças (sem novo ajuste):

```bash
python -m tetris_ai.cli.evaluate_agents --episodes 50 --seed 2000000 --max-pieces 1000 --genetic-model reports/genetic_agent/<run_id>/model.json --workers 0
```

Ao incluir um checkpoint Q-Learning treinado em 500 peças nesse estresse,
adicione `--allow-horizon-transfer`. A opção é explícita para impedir que uma mudança
de contrato seja confundida acidentalmente com o benchmark principal.

Assista ao modelo treinado:

```bash
python -m tetris_ai.cli.watch_agent --agent genetic --genetic-model reports/genetic_agent/<run_id>/model.json --seed 1000000
```

A formulacao, os genes, os hiperparametros, o protocolo experimental e as limitacoes
estao detalhados em [docs/GENETIC_AGENT.md](docs/GENETIC_AGENT.md).

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

No modo padrão `horizon_mode="finite"`, `terminated=True` pode significar
`game_over` ou `horizon_completed`; os motivos explícitos em `info` eliminam a
ambiguidade. No modo alternativo `time_limit`, `max_pieces` produz
`truncated=True` e `piece_limit`, preservando a máscara do próximo estado para
bootstrap. `done` permanece equivalente a `terminated or truncated`.

O agente recebe `DecisionContext`, nao a instancia real de `TetrisEnv`. Simulacoes consomem somente `next_pieces`, a fila publica configuravel. Quando ela acaba, o forward model retorna truncamento por `preview_horizon` em vez de consultar o RNG secreto.

Abra uma janela para assistir o agente jogando:

```bash
python -m tetris_ai.cli.watch_agent --agent state-goal --seed 0 --max-pieces 500 --search-depth 3 --search-strategy greedy --delay-ms 80 --min-delay-ms 18 --level-speed-factor 0.85
```

O visualizador usa o estilo da interface original, com tabuleiro à esquerda, painel lateral, `HOLD`, `NEXT`, score, level, linhas, última jogada e nós expandidos. A animação acelera a cada level: `delay = max(min_delay, round(delay_base * fator^(level - 1)))`. Esse tempo pertence somente à apresentação e não altera o contrato `planning-v2`.

A decisão completa de objetivo, reward shaping, horizonte e protocolo está em
[docs/EXPERIMENT_CONTRACT.md](docs/EXPERIMENT_CONTRACT.md). A especificação formal
do ambiente está em [docs/ENVIRONMENT_SPEC.md](docs/ENVIRONMENT_SPEC.md), e a
arquitetura/migração em
[docs/ARCHITECTURE_AND_MIGRATION.md](docs/ARCHITECTURE_AND_MIGRATION.md).
