# Arquitetura e migracao - Fase 1

## Antes e o que foi removido

O ponto de partida possuia um jogo grafico com um `Board` que representava celulas por cores e importava uma constante de interface. OpenGL, callbacks, menus, teclado, loop de FPS e queda por tempo foram descartados na migracao. Nada disso e importado pelo nucleo. O antigo `core/board.py` foi substituido por um tabuleiro numerico independente.

## Nova arquitetura

```text
src/tetris_ai/
  core/        regras puras: tabuleiro, pecas, 7-bag e metricas
  env/         Action, Observation, recompensa e TetrisEnv
  agents/      contrato de agente, RandomAgent e StateGoalHeuristicAgent
  evaluation/  execucao de episodios
  visualization/ janela Tkinter para acompanhar agentes
  cli/         comandos de avaliacao e visualizacao
tests/         testes sem interface
results/       saidas de avaliacao (CSV ignoravel pelo Git)
```

As dependencias seguem `core -> env -> agents/evaluation -> cli`: o nucleo nao conhece agentes nem renderizacao; o ambiente continua reutilizavel para experimentos de IA.

## Ambiente por turnos

`TetrisEnv.reset(seed)` cria um tabuleiro limpo, reinicia o 7-bag e retorna uma observacao. `step(Action(rotation, column))` valida uma acao, faz hard drop, trava a peca, limpa linhas, atualiza score/recompensa e promove a proxima peca. `step(Action.hold())` guarda ou troca a peca atual sem travar bloco no tabuleiro. `clone()` duplica tabuleiro, 7-bag, hold, B2B e contadores sem compartilhar estado.

## Acoes e observacoes

`Action` continua aceitando `rotation` e `column`, mas agora tambem possui `use_hold` e o construtor `Action.hold()`. `Observation` expoe matriz, peca atual, proxima peca, peca em hold, `can_hold`, score, level, linhas, pecas, B2B ativo, fim e metricas do tabuleiro.

As metricas continuam sendo alturas por coluna, altura agregada/maxima, buracos, bumpiness e linhas da ultima jogada.

## Score, level, T-Spin e B2B

O level e derivado de linhas limpas: `total_lines_cleared // 10 + 1`. O score de linhas e multiplicado pelo level atual apos a limpeza. A deteccao de T-Spin ocorre ao travar uma peca `T` quando 3 dos 4 cantos diagonais ao centro da peca estao preenchidos ou fora do tabuleiro.

Back-to-Back ativa em Tetris de 4 linhas ou T-Spin com limpeza. Jogadas especiais consecutivas recebem multiplicador `1.5`; uma limpeza comum quebra a sequencia. O `info` de `step` inclui `score_gain`, `level`, `is_t_spin`, `back_to_back_active`, `back_to_back_bonus_applied` e `hold_used`.

A recompensa segue separada do score e continua voltada a avaliacao de episodio: linhas, deltas de buracos, altura agregada, bumpiness e penalidade terminal.

## Agentes e avaliacao

`RandomAgent` escolhe uniformemente entre acoes legais com RNG privado e seed opcional. Ele fica como baseline simples.

`StateGoalHeuristicAgent` implementa o paradigma do enunciado: agente baseado em estado, objetivo e busca heuristica.

- Estado: `AgentState` guarda ultima observacao, plano escolhido, decisoes tomadas, profundidade efetiva e nos expandidos.
- Objetivo: `TetrisGoal` avalia somente metricas do tabuleiro, sem somar a recompensa do ambiente.
- Busca: o agente usa beam search. Ele ranqueia todas as acoes imediatas, mas aprofunda apenas as melhores `beam_width`.
- Dificuldade simulada: a profundidade diminui conforme o level sobe. Level 1-4 usa `search_depth`, level 5-9 limita para 2, level 10+ limita para 1.

`HeuristicAgent` continua existindo como nome compativel, herdando `StateGoalHeuristicAgent`.

O avaliador registra score, linhas, pecas, recompensa total, motivo de encerramento, profundidade maxima, profundidade efetiva, beam width, decisoes, nos expandidos, media de nos por decisao e maior custo de decisao.

Execute:

```bash
pytest
python -m tetris_ai.cli.evaluate_agents --episodes 5 --max-pieces 100 --search-depth 3 --beam-width 8
```

## Visualizacao grafica

A visualizacao fica isolada em `visualization/` e usa `tkinter`. Ela observa o agente jogar sem alterar o contrato do ambiente: as decisoes continuam passando por `Agent.select_action(env)` e as transicoes continuam em `TetrisEnv.step(action)`.

```bash
python -m tetris_ai.cli.watch_agent --agent state-goal --seed 0 --max-pieces 500 --search-depth 3 --beam-width 8 --delay-ms 80
```

O viewer mostra tabuleiro no estilo original, `HOLD`, `NEXT`, score, level, linhas, ultima acao, reward, valor heuristico e nos expandidos.

## Validacoes

Foram executados:

```bash
python -m pytest
python -m tetris_ai.cli.evaluate_agents --episodes 1 --max-pieces 20 --search-depth 3 --beam-width 4
python -m tetris_ai.cli.watch_agent --help
```

## Limitacoes e proximos passos

Esta fase ainda nao oferece modo humano, soft drop, wall kicks avancados, Gymnasium, algoritmo genetico, aprendizado por reforco, rede neural, multiplayer ou dashboards. Proximas etapas podem adicionar esses recursos sem modificar o nucleo principal.
