# Arquitetura e migração — Fase 1

## Antes e o que foi removido

O ponto de partida possuía um jogo gráfico com um `Board` que representava células por cores e importava uma constante de interface. OpenGL, callbacks, menus, teclado, loop de FPS e queda por tempo foram descartados na migração. Nada disso é importado por este repositório. O antigo `core/board.py` foi substituído por um tabuleiro numérico independente.

## Nova arquitetura

```text
src/tetris_ai/
  core/        regras puras: tabuleiro, peças, 7-bag e métricas
  env/         Action, Observation, recompensa e TetrisEnv
  agents/      contrato de agente, RandomAgent e HeuristicAgent
  evaluation/  execução de episódios
  cli/         comando de comparação
tests/         testes sem interface
results/       saídas de avaliação (CSV ignorável pelo Git)
```

As dependências seguem `core → env → agents/evaluation → cli`: o núcleo não conhece o ambiente; o ambiente não conhece agentes; não há código de renderização nessa cadeia.

## Ambiente por turnos

`TetrisEnv.reset(seed)` cria um tabuleiro limpo, reinicia o 7-bag e retorna uma observação. `step(Action(rotation, column))` valida uma ação, faz hard drop, trava a peça, limpa linhas, atualiza score/recompensa e promove a próxima peça. `legal_actions()` contém apenas rotações e colunas que podem nascer e cair. `clone()` duplica tabuleiro, estado do 7-bag e contadores sem compartilhar estado.

## Ações e observações

`Action` é um dataclass imutável com `rotation` e `column`. As peças têm todas as rotações únicas pré-calculadas (`O=1`, `I/S/Z=2`, demais=4). `Observation` também é imutável: expõe uma matriz de tuplas com `0`/`1`, peça atual/próxima, score, totais, fim e métricas. As métricas são alturas por coluna, altura agregada/máxima, buracos, bumpiness e linhas da última jogada.

## Score e recompensa

Score do jogo é independente da recompensa: linhas 0–4 rendem respectivamente `0, 100, 300, 500, 800`. A recompensa usa linhas `0, 1, 3, 5, 8`, menos os deltas de buracos (`0.75`), altura agregada (`0.10`) e bumpiness (`0.15`), além de `-10` ao encerrar. O `info` de `step` contém cada termo da decomposição. O episódio acaba por `game_over` (a próxima peça não nasce) ou `piece_limit` (padrão: 500).

## Agentes e avaliação

`RandomAgent` escolhe uniformemente entre ações legais com RNG privado e seed opcional. `HeuristicAgent` testa cada ação em clone e maximiza `linhas - 0.5*altura_agregada - 4*buracos - 0.8*bumpiness`; como as ações são percorridas em ordem, empates são determinísticos.

Após instalar o projeto, execute:

```bash
pytest
python -m tetris_ai.cli.evaluate_agents --episodes 5 --max-pieces 100
```

O segundo comando compara os dois agentes e escreve `results/evaluation.csv`.

## Validações executadas

Na implementação desta fase foram executados, a partir da raiz do repositório:

```bash
python -m pytest
python -m pip install -e .
python -m tetris_ai.cli.evaluate_agents --episodes 5 --max-pieces 100
```

O `pytest` concluiu com 11 testes aprovados. A avaliação criou o CSV com 10 episódios (cinco para cada agente).

## Limitações e próximos passos

Esta fase não oferece renderização, modo humano, hold, soft drop, níveis, T-spins, wall kicks avançados, Gymnasium, algoritmo genético, aprendizado por reforço, rede neural, multiplayer ou dashboards. Próximas etapas podem adicionar uma camada de visualização isolada, adaptador Gymnasium e agentes de busca/aprendizado sem modificar o núcleo.
