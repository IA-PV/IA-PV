# Arquitetura e migracao do Tetris AI

## Organizacao

```text
src/tetris_ai/
  core/        tabuleiro, tetrominos, 7-bag e metricas
  env/         configuracoes, acoes, observacoes, contexto e TetrisEnv
  agents/      contratos e politicas dos agentes
  evaluation/  execucao e resultados de episodios
  visualization/ visualizador Tkinter isolado
  cli/         comandos de avaliacao e visualizacao
tests/         testes unitarios e de integracao
results/       CSVs gerados
```

As dependencias seguem `core -> env -> agents/evaluation -> cli`. O nucleo nao importa agentes, interface grafica ou bibliotecas de RL.

## Limites de confianca

`TetrisEnv` e o ambiente real e o unico proprietario do RNG. `clone()` produz uma copia administrativa exata para determinismo e replay. Esse clone nao e entregue aos agentes.

`DecisionContext` e a fronteira publica dos agentes. Ele contem observacao e acoes legais e oferece simulacoes construidas sem RNG, limitadas a fila visivel. Essa separacao evita que uma busca descubra a sequencia secreta do 7-bag.

O agente heuristico transforma esse contrato em um `TetrisSearchProblem`
explicito. Seu estado contem matriz, peca corrente, fila publica (o estado do
gerador que e observavel), hold e a posicao final da peca travada. As acoes sao
as colocacoes finais por rotacao e coluna; portanto cada transicao trava uma
peca e acrescenta uma unidade ao custo de caminho `g(n)`.

Como Tetris nao possui uma vitoria terminal, o objetivo e local ao horizonte:
travar a peca atual e as pecas de lookahead visiveis no tabuleiro de menor
custo. A heuristica de custo e `h(n) = 35.6 * buracos + 0.51 *
altura_agregada`, com penalidade adicional para game over. `HeuristicSearchAgent`
mantem um plano interno e o reutiliza antes de formular o proximo problema. A
fronteira e uma fila de prioridade para busca gulosa (`h`) ou A* (`g+h`), nunca
um feixe de acoes. Um orcamento configuravel de expansoes protege o
planejamento de lookahead profundo; ele pode ser removido em experimentos
exhaustivos.

## Transicao

`reset(seed)` retorna `(observation, info)`. `step(action)` retorna `(observation, reward, terminated, truncated, info)`. A propriedade `done` permanece disponivel como `terminated or truncated`.

Cada transicao trava uma peca. Hold e colocacao sao atomicos, portanto `total_pieces_placed`, numero de decisoes e passos temporais possuem a mesma unidade.

## Configuracao

`TetrisConfig` centraliza dimensoes, limite, tamanho da fila, hold, modo de observacao e versao do ruleset. `RewardConfig` e `ScoringConfig` isolam recompensa e score. Todas sao imutaveis, validadas e serializaveis.

## Performance

O ambiente cacheia colocacoes legais por estado, reutiliza as metricas anteriores no calculo de recompensa, acessa celulas sem reconstruir matrizes durante T-Spin e clona o grid com copias explicitas de linhas.

Bitboards permanecem uma opcao futura. A migracao so deve ocorrer depois de profiling demonstrar que o tabuleiro e o gargalo dominante, acompanhada por testes diferenciais contra esta implementacao de referencia.

## Visualizacao e avaliacao

O visualizador usa `describe_action` para animar a colocacao sem acessar o tabuleiro mutavel. A velocidade de animacao pode cair por level ate um piso configuravel, mas pertence exclusivamente a interface: nao participa de `TetrisConfig`, do hash de estado, do RNG, da recompensa ou da avaliacao. O avaliador fornece `DecisionContext` aos agentes e registra `terminated`, `truncated` e `config_id`, alem das metricas de busca existentes.

Execute:

```bash
python -m pytest
python -m tetris_ai.cli.evaluate_agents --episodes 5 --max-pieces 100 --search-depth 3 --search-strategy greedy
```

## Limitacoes intencionais

O ruleset `planning-v1` nao possui gravidade em tempo real, linhas ocultas, soft drop, tucks, SRS, wall kicks ou T-Spin oficial. A aproximacao de T-Spin existe apenas como opcao desativada. Ainda nao ha adaptador concreto de `gymnasium.Env`, rede neural, algoritmo genetico ou aprendizado por reforco; o contrato de cinco retornos, o espaco fixo e a mascara deixam essa integracao preparada sem acoplar o nucleo a uma biblioteca externa.

Consulte [ENVIRONMENT_SPEC.md](ENVIRONMENT_SPEC.md) para a especificacao completa das regras e invariantes.
