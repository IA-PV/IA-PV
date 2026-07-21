# Arquitetura e migracao do Tetris AI

## Organizacao

```text
src/tetris_ai/
  execution.py paralelismo deterministico compartilhado
  core/        tabuleiro, tetrominos, 7-bag e metricas
  env/         configuracoes, acoes, observacoes, contexto e TetrisEnv
  agents/      contratos e politicas dos agentes
  evaluation/  execucao e resultados de episodios
  training/    treinamento e artefatos dos agentes que aprendem
  reporting/   manifests, estatistica, tabelas e graficos Matplotlib headless
  visualization/ visualizador Tkinter isolado
  cli/         comandos de avaliacao e visualizacao
tests/         testes unitarios e de integracao
reports/       Relatorios versionados por agente e execucao
results/       Artefatos legados
```

As dependencias seguem `core -> env -> agents/evaluation -> training/reporting -> cli`. O nucleo nao importa agentes, interface grafica ou bibliotecas de RL.

## Limites de confianca

`TetrisEnv` e o ambiente real e o unico proprietario do RNG. `clone()` produz uma copia administrativa exata para determinismo e replay. Esse clone nao e entregue aos agentes.

`DecisionContext` e a fronteira publica dos agentes. Ele contem observacao e acoes legais e oferece simulacoes construidas sem RNG, limitadas a fila visivel. Essa separacao evita que uma busca descubra a sequencia secreta do 7-bag.

## Transicao

`reset(seed)` retorna `(observation, info)`. `step(action)` retorna `(observation, reward, terminated, truncated, info)`. A propriedade `done` permanece disponível como `terminated or truncated`. O motivo explícito diferencia `game_over`, término do MDP (`horizon_completed`) e cortes externos (`piece_limit`/`preview_horizon`).

Cada transicao trava uma peca. Hold e colocacao sao atomicos, portanto `total_pieces_placed`, numero de decisoes e passos temporais possuem a mesma unidade.

## Configuracao

`TetrisConfig` centraliza dimensões, horizonte e sua semântica, tamanho da fila, hold,
modo de observação e versão do ruleset. `RewardConfig` e `ScoringConfig` isolam
recompensa e score. Todas são imutáveis, validadas, serializáveis e participam do
fingerprint. `canonical_reward_config()` define o benchmark limpo;
`rl_training_reward_config()` acrescenta PBRS somente ao treino de RL.

O horizonte restante pertence à observação. Assim, `planning-v2` é um MDP finito
quando `horizon_mode="finite"`; `time_limit` mantém a alternativa de truncamento
externo e fornece a máscara de ações sucessoras necessária ao bootstrap.

## Performance

O ambiente cacheia colocacoes legais por estado, reutiliza as metricas anteriores no calculo de recompensa, acessa celulas sem reconstruir matrizes durante T-Spin e clona o grid com copias explicitas de linhas.

Avaliacoes usam processos independentes porque busca e simulacao sao trabalho
majoritariamente limitado por CPU. Na comparacao, cada tarefa representa um par
agente/semente; no algoritmo genetico, um cromossomo e avaliado em todo o lote
de sementes dentro do mesmo worker. Somente o processo principal seleciona,
agrega e grava artefatos. `executor.map` preserva a ordem de entrada, mantendo
o comportamento deterministico da execucao serial.

Relatorios usam Matplotlib com backend `Agg` somente depois que os workers
terminam. Os dados canonicos permanecem em CSV/JSON; SVG e PNG sao derivados.
O renderizador usa paleta acessivel, PNG em 300 DPI, IDs SVG deterministas,
observacoes individuais e intervalos de confianca de 95% pela distribuicao t de
Student. A versao da biblioteca e os parametros do renderizador sao registrados
no manifesto.

Bitboards permanecem uma opcao futura. A migracao so deve ocorrer depois de profiling demonstrar que o tabuleiro e o gargalo dominante, acompanhada por testes diferenciais contra esta implementacao de referencia.

## Visualizacao e avaliacao

O visualizador usa `describe_action` para animar a colocação sem acessar o tabuleiro
mutável. A velocidade pertence exclusivamente à interface. O avaliador fornece
`DecisionContext`, acumula separadamente `task_return` e recompensa de treino, e
registra conclusão do horizonte, game over, custo de busca e latência de decisão.

O Double-DQN recebe o tabuleiro completo e o orçamento restante, aplica máscara tanto
na política quanto no alvo Double-DQN e não faz bootstrap em `terminated`. Um
truncamento externo continua o bootstrap. O Q-Learning segue a mesma semântica e sua
chave inclui altura máxima, peças visíveis, hold e faixa de orçamento. Ambos usam
`gamma=1` no contrato finito e versionam sua representação nos checkpoints.

Execute:

```bash
python -m pytest
python -m tetris_ai.cli.evaluate_agents --episodes 50 --max-pieces 500 --search-depth 3 --beam-width 8 --workers 0
```

## Limitacoes intencionais

O ruleset `planning-v2` não possui gravidade em tempo real, linhas ocultas, soft drop,
tucks, SRS, wall kicks ou T-Spin oficial. A aproximação de T-Spin existe apenas como
opção desativada. O algoritmo genético evolui uma política linear normalizada com
lookahead limitado. Ainda não há adaptador concreto de `gymnasium.Env`; o contrato de
cinco retornos permite adicioná-lo sem acoplar o núcleo.

Modelos `planning-v1` permanecem legíveis como baselines históricos, mas não são
comparáveis ao contrato novo. Q-table e Double-DQN recusam checkpoints com schema,
dimensões, espaço de ações ou horizonte incompatíveis; o procedimento correto é
retreinar. O modelo genético legado pode ser carregado, porém precisa ser reavaliado e
não deve ser apresentado como modelo selecionado por fitness `planning-v2`.

Q-table e Double-DQN recusam por padrão um horizonte diferente do checkpoint. Para o
teste de estresse pós-congelamento, `--allow-horizon-transfer` libera deliberadamente
a transferência: ambas as representações codificam orçamento em fração/faixas
relativas, e o relatório preserva os horizontes de origem e avaliação. Essa opção não
autoriza misturar o resultado com o benchmark principal de 500 peças.

Consulte [GENETIC_AGENT.md](GENETIC_AGENT.md) para a formulacao evolutiva, o protocolo de treinamento e as limitacoes.

Consulte [ENVIRONMENT_SPEC.md](ENVIRONMENT_SPEC.md) para a especificacao completa das regras e invariantes.

Consulte [EXPERIMENT_REPORTS.md](EXPERIMENT_REPORTS.md) para o schema dos relatorios e a politica de armazenamento.

Consulte [EXPERIMENT_CONTRACT.md](EXPERIMENT_CONTRACT.md) para a decisão canônica de
objetivo, shaping, horizonte, seeds e protocolo de benchmark.
