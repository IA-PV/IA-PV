# Contrato experimental canônico (`planning-v2`)

Este documento congela a definição usada para treinar, selecionar e comparar os
agentes. A ideia central é simples: o objetivo científico não pode mudar de um
algoritmo para outro, e a recompensa auxiliar de treinamento não pode contaminar o
benchmark.

## Objetivo e recompensa oficial

Uma transição que remove `n` linhas recebe a utilidade de tarefa:

```text
U(n) = (0, 1, 3, 5, 8)[n]
```

O retorno oficial de um episódio é `sum(U(n_t))`. O score Guideline-like continua
sendo calculado e reportado, inclusive level e Back-to-Back, mas é telemetria, não a
função objetivo. Isso evita que a recompensa mude apenas porque o level aumentou e
mantém o mesmo contrato para Random, busca heurística, algoritmo genético, Q-Learning
e Double-DQN.

A configuração canônica tem:

```text
line_rewards       = (0, 1, 3, 5, 8)
terminal_penalty   = 0
truncation_penalty = 0
enable_shaping     = false
```

Game over já tem custo de oportunidade: encerra as chances de pontuar. O fim do
orçamento experimental também não é uma derrota.

## Shaping exclusivo do treino de RL

Q-Learning e Double-DQN podem usar o preset de treino denso. Ele aplica potential-
based reward shaping (PBRS), preservando o objetivo ótimo do MDP finito:

```text
Phi(s) = -(0.50 * holes + 0.05 * aggregate_height + 0.10 * bumpiness)
F(s, s') = gamma * Phi(s') - Phi(s)
r_train = U(lines) + F(s, s')
gamma = 1.0
```

Em um estado terminal, `Phi(s') = 0`. Em truncamento externo, o potencial do próximo
estado é preservado. A telemetria separa `task_reward`, `potential_shaping` e a
recompensa total; relatórios e ranking usam sempre `task_return`.

O algoritmo genético não recebe PBRS. Sua aptidão é a média do `task_return` em
sementes compartilhadas, porque os próprios genes já representam a qualidade do
tabuleiro.

## Horizonte e semântica de término

O padrão é `max_pieces=500` com `horizon_mode="finite"`. O orçamento restante faz
parte da observação; ao colocar a última peça, o episódio termina com
`horizon_completed`. Esse é um estado terminal do MDP e, portanto, não há bootstrap.

O modo alternativo `horizon_mode="time_limit"` representa um corte externo. Nesse
caso, o motivo é `piece_limit`, `truncated=True`, a máscara do próximo estado é
preservada e algoritmos de valor podem fazer bootstrap. `preview_horizon` também é
truncamento externo e existe apenas no forward model público.

`game_over` e `horizon_completed` podem ocorrer na mesma transição. Somente
`game_over` é interpretado como derrota por heurísticas que possuam um custo explícito
de top-out.

## Estado dos agentes aprendentes

O Double-DQN codifica o tabuleiro completo, peças visíveis, hold, métricas públicas,
estado de término e orçamento restante. A máscara fixa de `8 * width` ações é aplicada
tanto na seleção quanto no alvo Double-DQN.

O Q-Learning usa uma discretização auditável contendo buracos, altura agregada,
irregularidade, altura máxima, peça atual, próxima peça, hold e faixa de orçamento
restante. Checkpoints registram a versão desse contrato e versões incompatíveis são
recusadas com uma mensagem para retreinamento.

## Protocolo recomendado

- Desenvolvimento: sementes separadas de treino e validação; nunca selecionar por
  resultados do teste.
- O CLI de avaliação começa por padrão em `1_000_000`, uma partição reservada para
  reduzir colisões acidentais com os lotes sequenciais de treino/validação.
- Algoritmo genético: busca prática em horizonte 200 e reranking dos finalistas no
  horizonte canônico 500.
- Avaliação final: pelo menos 50 sementes pareadas, idealmente 100, em horizonte 500.
- Estresse pós-congelamento: as mesmas políticas em horizonte 1000, sem novo ajuste.
  Checkpoints Q/RL exigem `--allow-horizon-transfer`; o relatório registra essa
  transferência deliberada.
- RL: no mínimo três execuções independentes de treinamento; avaliação das políticas
  congeladas com a recompensa canônica.
- Reportar `task_return`, score, linhas, peças, game-over, conclusão do horizonte,
  tempo de decisão e esforço de busca. Diferenças entre agentes devem ser calculadas
  por semente compartilhada, com intervalo de confiança.

## Regra de compatibilidade

O fingerprint da configuração identifica dimensões, regras, horizonte, recompensa e
score. Artefatos do `planning-v1` podem ser carregados apenas como baselines legados;
eles não são evidência de desempenho sob `planning-v2`. Resultados finais exigem
retreinamento e avaliação no contrato novo.
