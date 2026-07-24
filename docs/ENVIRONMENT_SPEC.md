# Especificação do ambiente Tetris `planning-v2`

## Objetivo

`planning-v2` é um ambiente determinístico por seed para comparar agentes de
planejamento, aprendizado por reforço e algoritmo genético sob o mesmo objetivo. A
unidade temporal é uma peça travada, não um frame de jogo humano.

## Configuracao e reproducibilidade

`TetrisConfig`, `RewardConfig` e `ScoringConfig` sao dataclasses imutaveis. A configuracao completa gera um `config_id` SHA-256 abreviado, incluido no `reset`, em cada `step` e no CSV de avaliacao.

O gerador de pecas usa 7-bag com RNG privado. `reset(seed=N)` reinicia o fluxo; `reset(seed=None)` continua o fluxo existente, seguindo a semantica de ambientes de RL. Um clone administrativo copia exatamente o estado do RNG para permitir testes e replay deterministas.

## Estado observado

Uma `Observation` e imutavel e contem:

- matriz binaria do tabuleiro;
- peca atual, fila publica `next_pieces` e hold;
- score, level, linhas e peças colocadas;
- horizonte máximo e quantidade de peças restantes;
- estado de Back-to-Back;
- `terminated`, `truncated` e a propriedade derivada `done`;
- mascara booleana do espaco fixo de acoes;
- metricas do tabuleiro no modo `featured`, ou `None` no modo `raw`.

O ruleset padrao mostra cinco pecas futuras. O conteudo restante do 7-bag e o estado do RNG nunca fazem parte da observacao.

## Contrato de decisao e forward model

Agentes recebem `DecisionContext`, composto pela observacao, pelas acoes legais e por `simulate(action)`. O contexto e criado a partir de um clone sem RNG e so pode consumir a fila publica.

Ao esgotar a fila visivel, uma simulacao retorna `truncated=True` com `preview_horizon`. Ela nao inventa uma peca uniforme e nao consulta o futuro real. Assim, uma busca profunda pode encerrar no horizonte conhecido sem data leakage. Um modelo probabilistico consistente com 7-bag pode ser adicionado posteriormente como estrategia explicita do agente.

## Acoes

O espaco fixo possui `8 * width` IDs:

```text
id = hold_offset + rotation * width + column
hold_offset = 0 ou 4 * width
```

Rotacoes inexistentes e colunas invalidas permanecem mascaradas. Cada acao legal faz hard drop vertical e trava exatamente uma peca.

Com `use_hold=False`, a peca atual e colocada. Com `use_hold=True`, a troca de hold ocorre antes e `rotation`/`column` colocam a peca resultante na mesma transicao. Isso evita um passo vazio que alteraria artificialmente o desconto temporal em RL.

## Regras de tabuleiro

- O tabuleiro padrao possui 10 colunas e 20 linhas visiveis.
- Nao existem linhas ocultas de spawn.
- Uma peca precisa caber na linha zero para possuir uma colocacao legal.
- A queda e exclusivamente vertical na rotacao e coluna escolhidas.
- Nao existem gravidade por frame, soft drop, movimentos laterais durante a queda, tucks, SRS ou wall kicks.
- Game over ocorre quando nenhuma colocacao normal ou via hold e possivel.

Essas regras definem um problema de planejamento de colocacoes finais. Elas nao pretendem reproduzir todos os controles do Tetris Guideline.

## Score, recompensa de tarefa e shaping

Score e recompensa sao contratos separados.

O score usa a tabela `(0, 100, 300, 500, 800)` para zero a quatro linhas e o level existente antes da limpeza. Back-to-Back e ativado por Tetris de quatro linhas, preservado por jogadas sem limpeza e quebrado por uma limpeza comum.

A aproximacao de T-Spin por tres cantos fica desativada por padrao porque o ruleset nao registra a ultima rotacao nem implementa SRS. Ela so pode ser habilitada explicitamente em `ScoringConfig`, e a telemetria identifica `three_corner_approximation`.

A recompensa canônica é estacionária e depende apenas das linhas removidas:
`(0, 1, 3, 5, 8)`. Penalidades terminal e de truncamento são zero. Esse valor é
registrado como `task_reward`; seu acumulado, `task_return`, é a medida oficial para
fitness e comparação.

O preset de treino de Q-Learning e Double-DQN acrescenta potential-based reward
shaping (PBRS):

```text
Phi(s) = -(0.50 * holes + 0.05 * aggregate_height + 0.10 * bumpiness)
r_train = task_reward + gamma * Phi(s') - Phi(s), gamma = 1
```

O potencial sucessor é zero em término real do MDP, mas é mantido em truncamento
externo. `task_reward`, `potential_shaping` e `total` aparecem separadamente em
`info["reward"]`. O benchmark usa o preset canônico sem shaping.

## Encerramento

`step` retorna:

```text
observation, reward, terminated, truncated, info
```

O padrão `horizon_mode="finite"` modela um MDP episódico de 500 peças. Ao alcançar o
orçamento, `terminated=True` e o motivo é `horizon_completed`; o orçamento restante
na observação preserva a propriedade de Markov. `game_over` também é um término e os
dois motivos podem coexistir.

Em `horizon_mode="time_limit"`, o mesmo corte é externo: `truncated=True`, motivo
`piece_limit`, e a observação terminal preserva a máscara legal para bootstrap.
`preview_horizon` é sempre um truncamento do forward model público. Todos os motivos
aparecem em `termination_reasons`; código consumidor não deve inferir derrota apenas
do booleano `terminated`.

## Telemetria

`info` e serializavel em JSON e inclui acao/ID, peca e linha de colocacao, metricas antes/depois, componentes da recompensa, score ganho, levels antes/depois, B2B, quantidade de acoes legais, motivos de encerramento, `config_id` e hash do estado publico.

O hash permite comparar trajetorias geradas pela mesma configuracao, seed e sequencia de acoes sem expor o RNG.

## Invariantes testados

- mesma seed e mesmas acoes produzem o mesmo estado;
- clone administrativo e exato e independente;
- forward model nao repoe a fila usando o RNG privado;
- mascara e IDs correspondem exatamente as acoes legais;
- acao invalida nao altera o estado;
- hold e colocacao formam uma unica transicao;
- horizonte finito não recebe penalidade de game over;
- modo de time limit preserva a máscara de bootstrap;
- game over e horizonte podem coexistir;
- PBRS telescopa para o mesmo objetivo no MDP finito;
- T-Spin aproximado exige habilitacao explicita;
- configuracoes invalidas sao rejeitadas.
