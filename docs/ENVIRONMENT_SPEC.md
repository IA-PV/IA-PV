# Especificacao do ambiente Tetris planning-v1

## Objetivo

`planning-v1` e um ambiente deterministico por seed para comparar agentes de planejamento e, futuramente, aprendizado por reforco e algoritmos geneticos. A unidade temporal e uma peca travada, nao um frame de jogo humano.

## Configuracao e reproducibilidade

`TetrisConfig`, `RewardConfig` e `ScoringConfig` sao dataclasses imutaveis. A configuracao completa gera um `config_id` SHA-256 abreviado, incluido no `reset`, em cada `step` e no CSV de avaliacao.

O gerador de pecas usa 7-bag com RNG privado. `reset(seed=N)` reinicia o fluxo; `reset(seed=None)` continua o fluxo existente, seguindo a semantica de ambientes de RL. Um clone administrativo copia exatamente o estado do RNG para permitir testes e replay deterministas.

## Estado observado

Uma `Observation` e imutavel e contem:

- matriz binaria do tabuleiro;
- peca atual, fila publica `next_pieces` e hold;
- score, level, linhas e pecas colocadas;
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

## Score e recompensa

Score e recompensa sao contratos separados.

O score usa a tabela `(0, 100, 300, 500, 800)` para zero a quatro linhas e o level existente antes da limpeza. Back-to-Back e ativado por Tetris de quatro linhas, preservado por jogadas sem limpeza e quebrado por uma limpeza comum.

A aproximacao de T-Spin por tres cantos fica desativada por padrao porque o ruleset nao registra a ultima rotacao nem implementa SRS. Ela so pode ser habilitada explicitamente em `ScoringConfig`, e a telemetria identifica `three_corner_approximation`.

A recompensa padrao combina linhas e deltas de buracos, altura agregada e bumpiness. `terminated` aplica penalidade de derrota. `truncated` possui penalidade independente, igual a zero por padrao. O shaping pode ser desligado para produzir observacoes e objetivos menos orientados por features manuais.

## Encerramento

`step` retorna:

```text
observation, reward, terminated, truncated, info
```

`terminated` representa game over. `truncated` representa `piece_limit` no ambiente real ou `preview_horizon` no forward model. Os dois podem ser verdadeiros no mesmo passo e todos os motivos aparecem em `termination_reasons`.

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
- truncamento nao recebe penalidade de game over;
- game over e limite podem coexistir;
- T-Spin aproximado exige habilitacao explicita;
- configuracoes invalidas sao rejeitadas.
