# Agente evolutivo

## Enquadramento no trabalho

Este repositorio usa o **Contexto I — ambiente proprio** do enunciado. Portanto, o vetor
de 128 bytes da RAM do Atari, exclusivo do Contexto II/PettingZoo, nao faz parte desta
solucao. O terceiro agente aprende no mesmo ambiente `planning-v1` usado pelos agentes
aleatorio e de busca heuristica.

## Formulacao computacional

- **Estado observado:** tabuleiro, peca atual, fila publica, hold, mascara de acoes e
  metricas publicas do tabuleiro.
- **Acao:** uma colocacao legal final, formada por rotacao, coluna e uso opcional do
  hold. Uma acao sempre trava exatamente uma peca.
- **Objetivo:** maximizar a recompensa total obtida durante um episodio.
- **Recompensa:** e definida pelo ambiente e combina linhas removidas com variacoes de
  buracos, altura e irregularidade; game over recebe penalidade separada.
- **Aptidao (fitness):** media da soma das recompensas de episodios completos.

## Cromossomo e politica

O cromossomo e um vetor de dez pesos reais. Para cada acao legal, o agente usa o
forward model publico para simular a colocacao, normaliza os atributos e calcula:

```text
valor(acao) = soma(peso_i * atributo_i)
```

| Gene | Atributo normalizado | Interpretacao esperada |
|---|---|---|
| `lines_cleared` | linhas / 4 | peso positivo favorece limpezas |
| `aggregate_height` | soma das alturas / area | peso negativo mantem o tabuleiro baixo |
| `holes` | buracos / area | peso negativo evita espacos inacessiveis |
| `bumpiness` | desnivel / maximo teorico | peso negativo favorece superficie regular |
| `max_height` | maior altura / altura do tabuleiro | peso negativo reduz risco |
| `game_over` | 0 ou 1 | peso negativo evita derrota imediata |
| `use_hold` | 0 ou 1 | custo ou beneficio geral de usar hold |
| `hold_store_i` | 0 ou 1 | distingue guardar uma peca I |
| `hold_retrieve_i` | 0 ou 1 | distingue retirar uma peca I do hold |
| `i_well_match` | reducao normalizada de poco | recompensa usar I para reduzir um poco real |

Os tres ultimos genes tornam o hold contextual. O agente pode aprender pesos distintos
para guardar e recuperar a peca I, enquanto `i_well_match` informa se a colocacao da I
realmente reduziu um poco. Isso e mais expressivo que um unico bonus plano de hold, mas
continua sendo uma aproximacao linear interpretavel.

## Lookahead limitado

A politica executa beam search usando os mesmos pesos evoluidos. Em cada nivel, todas
as acoes legais sao avaliadas e somente as melhores candidatas do feixe continuam para
o proximo nivel:

```text
valor(plano) = valor(a1) + desconto * valor(a2) + ...
```

Os padroes sao profundidade 2, feixe 4 e desconto 0,95. Isso permite aceitar uma
colocacao localmente pior quando a proxima peca compensa a decisao. A busca usa somente
a fila publica; ao esgotar o preview, termina em `preview_horizon` sem consultar o RNG
privado. Profundidade e largura maiores aumentam rapidamente o custo do treinamento.

## Protocolo de sementes e generalizacao

Todos os individuos de uma geracao usam o mesmo lote de sementes, garantindo uma
comparacao justa por *common random numbers*. O lote muda deterministicamente na
geracao seguinte e nao se sobrepoe aos anteriores. Assim, a populacao enfrenta varias
sequencias de pecas em vez de otimizar sempre as mesmas partidas.

Os elites de cada geracao sao guardados. Ao final, os candidatos unicos sao avaliados
uma unica vez em sementes de validacao separadas; o melhor resultado de validacao vira
o modelo salvo. As sementes de teste usadas para o relatorio ainda devem ser diferentes
das sementes de treino e validacao.

Como cada geracao enfrenta partidas diferentes, o melhor fitness de treino pode subir
ou descer entre geracoes. Isso nao representa regressao automaticamente: os valores nao
foram medidos na mesma amostra. A comparacao final correta e o fitness de validacao e,
depois, o desempenho nas sementes externas de teste.

## Algoritmo genetico

1. Inicializa pesos aleatorios uniformes entre -1 e 1 e normaliza cada vetor para norma
   Euclidiana 1, removendo a escala redundante da politica linear.
2. Avalia a populacao no lote de sementes exclusivo da geracao.
3. Copia os melhores individuos por elitismo.
4. Seleciona pais por torneio.
5. Aplica crossover aritmetico com `filho = alfa * pai1 + (1-alfa) * pai2`.
6. Aplica mutacao gaussiana independente por gene e normaliza o descendente.
7. Seleciona o modelo final nas sementes separadas de validacao.

O crossover aritmetico produz interpolacoes suaves entre pesos reais e tende a preservar
relacoes entre genes. Ele nao e universalmente superior ao uniforme: pode reduzir
diversidade quando os pais sao parecidos. A mutacao gaussiana continua sendo a principal
fonte de exploracao local.

## Execucao

Treino rapido para validar a instalacao:

```bash
python -m tetris_ai.cli.train_genetic_agent --population-size 6 --generations 2 --episodes-per-individual 1 --validation-episodes 2 --max-pieces 30 --lookahead-depth 1
```

Treino padrao com lookahead:

```bash
python -m tetris_ai.cli.train_genetic_agent --population-size 24 --generations 20 --episodes-per-individual 3 --validation-episodes 8 --max-pieces 200 --lookahead-depth 2 --lookahead-beam-width 4 --lookahead-discount 0.95 --seed 0
```

Comparacao em sementes nao usadas no treino nem na validacao:

```bash
python -m tetris_ai.cli.evaluate_agents --episodes 20 --seed 1000 --max-pieces 200 --genetic-model results/genetic_agent.json
```

Visualizacao:

```bash
python -m tetris_ai.cli.watch_agent --agent genetic --genetic-model results/genetic_agent.json --seed 1000
```

Para demonstrar um estagio inicial, selecione uma geracao registrada:

```bash
python -m tetris_ai.cli.watch_agent --agent genetic --genetic-model results/genetic_agent.json --genetic-generation 0 --seed 1000
```

## Hiperparametros principais

- `episodes_per_individual`: tamanho do lote de treino de cada geracao.
- `validation_episodes`: partidas separadas usadas para escolher o modelo final.
- `lookahead_depth`: quantidade maxima de colocacoes planejadas.
- `lookahead_beam_width`: planos parciais continuados em cada nivel.
- `lookahead_discount`: importancia relativa das colocacoes futuras.
- `population_size` e `generations`: capacidade e duracao da evolucao.
- `mutation_rate` e `mutation_stddev`: frequencia e intensidade da exploracao.
- `elite_count` e `tournament_size`: preservacao e pressao seletiva.

## Protocolo recomendado para o relatorio e o video

- Preserve o JSON, que registra configuracao, lotes de treino, sementes de validacao,
  politica de busca, cromossomo e metricas.
- Reserve sementes externas para o teste final; nao escolha hiperparametros pelos
  resultados de teste.
- Compare genetico, busca heuristica e aleatorio nas mesmas sementes.
- Reporte media e dispersao de recompensa, score, linhas e pecas colocadas, alem da
  curva de fitness por geracao. O CSV preserva os resultados individuais.
- Compare uma geracao inicial, o modelo final e situacoes de falha no visualizador.

## Compatibilidade e limitacoes

Artefatos antigos de sete genes continuam carregando: os tres novos genes recebem peso
zero e a profundidade permanece 1, reproduzindo a politica anterior.

A politica ainda e linear, e beam search nao garante o plano globalmente otimo. Os
genes contextuais de hold focam a estrategia comum da peca I, mas nao representam todas
as interacoes entre pecas e tabuleiro. Sementes rotativas reduzem sobreajuste, nao o
eliminam. Lookahead, mais episodios e populacoes maiores melhoram a capacidade do
experimento ao custo de processamento. Paralelismo e checkpoints retomaveis ainda nao
fazem parte desta implementacao.
