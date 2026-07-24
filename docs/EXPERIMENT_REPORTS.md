# Relatorios de experimentos

O projeto usa um formato unico e versionado para resultados de avaliacao e
treinamento. Novas execuções usam `schema_version: 3`; relatórios antigos nos
schemas 1 e 2 continuam legíveis e não são reescritos. Cada execução recebe um
`run_id` baseado no instante de inicio,
incluindo microssegundos e fuso horario, por exemplo:

```text
20260717T183045.123456-0300
```

Se esse identificador ja existir, o sistema acrescenta `-02`, `-03` e assim por
diante. Uma execucao existente nunca e sobrescrita.

## Estrutura

Uma avaliacao de varios agentes gera uma pasta por agente e uma comparacao com o
mesmo `run_id`:

```text
reports/
├── random_agent/<run_id>/
│   ├── metadata.json
│   ├── episodes.csv
│   ├── summary.json
│   ├── metrics.svg
│   └── metrics.png
├── state_goal_heuristic_agent/<run_id>/
│   └── ...
└── comparisons/<run_id>/
    ├── metadata.json
    ├── episodes.csv
    ├── summary.csv
    ├── summary.json
    ├── paired_task_return.csv
    ├── paired_score.csv
    ├── paired_lines_removed.csv
    ├── paired_pieces_placed.csv
    ├── paired_reward.csv
    ├── comparison.svg
    ├── comparison.png
    ├── paired_task_return.svg
    ├── paired_task_return.png
    ├── paired_reward.svg
    └── paired_reward.png
```

Treinos usam a mesma pasta do agente. O treino genetico inclui `model.json`,
`history.csv`, `summary.json`, `training.svg` e `training.png`. O treino
Double-DQN inclui `checkpoint.pt`, `episodes.csv`, `summary.json`,
`training.svg` e `training.png`.

## Estatistica e visualizacao

`episodes.csv` preserva as observacoes individuais. No diretorio de comparacao
ele agrega todos os agentes em uma unica tabela; os diretorios individuais
mantem suas tabelas separadas.

Para cada metrica, `summary.json` registra contagem, media, desvio-padrao
amostral, erro-padrao, intervalo de confianca bilateral de 95% da media pela
distribuicao t de Student, mediana, minimo e maximo. Com somente um episodio,
desvio-padrao, erro-padrao e intervalo de confianca sao `null`, pois nao podem
ser estimados honestamente. `summary.csv` achata media, desvio-padrao e limites
do IC 95% para planilhas.

Os graficos sao gerados por Matplotlib com backend headless `Agg`, somente no
processo principal. O SVG e vetorial e adequado para navegador, apresentacao e
edicao; o PNG e salvo em 300 DPI para documentos e video. O
`comparison.svg/png` mostra cada episodio como ponto e sobrepoe media e IC 95%,
incluindo `task_return`, score, linhas, sobrevivência, custo de busca e latência
de decisão. O `paired_task_return.svg/png` liga os resultados obtidos na mesma
semente, explorando corretamente o protocolo de *common random numbers*.
`paired_reward.svg/png` permanece apenas como diagnóstico/compatibilidade para a
recompensa total recebida durante treino.

Os arquivos `paired_<metrica>.csv` calculam, para cada par de agentes, a diferença
`comparação - referência` em sementes comuns, sua incerteza e contagens de vitórias,
empates e derrotas. `task_return` é a métrica primária. O conteúdo estruturado aparece
em `summary.json` sob `paired_comparisons` e `paired_task_return`; os campos antigos de
recompensa total são mantidos como aliases de compatibilidade.

Cada resumo de agente também registra contagem e taxa de `game_over`, conclusão do
horizonte e truncamento. `total_decision_time_seconds`, média, p50 e p95 por decisão
permitem separar qualidade da política de custo computacional. Tempos devem ser
comparados apenas entre execuções com o mesmo número de workers e hardware; nós
expandidos são a medida determinística de esforço de busca.

## Conteudo e reprodutibilidade

`metadata.json` e o manifesto da execucao. Ele
registra:

- instante de inicio e fim e duracao;
- comando executado;
- configuracao completa do experimento e do agente;
- versao do Python, sistema, arquitetura e versao do pacote;
- diretorio de trabalho, commit Git e indicador de arvore de trabalho suja,
  quando disponiveis;
- caminho, tamanho e SHA-256 de modelos de entrada;
- tamanho e SHA-256 de cada artefato produzido.
- biblioteca, versao, backend, formatos, DPI e metodo de incerteza dos graficos.

`episodes.csv` e a fonte tabular granular; CSV e JSON permanecem os dados
canonicos, enquanto SVG e PNG sao representacoes derivadas. Os SVGs podem ser
abertos diretamente no navegador.

O hash identifica exatamente um checkpoint sem duplica-lo em cada avaliacao.
Para preservar uma execucao em outra maquina, copie a pasta do relatorio e os
artefatos de entrada indicados no manifesto.

## Comandos

Avaliar os agentes basicos:

```bash
python -m tetris_ai.cli.evaluate_agents --episodes 50 --seed 1000000 --max-pieces 500 --workers 0
```

Avaliar exatamente os tres agentes do trabalho — busca heuristica, genetico e
Q-table — usando as mesmas sementes:

```bash
python -m tetris_ai.cli.evaluate_agents \
  --agents state-goal genetic q-table \
  --genetic-model reports/genetic_agent/<run_id>/model.json \
  --q-table-checkpoint reports/q_table_agent/<run_id>/checkpoint.pkl \
  --episodes 50 --seed 1000000 --max-pieces 500 \
  --search-depth 3 --beam-width 8 --workers 0
```

O argumento `--agents` define a comparacao. O comando sem ele preserva os
baselines aleatorio e heuristico. Cada agente selecionado que depende de modelo
exige seu artefato correspondente: `--genetic-model`, `--q-table-checkpoint` ou
`--rl-checkpoint`.

Treinar os agentes que possuem CLI de treinamento:

```bash
python -m tetris_ai.cli.train_genetic_agent --generations 10 --max-pieces 200 --validation-max-pieces 500 --seed 0 --workers 0
python -m tetris_ai.cli.train_q_table_agent --episodes 5000 --max-pieces 500 --seed 0
python -m tetris_ai.cli.train_rl --steps 200000 --max-pieces 500 --seed 0
```

O checkpoint canonico do Q-table e salvo como
`reports/q_table_agent/<run_id>/checkpoint.pkl`. O valor de `--max-pieces` no
treino e na avaliacao deve ser o mesmo, pois o estado discretizado inclui o
orcamento restante de pecas.

## Paralelismo

`--workers 1` mantem a execucao serial. `--workers 0` seleciona
automaticamente todos os processadores logicos menos um; um valor positivo,
como `--workers 4`, limita explicitamente a quantidade de processos.

Na comparacao, a unidade de trabalho e um par agente/semente. No treino
genetico, a unidade e um cromossomo avaliado em todo o lote de sementes da
geracao. O processo principal preserva a ordem das tarefas e e o unico que
grava relatorios. Assim, o paralelismo nao altera as regras, sementes, fitness,
desempates ou formato dos artefatos. Em execucoes muito pequenas, o custo de
criar processos pode superar o ganho; nesses casos use `--workers 1`.

O numero solicitado e o numero efetivamente usado ficam em `metadata.json`:
relatorios de treino registram ambos dentro de `execution`, e relatorios de
avaliacao registram `workers_requested` e `worker_processes` dentro de
`experiment`.

`--reports-root` altera a raiz para armazenamento externo. `--model-out`,
`--history-out` e `--checkpoint` criam copias extras para integracoes legadas;
os arquivos dentro da pasta temporal continuam sendo os artefatos canonicos.

## Politica de armazenamento

`reports/` e ignorado pelo Git porque checkpoints e series longas crescem
rapidamente. O diretorio e mantido por `reports/.gitkeep`. Em trabalho de dupla,
usem um armazenamento compartilhado de artefatos (Drive, bucket ou servidor) e
preservem a pasta inteira da execucao. O `run_id`, o commit e os hashes permitem
confirmar que ambos analisam o mesmo experimento.

O diretorio `results/` permanece apenas para compatibilidade com artefatos
antigos; os novos comandos gravam em `reports/` por padrao.
