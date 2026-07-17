# Tetris AI

Base headless e por turnos para experimentar agentes de IA em Tetris. O nucleo nao importa OpenGL, teclado ou qualquer outro componente grafico.

## Instalacao

Requer Python 3.10 ou superior.

```bash
python -m pip install -e ".[dev]"
```

Para usar o agente de aprendizado por reforco baseado em PyTorch, instale o
extra de RL:

```bash
python -m pip install -e ".[dev,rl]"
```

## Comandos

Execute os testes:

```bash
pytest
```

Compare o agente aleatorio com o agente baseado em estado, objetivo e busca heuristica:

```bash
python -m tetris_ai.cli.evaluate_agents --episodes 5 --max-pieces 100 --search-depth 3 --beam-width 8
```

O CSV e salvo em `results/evaluation.csv` com score, linhas removidas, recompensa, motivo de termino, profundidade efetiva, largura do feixe e nos expandidos.

Abra uma janela para assistir o agente jogando:

```bash
python -m tetris_ai.cli.watch_agent --agent state-goal --seed 0 --max-pieces 500 --search-depth 3 --beam-width 8 --delay-ms 80
```

O visualizador usa o estilo da interface original, com tabuleiro a esquerda, painel lateral, `HOLD`, `NEXT` desenhado, score, level, linhas, ultima jogada e nos expandidos pela busca.

A arquitetura, a migracao e as limitacoes estao em [docs/ARCHITECTURE_AND_MIGRATION.md](docs/ARCHITECTURE_AND_MIGRATION.md).
