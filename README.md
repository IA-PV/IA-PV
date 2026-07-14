# Tetris AI

Base headless e por turnos para experimentar agentes de IA em Tetris. O núcleo não importa OpenGL, teclado ou qualquer outro componente gráfico.

## Instalação

Requer Python 3.10 ou superior.

```bash
python -m pip install -e ".[dev]"
```

## Comandos

Execute os testes:

```bash
pytest
```

Compare os agentes aleatório e heurístico (sem abrir janela):

```bash
python -m tetris_ai.cli.evaluate_agents --episodes 5 --max-pieces 100
```

O CSV é salvo em `results/evaluation.csv`. A arquitetura, a migração e as limitações estão em [docs/ARCHITECTURE_AND_MIGRATION.md](docs/ARCHITECTURE_AND_MIGRATION.md).
