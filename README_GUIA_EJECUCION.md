# Guía de ejecución: Atari Breakout

Todos los comandos se ejecutan desde la carpeta raíz `rl_games`.

## Instalación

Instala ALE, el preprocesamiento de imágenes y el renderizador:

```powershell
uv sync
```

## Comprobar el entorno

```powershell
uv run rlgames version
uv run rlgames list
uv run rlgames inspect --steps 10
```

El único entorno disponible es `ALE/Breakout-v5` y el único agente del CLI es
DQN. El entorno se preprocesa como cuatro frames grises apilados de 84x84.

## Entrenar

El comando crea un checkpoint nuevo o reanuda uno existente:

```powershell
uv run rlgames train dqn --episodes 500
```

Al terminar, se genera una gráfica PNG con la recompensa por episodio y su
promedio móvil en `graficos/dqn_learning_curve_ALE_Breakout-v5_NNN_episodes.png`.

## Evaluar y renderizar

La evaluación calcula recompensas en 10 episodios. El render abre la ventana
del entorno y reproduce la política guardada:

```powershell
uv run rlgames load dqn --eval
uv run rlgames render dqn --episodes 3
```

Los checkpoints se guardan en `saves/dqn_ALE_Breakout-v5.pt`. Para eliminarlo y
empezar desde cero:

```powershell
uv run rlgames delete dqn
```