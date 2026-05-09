# Ejemplos

Los scripts generan salidas en `examples/ouput`.

- `01_correccion_filtrado.py`: corrige baseline, filtra, integra, calcula metricas y espectro.
- `02_parametros_espectro.py`: calcula parametros y espectro SDOF de un registro sin modificar.
- `03_escalamiento_lineal.py`: escala un registro al espectro objetivo `EPU_475.csv`.
- `04_escalamiento_suite.py`: escala seis registros y calcula la media geometrica de la suite.
- `05_ajuste_espectral_rapido.py`: aplica un ajuste espectral aproximado en frecuencia.

Los archivos de configuracion TOML equivalentes estan en `examples/config`.
