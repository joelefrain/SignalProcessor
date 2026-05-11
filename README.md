# SignalProcessor

SignalProcessor es un proyecto Python para procesamiento rapido y trazable de
acelerogramas de terremoto. Cubre dos flujos principales:

- correccion de senales sismicas, incluyendo linea base, filtrado, integracion,
  medidas de intensidad y recomendacion automatica de parametros;
- escalamiento sismico hacia un espectro objetivo mediante escalamiento lineal,
  ajuste en dominio de Fourier y matching espectral con wavelets en el dominio
  del tiempo.

El diseno sigue el marco teorico de `docs/` y trabaja internamente en unidades
SI. Los CSV de ejemplo pueden estar en `g`; al leerlos se convierten a m/s2 y
los espectros se reportan tambien en `g`.

## Instalacion local

```powershell
python -m pip install -e ".[dev]"
```

## Estructura

```text
src/signalprocessor/       API principal
examples/data/             registros, espectros objetivo y benchmarks
examples/config/           configuraciones reproducibles
notebooks/                 casos de uso ejecutables
tests/                     pruebas rapidas de regresion
```

## Casos de uso principales

### Correccion con parametros directos

```python
from signalprocessor import read_motion_csv, process_motion

motion = read_motion_csv("examples/data/motion/LIMANS.csv", acceleration_unit="g")
result = process_motion(
    motion,
    baseline={"method": "polynomial", "order": 1},
    filtering={"highpass_hz": 0.26, "lowpass_hz": 20.0, "order": 4},
    recommend=False,
)
```

### Correccion con recomendacion automatica

```python
from signalprocessor import load_json, process_motion

config = load_json("examples/config/correction.json")
result = process_motion(motion, config=config, recommend=True)
print(result.summary())
```

### Escalamiento hacia espectro objetivo

```python
from signalprocessor import read_target_spectrum_csv, compare_scaling_methods

target = read_target_spectrum_csv("examples/data/target_response_spectrum/EPU_475.csv")
comparison = compare_scaling_methods(
    motion,
    target.periods_s,
    target.sa_g,
    methods=("linear", "frequency", "wavelet"),
)
print(comparison.summary)
```

## Notebooks

Ejecuta los casos de uso desde:

- `notebooks/01_correccion_senales.ipynb`
- `notebooks/02_escalamiento_sismico.ipynb`
- `notebooks/03_benchmarks.ipynb`

## CLI

```powershell
signalprocessor correct examples/data/motion/LIMANS.csv --config examples/config/correction.json
signalprocessor scale examples/data/motion/LIMANS.csv examples/data/target_response_spectrum/EPU_475.csv --method wavelet
signalprocessor benchmark --examples-root examples
```
