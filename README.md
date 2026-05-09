# SignalProcessor

## Instalacion

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

Para compilar la extension Cython opcional en modo editable:

```powershell
.\.venv\Scripts\python.exe -m pip install Cython
.\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation
```

Si Cython no esta instalado, el paquete usa el fallback Numba/NumPy sin romper imports.

## Uso CLI

Correccion USGS/SMC:

```powershell
.\.venv\Scripts\python.exe -m signalprocessor.cli correct examples/data/benchmark/uncorrected_motion/CCSP.HNN.._u.smc examples/output/CCSP.HNN.corrected.csv --output-units g
```

Espectro de respuesta:

```powershell
.\.venv\Scripts\python.exe -m signalprocessor.cli spectrum examples/data/motion/ATICOEW.csv examples/output/ATICOEW_spectrum.csv --t-min 0.01 --t-max 6
```

Escalamiento lineal:

```powershell
.\.venv\Scripts\python.exe -m signalprocessor.cli scale examples/data/motion/ATICOEW.csv examples/data/response_spectrum/EPU_475.csv examples/output/ATICOEW_scaled.csv --t-min 0.2 --t-max 2.0
```

Ajuste espectral iterativo:

```powershell
.\.venv\Scripts\python.exe -m signalprocessor.cli match examples/data/motion/ATICOEW.csv examples/data/response_spectrum/EPU_475.csv examples/output/ATICOEW_matched.csv
```

## Notebooks

- `notebooks/01_correccion_senales.ipynb`: correccion SMC y comparacion contra USGS.
- `notebooks/02_escalamiento_lineal.ipynb`: factores lineales para todos los CSV de ejemplo.
- `notebooks/03_ajuste_espectral_wavelet.ipynb`: ajuste espectral iterativo y comparacion contra objetivo.
- `notebooks/04_benchmarks.ipynb`: tablas de benchmark USGS y SeismoMatch.

## Estructura principal

- `records.py`: dataclasses `MotionRecord` y `Spectrum`.
- `io.py`: lectores CSV, SeismoMatch TXT y COSMOS/SMC.
- `processing.py`: baseline, despiking, taper, Butterworth y pipeline de correccion.
- `metrics.py`: PGA, PGV, PGD, Arias, duraciones, CAV y RMS.
- `spectra.py`: espectros elasticos con Newmark promedio aceleracion.
- `scaling.py`: escalamiento lineal logaritmico y misfit espectral.
- `matching.py`: ajuste espectral iterativo en frecuencia, con wavelets gaussianas-coseno disponibles como modo alternativo.
- `_core.py`: kernels Numba para integracion, derivacion y espectros.
- `_core_cy.pyx`: kernels Cython opcionales.

## Validacion rapida

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Benchmarks actuales:

- Correccion contra USGS: PGA ratio cercano a 1.0 y RMS normalizado bajo en los tres componentes CCSP.
- Ajuste espectral contra objetivo EPU_475: el modo `matched` queda alrededor de 2-3% RMS log en los registros de ejemplo, comparable con las salidas SeismoMatch incluidas.
