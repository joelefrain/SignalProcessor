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

Correccion con control fuerte de drift en desplazamiento, usando filtro Bessel y polinomio posterior al filtrado:

```powershell
.\.venv\Scripts\python.exe -m signalprocessor.cli correct examples/data/benchmark/uncorrected_motion/CCSP.HNN.._u.smc examples/output/CCSP.HNN.corrected.csv --output-units g --filter-type bessel --highpass 0.08 --lowpass 25 --baseline-order 1 --post-filter-baseline-order 1 --post-filter-final-displacement-constraint
```

Recomendacion automatica restringiendo las familias de filtro evaluadas. Por defecto `--filter-types all` evalua todas: `butterworth`, `cheby1`, `cheby2`, `ellip` y `bessel`. Tambien acepta una sola familia o una lista separada por comas:

```powershell
.\.venv\Scripts\python.exe -m signalprocessor.cli recommend examples/data/benchmark/uncorrected_motion/CCSP.HNN.._u.smc --filter-types bessel,cheby2 --top 10
```

Desde Python, camino automatico:

```python
from signalprocessor.recommendation import recommend_correction_method

recommendation = recommend_correction_method(record, filter_types="bessel, cheby2")
# Use filter_types=None, "all" o "todas" para evaluar todas las familias disponibles.
result_auto = recommendation.best.result
```

Desde Python, camino manual con parametros escritos por el usuario:

```python
from signalprocessor.processing import CorrectionConfig, correct_record

manual_config = CorrectionConfig(
    filter_type="butterworth",
    highpass_hz=0.02,
    lowpass_hz=25.0,
    filter_order=4,
    baseline_order=1,
    post_filter_baseline_order=1,
    post_filter_constrain_final_velocity=True,
    post_filter_constrain_final_displacement=True,
)
result_manual = correct_record(record, manual_config)
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

- `notebooks/01_correccion_senales.ipynb`: correccion SMC por dos caminos, recomendacion automatica y parametros manuales, con comparacion contra USGS. Incluye el caso manual Butterworth 0.02-25 Hz.
- `notebooks/02_escalamiento_lineal.ipynb`: factores lineales para todos los CSV de ejemplo.
- `notebooks/03_ajuste_espectral_wavelet.ipynb`: ajuste espectral iterativo y comparacion contra objetivo.
- `notebooks/04_benchmarks.ipynb`: tablas de benchmark USGS y SeismoMatch.

## Estructura principal

- `records.py`: dataclasses `MotionRecord` y `Spectrum`.
- `io.py`: lectores CSV, SeismoMatch TXT y COSMOS/SMC.
- `processing.py`: baseline polinomial con restricciones discretas de velocidad/desplazamiento, despiking, taper, filtros IIR Butterworth/Chebyshev I/Chebyshev II/eliptico/Bessel y pipeline de correccion.
- `metrics.py`: PGA, PGV, PGD, Arias, duraciones, CAV y RMS. Internamente las metricas se almacenan en SI; use `ground_motion_parameters_to_dict(...)` para tablas en cm/s2, cm/s y cm.
- `spectra.py`: espectros elasticos con Newmark promedio aceleracion.
- `scaling.py`: escalamiento lineal logaritmico y misfit espectral.
- `matching.py`: ajuste espectral iterativo en frecuencia, con wavelets gaussianas-coseno disponibles como modo alternativo.
- `_core.py`: kernels Numba para integracion, derivacion y espectros.
- `_core_cy.pyx`: kernels Cython opcionales.


## Medidas de intensidad y unidades

`compute_ground_motion_parameters(record)` devuelve valores en SI: PGA en m/s2, PGV en m/s, PGD en m, Arias y CAV en m/s. Esto evita mezclar unidades internamente. Para reportes o notebooks use:

```python
from signalprocessor.metrics import ground_motion_parameters_to_dict

row = ground_motion_parameters_to_dict(
    params,
    acceleration_units="cm/s^2",
    velocity_units="cm/s",
    displacement_units="cm",
    cav_units="cm/s",
    suffix_units=True,
)
```

Para comparar con productos USGS separados (`_a`, `_v`, `_d`), use `compute_ground_motion_parameters_from_series(...)` y entregue los canales publicados de velocidad y desplazamiento. No reintegre el canal `_a` si la tabla busca representar los mismos valores mostrados en los graficos de velocidad/desplazamiento USGS.

## Validacion rapida

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Correccion de drift:

- `CorrectionConfig.filter_type` acepta `butterworth`, `cheby1`, `cheby2`, `ellip` y `bessel`.
- `post_filter_baseline_order=1` con `post_filter_constrain_final_displacement=True` aplica una correccion polinomial posterior al filtrado para remover drift terminal introducido por taper/filtro antes de integrar a velocidad y desplazamiento.
- El recomendador evalua por defecto todas las familias de filtro disponibles (`butterworth`, `cheby1`, `cheby2`, `ellip`, `bessel`) y permite restringir el barrido con `filter_types` o `--filter-types`.
- Cuando detecta drift medio/alto agrega candidatos con polinomio post-filtro.

Benchmarks actuales:

- Correccion contra USGS: PGA ratio cercano a 1.0 y RMS normalizado bajo en los tres componentes CCSP.
- Ajuste espectral contra objetivo EPU_475: el modo `matched` queda alrededor de 2-3% RMS log en los registros de ejemplo, comparable con las salidas SeismoMatch incluidas.
