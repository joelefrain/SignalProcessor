# SismoSignal

Proyecto Python para correccion de acelerogramas, filtrado de ruido, calculo de parametros/espectros y escalamiento sismico. Esta implementado con `numpy`/`scipy` y usa `numba` cuando esta disponible para acelerar el calculo de espectros SDOF por Newmark.

## Instalacion

```powershell
python -m pip install -e .
```

## Casos de uso incluidos

Los casos de uso viven como notebooks en `notebooks/`:

- `01_correccion_filtrado.ipynb`
- `02_parametros_espectro.ipynb`
- `03_escalamiento_lineal.ipynb`
- `04_escalamiento_suite.ipynb`
- `05_ajuste_espectral_rapido.ipynb`

Tambien se puede usar la CLI:

```powershell
sismosignal process --config examples/config/process_lima.toml
sismosignal scale --config examples/config/scale_lima.toml
sismosignal suite --config examples/config/suite_scaling.toml
```

Si Windows no tiene la carpeta de scripts de Python en el `PATH`, usa la forma equivalente:

```powershell
python -m signalprocessor.cli process --config examples/config/process_lima.toml
```

Los ouputados se escriben en `examples/ouputs/`: CSV procesados, espectros, metricas JSON y figuras PNG.

## Unidades

Los CSV de entrada de `examples/data/motion` se leen como aceleracion en `g` por defecto. Internamente la libreria convierte a `m/s2` para integrar y reporta PGA/Sa en `g`, PGV en `cm/s` y PGD en `cm`.

## Flujo implementado

1. Lectura de acelerogramas sin cabecera (`time, accel`).
2. Correccion de linea base constante, lineal o polinomica, con opcion de restricciones terminales `v(T)=0` y `u(T)=0`.
3. Taper, padding y filtros Butterworth high-pass, low-pass o band-pass con fase cero.
4. Integracion trapezoidal a velocidad/desplazamiento.
5. Parametros: PGA, PGV, PGD, Arias, duracion significativa, CAV, RMS y duraciones por umbral.
6. Espectro elastico SDOF por Newmark-beta.
7. Escalamiento lineal a un periodo, por minimos cuadrados lineales, por minimos cuadrados logaritmicos y escalamiento de suites.
8. Ajuste espectral rapido aproximado en frecuencia para exploracion. Para entrega normativa debe validarse con controles adicionales.

## Nota tecnica

El procesamiento sismico no es unico. El filtro define el rango confiable del registro, y el escalamiento no reemplaza la seleccion fisica por magnitud, distancia, mecanismo, sitio, duracion e intensidad energetica.
