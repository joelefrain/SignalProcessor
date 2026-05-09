# Notebooks

Cada notebook es un caso de uso reproducible. La primera celda inserta `../src` al inicio de `sys.path`, de modo que los imports usan el codigo local del repo:

```python
from signalprocessor.io import read_motion_csv
```

Salidas: `examples/ouputs/case_*`.
