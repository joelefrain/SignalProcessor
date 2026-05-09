from pathlib import Path

import numpy as np

from signalprocessor.io import read_motion_csv, write_json, write_spectrum_csv
from signalprocessor.metrics import motion_metrics
from signalprocessor.plotting import save_spectrum_plot
from signalprocessor.spectra import response_spectrum


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "ouput" / "case_02_parametros_espectro"
OUT.mkdir(parents=True, exist_ok=True)

motion = read_motion_csv(ROOT / "examples" / "data" / "motion" / "ATICOEW.csv", unit="g", name="ATICOEW")
periods = np.logspace(np.log10(0.02), np.log10(5.0), 120)
spec = response_spectrum(motion, periods, damping=0.05)
metrics = motion_metrics(motion)

write_json(OUT / "ATICOEW_metricas_raw.json", metrics)
write_spectrum_csv(OUT / "ATICOEW_espectro_raw.csv", spec["period_s"], spec["sa_g"], extra={"sd_m": spec["sd_m"], "sv_m_s": spec["sv_m_s"]})
save_spectrum_plot(OUT / "ATICOEW_espectro_raw.png", spec["period_s"], {"ATICOEW raw": spec["sa_g"]})
print(f"PGA={metrics['pga_g']:.4f} g, PGV={metrics['pgv_cm_s']:.2f} cm/s, resultados en {OUT}")
