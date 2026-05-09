from pathlib import Path

from signalprocessor.io import read_motion_csv, write_json, write_motion_csv, write_spectrum_csv
from signalprocessor.plotting import save_motion_plot, save_spectrum_plot
from signalprocessor.processing import ProcessConfig, process_motion


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "ouput" / "case_01_correccion_filtrado"
OUT.mkdir(parents=True, exist_ok=True)

motion = read_motion_csv(ROOT / "examples" / "data" / "motion" / "LIMAEW.csv", unit="g", name="LIMAEW")
config = ProcessConfig(
    baseline_method="polynomial",
    baseline_order=1,
    enforce_zero_end=True,
    highpass_hz=0.08,
    lowpass_hz=18.0,
    filter_order=4,
    taper_fraction=0.02,
    pad_seconds=5.0,
    spectrum_min_period=0.02,
    spectrum_max_period=5.0,
    spectrum_points=100,
)
result = process_motion(motion, config)

write_motion_csv(OUT / "LIMAEW_corregido_filtrado.csv", result.filtered.motion, unit="g")
write_json(OUT / "LIMAEW_metricas.json", result.metrics | {"baseline": result.baseline.info, "filter": result.filtered.info})
write_spectrum_csv(OUT / "LIMAEW_espectro.csv", result.spectrum["period_s"], result.spectrum["sa_g"])
save_motion_plot(
    OUT / "LIMAEW_corregido_filtrado.png",
    result.filtered.motion,
    result.velocity_m_s,
    result.displacement_m,
    title="LIMAEW corregido y filtrado",
)
save_spectrum_plot(OUT / "LIMAEW_espectro.png", result.spectrum["period_s"], {"Sa 5%": result.spectrum["sa_g"]})
print(f"Resultados escritos en {OUT}")
