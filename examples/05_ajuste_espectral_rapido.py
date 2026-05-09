from pathlib import Path

from signalprocessor.io import read_motion_csv, read_spectrum_csv, write_json, write_motion_csv, write_spectrum_csv
from signalprocessor.metrics import motion_metrics
from signalprocessor.plotting import save_spectrum_plot
from signalprocessor.scaling import frequency_domain_spectral_match


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "ouput" / "case_05_ajuste_espectral_rapido"
OUT.mkdir(parents=True, exist_ok=True)

motion = read_motion_csv(ROOT / "examples" / "data" / "motion" / "LIMAEW.csv", unit="g", name="LIMAEW")
target_t, target_sa = read_spectrum_csv(ROOT / "examples" / "data" / "response_spectrum" / "EPU_475.csv")
result = frequency_domain_spectral_match(
    motion,
    target_t,
    target_sa,
    iterations=3,
    max_factor_per_iteration=1.6,
    smoothing_width=7,
    highpass_hz=0.08,
)

write_motion_csv(OUT / "LIMAEW_ajuste_frecuencia.csv", result.motion, unit="g")
write_json(OUT / "LIMAEW_ajuste_frecuencia.json", {"method": result.method, "metrics": motion_metrics(result.motion)})
write_spectrum_csv(OUT / "LIMAEW_ajuste_frecuencia_espectro.csv", result.periods, result.scaled_sa_g)
save_spectrum_plot(
    OUT / "LIMAEW_ajuste_frecuencia.png",
    result.periods,
    {"ajustado": result.scaled_sa_g, "objetivo": result.target_sa_g},
    title="Ajuste espectral rapido en frecuencia",
)
print(f"Resultados en {OUT}")
