from pathlib import Path

from signalprocessor.io import read_motion_csv, read_spectrum_csv, write_json, write_motion_csv, write_spectrum_csv
from signalprocessor.metrics import motion_metrics
from signalprocessor.plotting import save_spectrum_plot
from signalprocessor.scaling import scale_motion_to_target


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "ouput" / "case_03_escalamiento_lineal"
OUT.mkdir(parents=True, exist_ok=True)

motion = read_motion_csv(ROOT / "examples" / "data" / "motion" / "LIMANS.csv", unit="g", name="LIMANS")
target_t, target_sa = read_spectrum_csv(ROOT / "examples" / "data" / "response_spectrum" / "EPU_475.csv")
result = scale_motion_to_target(
    motion,
    target_t,
    target_sa,
    method="log_least_squares",
    period_range=(0.10, 2.00),
    factor_bounds=(0.20, 5.00),
)

write_motion_csv(OUT / "LIMANS_escalado.csv", result.motion, unit="g")
write_json(OUT / "LIMANS_escalamiento.json", {"factor": result.factor, "method": result.method, "metrics": motion_metrics(result.motion)})
write_spectrum_csv(OUT / "LIMANS_espectro_escalado.csv", result.periods, result.scaled_sa_g)
save_spectrum_plot(
    OUT / "LIMANS_escalamiento.png",
    result.periods,
    {"registro": result.record_sa_g, "escalado": result.scaled_sa_g, "objetivo": result.target_sa_g},
    title=f"LIMANS alpha={result.factor:.3f}",
)
print(f"Factor LIMANS={result.factor:.4f}. Resultados en {OUT}")
