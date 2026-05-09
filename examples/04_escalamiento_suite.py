from pathlib import Path

from signalprocessor.io import read_motion_csv, read_spectrum_csv, write_json, write_motion_csv, write_spectrum_csv
from signalprocessor.plotting import save_spectrum_plot
from signalprocessor.scaling import scale_suite_to_target


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "ouput" / "case_04_escalamiento_suite"
OUT.mkdir(parents=True, exist_ok=True)

names = ["LIMAEW", "LIMANS", "ATICOEW", "ATICONS", "TARAPACAEW", "TARAPACANS"]
motions = [read_motion_csv(ROOT / "examples" / "data" / "motion" / f"{name}.csv", unit="g", name=name) for name in names]
target_t, target_sa = read_spectrum_csv(ROOT / "examples" / "data" / "response_spectrum" / "EPU_475.csv")
ouput, geo = scale_suite_to_target(
    motions,
    target_t,
    target_sa,
    method="log_least_squares",
    period_range=(0.10, 2.00),
    factor_bounds=(0.20, 5.00),
)

summary = {}
series = {"objetivo": target_sa, "media_geometrica_suite": geo}
for result in ouput:
    write_motion_csv(OUT / f"{result.motion.name}.csv", result.motion, unit="g")
    base_name = result.motion.name.removesuffix("_scaled")
    summary[base_name] = result.factor
    series[base_name] = result.scaled_sa_g

write_json(OUT / "factores_suite.json", summary)
write_spectrum_csv(OUT / "suite_media_geometrica.csv", target_t, geo, extra={"target_sa_g": target_sa})
save_spectrum_plot(OUT / "suite_escalada.png", target_t, series, title="Escalamiento de suite")
print("Factores:", {key: round(value, 4) for key, value in summary.items()})
print(f"Resultados en {OUT}")
