from festim_sim import festim_model_standard
import numpy as np


test_temperature_values = np.linspace(450, 750, num=7)
test_pressure_values = np.geomspace(1e2, 1e5, num=10)


if __name__ == "__main__":

    for pressure_value in test_pressure_values:
        for temp_value in test_temperature_values:
            print(f"Testing case P={pressure_value:.2e}, T={temp_value:.0f}")
            festim_model_standard(
                T=temp_value,
                pressure=pressure_value,
                foldername=f"results/parameter_exploration/P={pressure_value:.2e}/T={temp_value:.0f}",
                regime="diff",
                atol=1e08,
                final_time=1e7,
            )
