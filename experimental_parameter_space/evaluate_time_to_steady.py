from festim_sim import festim_model_standard
import numpy as np


test_temperature_values = np.linspace(400, 750, num=8)

if __name__ == "__main__":

    for T_value in test_temperature_values:
        print(f"Running case {T_value:.1f} K")
        festim_model_standard(
            T=T_value,
            pressure=1e05,
            foldername=f"results/time_to_ss/T={T_value:.1f}K",
            regime="diff",
            atol=1e08,
            final_time=1e07,
        )
