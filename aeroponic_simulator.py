import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def simulate_aeroponic_absorption(droplet_size, nutrient_concentration, 
                                 root_diameter=0.5, airflow_velocity=0.1, 
                                 temperature=25, exposure_time=60, 
                                 root_density=1000, plot_results=False):
    if not (20 <= droplet_size <= 50):
        raise ValueError("Droplet size must be between 20-50 μm")

    air_viscosity = 1.8e-5 * (1 + 0.00284 * (temperature - 20))
    water_density = 998.0
    gravity = 9.81

    droplet_size_m = droplet_size * 1e-6
    root_diameter_m = root_diameter * 1e-3

    stokes_number = (water_density * droplet_size_m**2 * airflow_velocity) / (18 * air_viscosity * root_diameter_m)
    reynolds_number = (water_density * airflow_velocity * root_diameter_m) / air_viscosity
    settling_velocity = (water_density * droplet_size_m**2 * gravity) / (18 * air_viscosity)
    impact_time = root_diameter_m / airflow_velocity

    interception_param = droplet_size_m / root_diameter_m
    interception_efficiency = 0.6 * interception_param**2 / (1 + interception_param)

    if stokes_number > 0.1:
        impaction_efficiency = (stokes_number**2) / (stokes_number**2 + 0.25)
    else:
        impaction_efficiency = 0

    sedimentation_param = settling_velocity / airflow_velocity
    sedimentation_efficiency = sedimentation_param

    if droplet_size < 30:
        bounce_factor = 1 - 0.01 * (30 - droplet_size)
    else:
        bounce_factor = 1.0

    single_fiber_efficiency = (interception_efficiency + 
                              impaction_efficiency + 
                              sedimentation_efficiency) * bounce_factor

    if nutrient_concentration > 1000:
        concentration_factor = 0.9
    else:
        concentration_factor = 1.0

    film_formation_factor = 1.0
    if droplet_size > 40 and exposure_time > 30:
        film_saturation_time = 30 + (50 - droplet_size)
        if exposure_time > film_saturation_time:
            excess_time = (exposure_time - film_saturation_time) / exposure_time
            film_formation_factor = 1.0 - (0.3 * excess_time)

    absorption_efficiency = single_fiber_efficiency * concentration_factor * film_formation_factor
    root_surface_area = np.pi * root_diameter_m * 0.01 * root_density

    droplet_volume = (4/3) * np.pi * (droplet_size_m/2)**3
    droplet_density = 5e8
    water_flow_rate = airflow_velocity * droplet_volume * droplet_density

    absorption_rate = water_flow_rate * absorption_efficiency * root_surface_area * 60 * 1e6
    nutrients_absorbed = absorption_rate * nutrient_concentration / 1000

    results = {
        'absorption_rate': absorption_rate,
        'efficiency': absorption_efficiency,
        'total_nutrients_absorbed': nutrients_absorbed,
        'parameters': {
            'stokes_number': stokes_number,
            'reynolds_number': reynolds_number,
            'interception_efficiency': interception_efficiency,
            'impaction_efficiency': impaction_efficiency,
            'sedimentation_efficiency': sedimentation_efficiency,
            'film_formation_factor': film_formation_factor,
            'concentration_factor': concentration_factor
        }
    }

    if plot_results:
        plot_absorption_characteristics(droplet_size, nutrient_concentration, results)

    return results

def plot_absorption_characteristics(droplet_size, nutrient_concentration, results):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    params = results['parameters']
    efficiencies = [
        params['interception_efficiency'], 
        params['impaction_efficiency'], 
        params['sedimentation_efficiency']
    ]
    labels = ['Interception', 'Impaction', 'Sedimentation']
    plt.bar(labels, efficiencies)
    plt.title('Efficiency Components')
    plt.ylim(0, 1)

    plt.subplot(2, 2, 2)
    sizes = np.linspace(20, 50, 100)
    effs = []
    for size in sizes:
        res = simulate_aeroponic_absorption(size, nutrient_concentration, plot_results=False)
        effs.append(res['efficiency'])
    plt.plot(sizes, effs)
    plt.scatter([droplet_size], [results['efficiency']], color='red', s=100)
    plt.title('Droplet Size vs. Absorption Efficiency')
    plt.xlabel('Droplet Size (μm)')
    plt.ylabel('Absorption Efficiency')

    plt.subplot(2, 2, 3)
    concs = np.linspace(100, 2000, 100)
    rates = []
    for conc in concs:
        res = simulate_aeroponic_absorption(droplet_size, conc, plot_results=False)
        rates.append(res['absorption_rate'])
    plt.plot(concs, rates)
    plt.scatter([nutrient_concentration], [results['absorption_rate']], color='red', s=100)
    plt.title('Nutrient Concentration vs. Absorption Rate')
    plt.xlabel('Nutrient Concentration (ppm)')
    plt.ylabel('Absorption Rate (ml/min)')

    plt.subplot(2, 2, 4)
    plt.axis('off')
    summary_text = (
        f"Absorption Summary:\n\n"
        f"Droplet Size: {droplet_size} μm\n"
        f"Nutrient Concentration: {nutrient_concentration} ppm\n\n"
        f"Absorption Rate: {results['absorption_rate']:.4f} ml/min\n"
        f"Efficiency: {results['efficiency']:.4f}\n"
        f"Nutrients Absorbed: {results['total_nutrients_absorbed']:.4f} mg/min\n\n"
        f"Stokes Number: {results['parameters']['stokes_number']:.4f}\n"
        f"Reynolds Number: {results['parameters']['reynolds_number']:.4f}\n"
    )
    plt.text(0.1, 0.1, summary_text, fontsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    droplet_size = 35
    nutrient_concentration = 500
    results = simulate_aeroponic_absorption(
        droplet_size=droplet_size,
        nutrient_concentration=nutrient_concentration,
        plot_results=True
    )
    print(f"Absorption rate: {results['absorption_rate']:.4f} ml/min")
    print(f"Absorption efficiency: {results['efficiency']:.4f}")
    print(f"Nutrients absorbed: {results['total_nutrients_absorbed']:.4f} mg/min")