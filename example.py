"""
Example usage of pydt-utils library

This script demonstrates the main features of the pydt-utils library.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from pydt_utils import (
    # Cycle functions
    calculate_ramp_rate,
    smooth_temperature_data,
    detect_soak_zones,
    segment_thermal_cycle,
    # Brazing functions
    detect_liquidus_crossing,
    validate_brazing_stage,
    calculate_brazing_quality_score,
    # Thermal functions
    calculate_heat_flux,
    calculate_biot_number,
    # Anomaly functions
    detect_temperature_spikes,
    detect_overshoot,
    # Plot functions
    plot_thermal_cycle,
    plot_brazing_stages,
)


def create_sample_brazing_cycle():
    """Create a realistic brazing thermal cycle"""
    time = np.linspace(0, 200, 2000)  # 200 minutes, 2000 points
    
    # Create piecewise thermal cycle
    temp = np.zeros_like(time)
    
    # Ramp up: 0-50 min, heat from 20°C to 520°C at ~10°C/min
    mask1 = time < 50
    temp[mask1] = 20 + 10 * time[mask1]
    
    # Soak: 50-150 min, hold at 520°C
    mask2 = (time >= 50) & (time < 150)
    temp[mask2] = 520
    
    # Ramp down: 150-200 min, cool to 20°C at ~10°C/min
    mask3 = time >= 150
    temp[mask3] = 520 - 10 * (time[mask3] - 150)
    
    # Add some realistic noise
    temp += np.random.normal(0, 1, len(temp))
    
    return time, temp


def main():
    print("=" * 60)
    print("pydt-utils Library Example")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating sample brazing cycle...")
    time, temperature = create_sample_brazing_cycle()
    print(f"   Generated {len(time)} data points over {time[-1]:.0f} minutes")
    
    # Calculate ramp rates
    print("\n2. Calculating ramp rates...")
    rates = calculate_ramp_rate(time, temperature)
    heating_rates = rates[rates > 0.5]
    cooling_rates = rates[rates < -0.5]
    print(f"   Average heating rate: {np.mean(heating_rates):.2f} °C/min")
    print(f"   Average cooling rate: {np.mean(cooling_rates):.2f} °C/min")
    
    # Detect soak zones
    print("\n3. Detecting soak zones...")
    soaks = detect_soak_zones(time, temperature, tolerance=2.0, min_duration=10)
    print(f"   Found {len(soaks)} soak zone(s)")
    for i, soak in enumerate(soaks):
        print(f"   Soak {i+1}: {soak['avg_temp']:.1f}°C for {soak['duration']:.1f} min")
    
    # Segment thermal cycle
    print("\n4. Segmenting thermal cycle...")
    stages = segment_thermal_cycle(time, temperature)
    print(f"   Found {len(stages)} stages:")
    for stage in stages:
        print(f"   - {stage['stage_type']:12s}: {stage['duration']:.1f} min")
    
    # Detect liquidus crossing
    print("\n5. Detecting liquidus crossing...")
    liquidus_temp = 450  # °C (typical silver-based filler)
    crossings = detect_liquidus_crossing(time, temperature, liquidus_temp)
    print(f"   Liquidus crossed {len(crossings)} times")
    for crossing in crossings:
        print(f"   - {crossing['crossing_type']:8s} at t={crossing['time']:.1f} min")
    
    # Validate brazing stage
    print("\n6. Validating brazing stage...")
    validation = validate_brazing_stage(
        time, temperature,
        target_temp=520,
        hold_time=80,
        temp_tolerance=10.0
    )
    print(f"   Valid: {validation['valid']}")
    print(f"   Actual hold time: {validation['actual_hold_time']:.1f} min")
    print(f"   Status: {validation['status']}")
    
    # Calculate quality score
    print("\n7. Calculating brazing quality score...")
    score = calculate_brazing_quality_score(
        time, temperature,
        liquidus_temp=450,
        target_temp=520,
        target_hold_time=80
    )
    print(f"   Overall Score: {score['overall_score']:.1f}/100")
    print(f"   Grade: {score['grade']}")
    print(f"   Liquidus Score: {score['liquidus_score']}/25")
    print(f"   Hold Time Score: {score['hold_time_score']}/30")
    print(f"   Stability Score: {score['stability_score']}/25")
    print(f"   Overshoot Score: {score['overshoot_score']}/20")
    if score['issues']:
        print(f"   Issues: {', '.join(score['issues'])}")
    
    # Detect anomalies
    print("\n8. Detecting anomalies...")
    spikes = detect_temperature_spikes(time, temperature, threshold_std=3.0)
    overshoots = detect_overshoot(time, temperature, target_temp=520, tolerance=10.0)
    print(f"   Temperature spikes: {len(spikes)}")
    print(f"   Overshoots: {len(overshoots)}")
    
    # Thermal calculations
    print("\n9. Thermal physics calculations...")
    Bi = calculate_biot_number(h=10, L=0.01, k=15)
    print(f"   Biot number: {Bi:.4f}")
    if Bi < 0.1:
        print(f"   → Lumped capacitance model is valid")
    
    flux = calculate_heat_flux(temperature_gradient=100, thermal_conductivity=15)
    print(f"   Heat flux (100 K/m gradient): {flux:.1f} W/m²")
    
    # Create visualization
    print("\n10. Creating visualization...")
    fig = plot_thermal_cycle(
        time, temperature,
        stages=stages,
        title="Sample Brazing Cycle",
        save_path="example_cycle.png",
        show=False
    )
    print("   Saved plot to: example_cycle.png")
    
    fig2 = plot_brazing_stages(
        time, temperature,
        liquidus_temp=450,
        target_temp=520,
        save_path="example_brazing.png",
        show=False
    )
    print("   Saved plot to: example_brazing.png")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
