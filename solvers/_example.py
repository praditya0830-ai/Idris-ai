"""A real variant from /generate-variants (11.19 kW, 6000 rpm, ratio 8, AMS6265, envelope 200x200x150),
used as the example input and regression case of every housing analysis."""
EXAMPLE_VARIANT = {
    "design": {
        "module_mm": 2.0,
        "pinion_teeth": 18,
        "gear_teeth": 51,
        "face_width_mm": 36,
        "stages": 2,
        "center_distance_mm": 69.0
    },
    "kinematics": {
        "ratio_target": 8,
        "ratio_actual": 8.0278,
        "ratio_error_pct": 0.347,
        "input_torque_nm": 17.809,
        "output_torque_nm": 137.304,
        "output_rpm": 747.4,
        "efficiency": 0.9604
    },
    "loads": {
        "critical_stage": 2,
        "tangential_force_n": 2747.19,
        "pitch_line_velocity_ms": 3.992,
        "K_total": 2.1152,
        "K_v": 1.3017
    },
    "shaft": {
        "output_diameter_mm": 25.0
    },
    "housing_params": {
        "envelope_width": 195,
        "envelope_height": 126,
        "envelope_depth": 98,
        "gear_ratio": 8.0278,
        "wall_thickness": 4.0,
        "bore_diameter": 27.0
    },
    "housing_material": "Al7075-T6"
}
