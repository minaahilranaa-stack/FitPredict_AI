
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def generate_dataset(n_rows=15000, seed=42):
    rng = np.random.default_rng(seed)

    age = np.clip(np.rint(rng.normal(36, 13, n_rows)), 18, 75).astype(int)
    gender = rng.choice(
        ["Female", "Male", "Other"],
        n_rows,
        p=[0.49, 0.49, 0.02]
    )

    height = np.where(
        gender == "Male",
        rng.normal(175, 7, n_rows),
        np.where(
            gender == "Female",
            rng.normal(162, 6.5, n_rows),
            rng.normal(168, 8, n_rows)
        )
    )
    height = np.clip(height, 145, 200).round(1)

    exercise_days = rng.choice(
        range(8),
        n_rows,
        p=[0.07, 0.09, 0.13, 0.19, 0.20, 0.17, 0.12, 0.03]
    )

    exercise_hours = np.clip(
        exercise_days * rng.normal(0.65, 0.22, n_rows)
        + rng.normal(0.2, 0.35, n_rows),
        0, 12
    )

    intensity = np.array([
        rng.choice(
            ["Low", "Moderate", "High"],
            p=(
                [0.48, 0.39, 0.13] if d < 2 else
                [0.20, 0.53, 0.27] if d < 5 else
                [0.10, 0.48, 0.42]
            )
        )
        for d in exercise_days
    ])

    workout_type = np.array([
        rng.choice(
            ["Walking", "Cardio", "Strength", "Mixed", "Yoga"],
            p=(
                [0.35, 0.20, 0.10, 0.15, 0.20]
                if intensity[i] == "Low" else
                [0.18, 0.25, 0.22, 0.25, 0.10]
                if intensity[i] == "Moderate" else
                [0.08, 0.28, 0.28, 0.31, 0.05]
            )
        )
        for i in range(n_rows)
    ])

    steps = np.clip(
        np.rint(rng.normal(
            4800 + exercise_days * 1050, 2200, n_rows
        )),
        300, 25000
    ).astype(int)

    sleep = np.clip(
        rng.normal(
            7 - np.maximum(age - 40, 0) * 0.008
            + np.maximum(exercise_days - 3, 0) * 0.04,
            1.0, n_rows
        ),
        3, 10.5
    )

    water = np.clip(
        rng.normal(1.7 + exercise_hours * 0.13, 0.45, n_rows),
        0.4, 5.5
    )

    bmi = np.clip(
        rng.normal(25.2 - exercise_days * 0.20, 4.4, n_rows),
        16.5, 42
    )

    weight = np.clip(bmi * (height / 100) ** 2, 42, 180)
    bmi = weight / (height / 100) ** 2

    heart_rate = np.clip(
        np.rint(rng.normal(
            78 - exercise_days * 1.6
            - (intensity == "High") * 2,
            8, n_rows
        )),
        42, 115
    ).astype(int)

    systolic = np.clip(
        np.rint(
            108 + age * 0.34 + (bmi - 23) * 0.55
            + rng.normal(0, 11, n_rows)
        ),
        85, 190
    ).astype(int)

    diastolic = np.clip(
        np.rint(
            67 + (bmi - 23) * 0.25
            + rng.normal(0, 7, n_rows)
        ),
        50, 115
    ).astype(int)

    calories = np.clip(
        np.rint(
            1750 + (weight - 65) * 8
            + exercise_hours * 90
            + rng.normal(0, 260, n_rows)
        ),
        1000, 4200
    ).astype(int)

    protein = np.clip(
        np.rint(weight * rng.normal(1.15, 0.25, n_rows)),
        25, 250
    ).astype(int)

    carbs = np.clip(
        np.rint(calories * rng.normal(0.48, 0.06, n_rows) / 4),
        70, 600
    ).astype(int)

    fat = np.clip(
        np.rint(calories * rng.normal(0.29, 0.04, n_rows) / 9),
        20, 180
    ).astype(int)

    consistency = np.clip(
        np.round(exercise_days / 7 * 75
                 + rng.normal(15, 12, n_rows), 1),
        0, 100
    )

    cardio = np.clip(
        np.rint(exercise_hours * rng.uniform(20, 42, n_rows)),
        0, 600
    ).astype(int)

    strength = np.clip(
        np.rint(
            exercise_hours * rng.uniform(10, 32, n_rows)
            * np.isin(workout_type, ["Strength", "Mixed", "Cardio"])
        ),
        0, 500
    ).astype(int)

    goals = np.array([
        rng.choice(
            ["General Fitness", "Weight Management",
             "Build Muscle", "Improve Endurance"],
            p=(
                [0.40, 0.25, 0.12, 0.23]
                if d < 3 else
                [0.30, 0.18, 0.25, 0.27]
            )
        )
        for d in exercise_days
    ])

    intensity_score = np.select(
        [intensity == "Low", intensity == "Moderate"],
        [0, 0.8],
        default=1.5
    )

    score = (
        0.0015 * (steps - 5000)
        + 0.42 * exercise_days
        + 0.28 * exercise_hours
        + 0.30 * (sleep - 6.5)
        + 0.55 * intensity_score
        - 0.19 * np.abs(bmi - 23)
        - 0.045 * (heart_rate - 70)
        + rng.normal(0, 1.15, n_rows)
    )

    q1, q2 = np.quantile(score, [0.34, 0.68])

    fitness_level = np.where(
        score < q1, "Beginner",
        np.where(score < q2, "Intermediate", "Advanced")
    )

    df = pd.DataFrame({
        "User_ID": [f"FP{i:06d}" for i in range(1, n_rows + 1)],
        "Age": age,
        "Gender": gender,
        "Height_cm": height,
        "Weight_kg": np.round(weight, 1),
        "BMI": np.round(bmi, 2),
        "Resting_Heart_Rate": heart_rate,
        "Blood_Pressure_Systolic": systolic,
        "Blood_Pressure_Diastolic": diastolic,
        "Daily_Steps": steps,
        "Sleep_Hours": np.round(sleep, 1),
        "Water_Intake_Liters": np.round(water, 2),
        "Exercise_Days_Per_Week": exercise_days,
        "Exercise_Hours_Per_Week": np.round(exercise_hours, 2),
        "Exercise_Intensity": intensity,
        "Workout_Type": workout_type,
        "Cardio_Minutes_Per_Week": cardio,
        "Strength_Training_Minutes_Per_Week": strength,
        "Daily_Calorie_Intake": calories,
        "Protein_Grams": protein,
        "Carbohydrates_Grams": carbs,
        "Fat_Grams": fat,
        "Workout_Consistency": consistency,
        "Fitness_Level": fitness_level,
        "Fitness_Goal": goals
    })

    for col, rate in {
        "Sleep_Hours": 0.012,
        "Water_Intake_Liters": 0.01,
        "Protein_Grams": 0.008,
        "Workout_Type": 0.006
    }.items():
        mask = rng.random(n_rows) < rate
        df.loc[mask, col] = np.nan

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path("data/fitness_dataset.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(args.rows, args.seed)
    df.to_csv(output, index=False)

    print("Dataset generated successfully!")
    print("Saved to:", output.resolve())
    print("Shape:", df.shape)
    print("\nFitness categories:")
    print(df["Fitness_Level"].value_counts())
    print("\nMissing values:", df.isna().sum().sum())


if __name__ == "__main__":
    main()
