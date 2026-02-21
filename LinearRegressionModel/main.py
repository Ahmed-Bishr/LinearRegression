from ModelCalculations import ModelCalculations as mc
import numpy as np


def main():
    x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_train = np.array([3.0, 5.0, 7.0, 9.0, 11.0])

    model = mc(
        w=0.0,
        b=0.0,
        x=x_train,
        y=y_train,
        iteration=1000,
        alpha=0.1,
        m=len(x_train)
    )

    print("--- Starting Training ---")

    model.linear_regression()

    print("--- Training Complete ---")

    print(f"Calculated Weight (w): {model.w:.4f}")
    print(f"Calculated Bias   (b): {model.b:.4f}")
    prediction = model.w * 10 + model.b
    print(f"Prediction for x = 10: {prediction:.2f}")


if __name__ == "__main__":
    main()
