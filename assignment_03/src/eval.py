import tensorflow as tf

from orc.assignment_03.src.environment.pendulum import SinglePendulum


if __name__ == "__main__":
    model_path = "models/"

    model = tf.keras.models.load_model(model_path)

    num_controls = 11
    environment = SinglePendulum(num_controls=num_controls, display=True)

    environment.render_greedy_policy(model)

