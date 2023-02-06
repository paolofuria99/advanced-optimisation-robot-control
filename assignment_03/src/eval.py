import tensorflow as tf

from orc.assignment_03.src.environment.pendulum import SinglePendulum

if __name__ == "__main__":
    model_path = "models/"
    model_name = "single"

    model = tf.keras.models.load_model(model_path + model_name)

    num_controls = 64
    environment = SinglePendulum(num_controls=num_controls)

    environment.render_greedy_policy(model)
