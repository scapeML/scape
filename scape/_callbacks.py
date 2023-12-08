from tensorflow import keras
import keras.backend as K


class StepwiseScheduler(keras.callbacks.Callback):
    def __init__(self, penalty, dict_epoch_values=None, verbose=1):
        super(StepwiseScheduler, self).__init__()
        self.reg_penalty = penalty
        self.dict_epoch_values = dict_epoch_values
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        v = K.get_value(self.reg_penalty)
        vn = self.dict_epoch_values.get(epoch, v)
        if v != vn and self.verbose > 0:
            print(f"\tValue changed from {v} to {vn}")
            K.set_value(self.reg_penalty, vn)


def format_line(name, value, factor, percentage, max_name_length=20):
    # Truncate and add ellipsis if the name is longer than max_name_length characters
    display_name = (name[:max_name_length - 3] + '...') if len(name) > max_name_length else name
    # Format the line with the name, value, factor, and percentage using dynamic field width
    formatted_line = f"{display_name: <{max_name_length}} (err={value:.4f}): {factor:.4f}, {percentage:.2f}%"
    return formatted_line


class MonitorCallback(keras.callbacks.Callback):
    def __init__(self, monitor="val_decoder_mrrmse", baseline_errors=None):
        super(MonitorCallback, self).__init__()
        self.monitor = monitor
        self.best = None
        self.baseline_errors = baseline_errors

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        lr = float(K.get_value(self.model.optimizer.learning_rate))

        if self.best is None or current < self.best:
            # Get the train error:
            print(
                f"Epoch {(epoch+1):>4}: {self.monitor} "
                + (f"{self.best:.4f}" if self.best is not None else "None")
                + f" -> {current:.4f}, train loss: {logs['loss']:.4f} (lr={lr:.5f})"
            )
            if self.baseline_errors is not None:
                if not isinstance(self.baseline_errors, dict):
                    if isinstance(self.baseline_errors, list):
                        self.baseline_errors = {
                            f"baseline_{i}": b
                            for i, b in enumerate(self.baseline_errors)
                        }
                    else:
                        raise ValueError(
                            f"baseline_errors must be a dict or list, got {type(self.baseline_errors)}"
                        )
                for b_name, baseline_error in self.baseline_errors.items():
                    improv_factor = baseline_error / current
                    improv_percent = 100 * (1 - (current / baseline_error))
                    print("\t > Improvement factor/percentage " + format_line(b_name, baseline_error, improv_factor, improv_percent))
            self.best = current
