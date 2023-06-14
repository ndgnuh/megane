from functools import wraps


def with_batch_mode(f):
    """
    Decorator that enables batch mode for a function.

    Args:
        f (function): The function to be decorated.

    Returns:
        function: The wrapped function that supports batch mode.

    Example:
        @with_batch_mode
        def process_data(data, param1, param2, batch=False):
            # Process data

        # Usage without batch mode
        result = process_data(data, param1, param2)

        # Usage with batch mode
        results = process_data(data_batch, param1, param2, batch=True)
    """

    @wraps(f)
    def wrapped(x, *a, batch=False, **k):
        if batch:
            return [f(x_i, *a, **k) for x_i in x]
        return f(x, *a, **k)

    return wrapped
