"""
Prometheus metrics for the FastAPI application.
This module defines counters, histograms, and gauges for tracking HTTP requests
and model prediction metrics.
"""
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

# Define a counter for total HTTP requests, labeled by method, endpoint, and status code.
REQUESTS = Counter(
    'http_requests_total', 
    'Total HTTP requests', 
    ['method', 'endpoint', 'status_code']
)

# Define a histogram to measure the duration of HTTP requests.
REQUEST_TIME = Histogram(
    'http_request_duration_seconds', 
    'HTTP request duration in seconds',
    ['method', 'endpoint', 'status_code']
)

# Define a counter for total model predictions.
PREDICTIONS = Counter(
    'model_predictions_total', 
    'Total number of model predictions'
)

# Define a histogram for the distribution of predicted RUL values with custom buckets.
PREDICTION_VALUE = Histogram(
    'model_prediction_rul', 
    'Distribution of predicted RUL values',
    buckets=[0, 25, 50, 75, 100, 150, 200, 250, 300, float("inf")]
)

# Define a counter for model prediction errors.
PREDICTION_ERROR = Counter(
    'model_prediction_errors_total', 
    'Total number of model prediction errors'
)

# Define a gauge to track the current version of the deployed model.
MODEL_VERSION = Gauge(
    'model_version',
    'Currently deployed model version',
    ['model_name']
)

def set_model_version(model_name, version):
    """
    Set the current model version in the metrics.
    
    Parameters:
    model_name (str): Name of the model.
    version (number or str): The version to be set.
    """
    # Set the gauge value for the provided model name.
    MODEL_VERSION.labels(model_name=model_name).set(float(version))

def track_requests():
    """
    Decorator to track request metrics for FastAPI endpoints.
    Wraps the endpoint functions to measure request count and duration.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Derive the HTTP method from kwargs or positional args.
            # This assumes that the first argument or 'request' in kwargs provides the request object.
            method = kwargs.get('request', args[0] if args else None).__class__.__name__
            endpoint = func.__name__  # Use the function name as the endpoint identifier.
            start_time = time.time()  # Start timer before function call.
            
            try:
                response = await func(*args, **kwargs)
                # Determine status code based on response type.
                # If response is a dict, assume a successful HTTP 200.
                if isinstance(response, dict):
                    status_code = 200  
                else:
                    status_code = response.status_code
                # Increment the request counter.
                REQUESTS.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
                # Observe the request duration.
                REQUEST_TIME.labels(method=method, endpoint=endpoint, status_code=status_code).observe(
                    time.time() - start_time
                )
                return response
            except Exception as e:
                status_code = 500  # On exception, consider it as a server error.
                # Increment metrics for the error case.
                REQUESTS.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
                REQUEST_TIME.labels(method=method, endpoint=endpoint, status_code=status_code).observe(
                    time.time() - start_time
                )
                # Re-raise the exception after recording the metrics.
                raise e
                
        return wrapper
    return decorator
