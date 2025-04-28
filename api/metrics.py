"""
Prometheus metrics for the FastAPI application.
"""
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

# Define metrics
REQUESTS = Counter(
    'http_requests_total', 
    'Total HTTP requests', 
    ['method', 'endpoint', 'status_code']
)

REQUEST_TIME = Histogram(
    'http_request_duration_seconds', 
    'HTTP request duration in seconds',
    ['method', 'endpoint', 'status_code']
)

PREDICTIONS = Counter(
    'model_predictions_total', 
    'Total number of model predictions'
)

PREDICTION_VALUE = Histogram(
    'model_prediction_rul', 
    'Distribution of predicted RUL values',
    buckets=[0, 25, 50, 75, 100, 150, 200, 250, 300, float("inf")]
)

PREDICTION_ERROR = Counter(
    'model_prediction_errors_total', 
    'Total number of model prediction errors'
)

MODEL_VERSION = Gauge(
    'model_version',
    'Currently deployed model version',
    ['model_name']
)

# Set initial model version
def set_model_version(model_name, version):
    """Set the current model version in the metrics"""
    MODEL_VERSION.labels(model_name=model_name).set(float(version))

# Decorator to track endpoint metrics
def track_requests():
    """Decorator to track request metrics for FastAPI endpoints"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            method = kwargs.get('request', args[0] if args else None).__class__.__name__
            endpoint = func.__name__
            start_time = time.time()
            
            try:
                response = await func(*args, **kwargs)
                # Check if response is a dict or has status_code attribute
                if isinstance(response, dict):
                    status_code = 200  # Assume success for dict responses
                else:
                    status_code = response.status_code
                REQUESTS.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
                REQUEST_TIME.labels(method=method, endpoint=endpoint, status_code=status_code).observe(
                    time.time() - start_time
                )
                return response
            except Exception as e:
                status_code = 500
                REQUESTS.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
                REQUEST_TIME.labels(method=method, endpoint=endpoint, status_code=status_code).observe(
                    time.time() - start_time
                )
                raise e
                
        return wrapper
    return decorator
