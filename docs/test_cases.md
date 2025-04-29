## Turbofan RUL Test Cases

### TC1: Root Endpoint
- **Steps**: GET /
- **Expected**: 200 OK with welcome message
- **Actual**: [Passed in test_root()]

### TC2: Prediction Endpoint
- **Steps**: POST /predict with valid data
- **Expected**: 200 OK with RUL prediction
- **Actual**: [Passed in test_predict_endpoint()]

### TC3: Metrics Endpoint 
- **Steps**: GET /metrics
- **Expected**: 200 OK with Prometheus metrics
- **Actual**: [Passed in test_metrics_endpoint()]
