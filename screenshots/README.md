# Screenshots for Documentation

This folder contains screenshots used in the project documentation and final report.

## Required Screenshots

### 1. CI/CD Pipeline
- **File:** `github-actions-pipeline.png`
- **Description:** GitHub Actions workflow showing all 4 jobs passing
- **How to capture:** Go to https://github.com/CHANDAN727/mlops/actions

### 2. MLflow Experiments
- **File:** `mlflow-experiments.png`
- **Description:** MLflow UI showing comparison of Logistic Regression vs Random Forest
- **How to capture:** Run `mlflow ui --backend-store-uri mlruns --port 5000` and open http://localhost:5000

### 3. API Documentation
- **File:** `api-swagger-docs.png`
- **Description:** Swagger UI with /predict endpoint
- **How to capture:** Visit http://35.238.24.35/docs or http://localhost:8080/docs

### 4. Grafana Dashboard
- **File:** `grafana-dashboard.png`
- **Description:** Real-time monitoring dashboard with metrics
- **How to capture:** Visit http://35.238.24.35/grafana/ (login: admin/mlops2024)

### 5. Prometheus Metrics
- **File:** `prometheus-targets.png`
- **Description:** Prometheus showing API target as UP
- **How to capture:** Visit http://35.238.24.35/prometheus/targets

### 6. Production Dashboard
- **File:** `production-dashboard.png`
- **Description:** Landing page with service navigation
- **How to capture:** Visit http://35.238.24.35

### 7. Docker Compose Stack
- **File:** `docker-compose-ps.png`
- **Description:** Terminal output showing all containers running
- **How to capture:** Run `docker-compose ps` on the VM

### 8. Prediction Example
- **File:** `prediction-example.png`
- **Description:** Successful POST /predict request with response
- **How to capture:** Use Swagger UI or Postman to make a prediction request

## Instructions for Screenshots

1. **GitHub Actions:**
   ```bash
   # Go to Actions tab in GitHub
   https://github.com/CHANDAN727/mlops/actions
   # Click on latest workflow run
   # Take screenshot showing all jobs green
   ```

2. **MLflow:**
   ```bash
   cd /Users/cpanda2/Documents/mlopsi
   mlflow ui --backend-store-uri mlruns --port 5000
   # Open http://localhost:5000
   # Click "Compare" to see both runs side-by-side
   ```

3. **API & Services:**
   - Use browser for web interfaces (Swagger, Grafana, Prometheus, Dashboard)
   - Use screenshot tool (Cmd+Shift+4 on macOS)
   - Save with descriptive filenames

4. **Terminal Outputs:**
   - Use terminal screenshot for docker-compose ps, test results, etc.

## Screenshot Quality Guidelines

- **Resolution:** At least 1920x1080
- **Format:** PNG (preferred) or JPG
- **Content:** Ensure all relevant information is visible
- **Annotations:** Use arrows or highlights if needed
- **File Size:** Keep under 2MB per image

## Naming Convention

Use descriptive names matching the documentation:
- `github-actions-pipeline.png`
- `mlflow-experiments.png`
- `api-swagger-docs.png`
- `grafana-dashboard.png`
- `prometheus-targets.png`
- `production-dashboard.png`
- `docker-compose-ps.png`
- `prediction-example.png`

Additional screenshots can be added as needed for the final report.
