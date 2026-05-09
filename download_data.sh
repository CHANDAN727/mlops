#!/bin/bash
# Heart Disease Dataset Download Script
# UCI Machine Learning Repository - Cleveland Heart Disease Dataset

echo "=========================================="
echo "Heart Disease Dataset Download Script"
echo "=========================================="
echo ""

# Create data directory if it doesn't exist
mkdir -p data

echo "Dataset Information:"
echo "- Source: UCI Machine Learning Repository"
echo "- Name: Heart Disease (Cleveland)"
echo "- Samples: 303 patients"
echo "- Features: 13 clinical features + target"
echo "- URL: https://archive.ics.uci.edu/ml/datasets/Heart+Disease"
echo ""

# Dataset is already included in the repository
if [ -f "data/processed.cleveland.data" ]; then
    echo "✅ Dataset already exists: data/processed.cleveland.data"
    echo ""
    echo "Dataset Preview:"
    head -n 5 data/processed.cleveland.data
    echo ""
    echo "Total records: $(wc -l < data/processed.cleveland.data)"
else
    echo "❌ Dataset not found!"
    echo ""
    echo "Downloading from UCI repository..."
    
    # Download the dataset
    curl -o data/processed.cleveland.data \
        https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
    
    if [ -f "data/processed.cleveland.data" ]; then
        echo "✅ Download complete!"
        echo "Saved to: data/processed.cleveland.data"
    else
        echo "❌ Download failed!"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Dataset Features:"
echo "=========================================="
echo "1.  age       - Age in years"
echo "2.  sex       - Gender (1=male, 0=female)"
echo "3.  cp        - Chest pain type (1-4)"
echo "4.  trestbps  - Resting blood pressure (mm Hg)"
echo "5.  chol      - Serum cholesterol (mg/dl)"
echo "6.  fbs       - Fasting blood sugar > 120 mg/dl"
echo "7.  restecg   - Resting ECG results (0-2)"
echo "8.  thalach   - Maximum heart rate achieved"
echo "9.  exang     - Exercise induced angina"
echo "10. oldpeak   - ST depression"
echo "11. slope     - Slope of peak exercise ST"
echo "12. ca        - Number of major vessels (0-3)"
echo "13. thal      - Thalassemia (3, 6, 7)"
echo "14. num       - Target (0 = no disease, 1-4 = disease)"
echo ""
echo "✅ Dataset is ready for use!"
echo "=========================================="
