# 🍅 Tomato Guard AI

> **AI-Powered Tomato Leaf Disease Detection System**

Tomato Guard AI is a web-based artificial intelligence system designed specifically for **tomato plants**. It analyzes an uploaded tomato leaf image using a **MobileNetV2 deep-learning model** and provides a preliminary prediction of the most likely tomato leaf condition along with a confidence score and top predictions.

The system is intended to assist **tomato growers and harvesters** by helping them identify leaves that may require closer inspection.

---

## 📌 About the Project

Plant diseases can affect tomato crop quality, productivity, and overall farm management. Early identification of visible symptoms can help growers take timely action.

**Tomato Guard AI** provides a simple image-based interface where a user can:

1. Upload a tomato leaf image.
2. Preview the selected image.
3. Send the image to the AI backend.
4. Analyze the image using a trained MobileNetV2 model.
5. Receive the predicted tomato leaf condition.
6. View the model confidence.
7. View the top 3 predictions.
8. Get tomato-harvester-oriented preliminary guidance.

### Important Scope

This model is trained **only on tomato leaf classes**.

It is **not designed to diagnose potato, pepper, or other plant leaves**.

The prediction should be treated as a **preliminary AI indication**, not as a definitive agricultural diagnosis.

---

# 🎯 Project Objectives

- Develop an AI-based tomato leaf condition detection system.
- Train a lightweight MobileNetV2-based classification model.
- Restrict the trained classification problem to tomato leaves.
- Provide an easy-to-use web interface.
- Connect the frontend to the trained AI model through a Flask API.
- Display prediction confidence and top alternative predictions.
- Build an application that can assist tomato growers and harvesters during preliminary crop inspection.

---

# 🧠 Machine Learning Model

## Model Architecture

The project uses **MobileNetV2** with ImageNet pretrained weights as the feature extraction backbone.

The model architecture is:

```text
Input Image
   ↓
224 × 224 × 3
   ↓
MobileNetV2
   ↓
Global Average Pooling
   ↓
Dropout (0.3)
   ↓
Dense Layer (128 neurons, ReLU)
   ↓
Dropout (0.2)
   ↓
Dense Layer (10 neurons, Softmax)
   ↓
Tomato Class Prediction
```

### Model characteristics

- Architecture: **MobileNetV2**
- Transfer learning: **Yes**
- Initial base-model training: **Frozen**
- Input size: **224 × 224**
- Output classes: **10**
- Final activation: **Softmax**
- Loss: **Sparse Categorical Crossentropy**
- Optimizer: **Adam**
- Maximum training epochs: **15**
- Early stopping: **Enabled**
- Dropout: **0.3 and 0.2**
- Final trained model: `plant_disease_model.h5`

---

# 🍅 Supported Tomato Classes

The trained model contains the following 10 classes:

| # | Class | Type |
|---|---|---|
| 1 | Tomato_Bacterial_spot | Disease |
| 2 | Tomato_Early_blight | Disease |
| 3 | Tomato_Late_blight | Disease |
| 4 | Tomato_Leaf_Mold | Disease |
| 5 | Tomato_Septoria_leaf_spot | Disease |
| 6 | Tomato_Spider_mites_Two_spotted_spider_mite | Pest condition |
| 7 | Tomato__Target_Spot | Disease |
| 8 | Tomato__Tomato_YellowLeaf__Curl_Virus | Viral disease |
| 9 | Tomato__Tomato_mosaic_virus | Viral disease |
| 10 | Tomato_healthy | Healthy class |

Therefore, the system contains **9 disease/pest-condition classes + 1 healthy class**.

---

# 📊 Model Performance

The current trained model was evaluated on the held-out test dataset.

### Test Performance

**Test Accuracy: 90.99%**

Test set:

- **1,610 images**
- **10 classes**

The evaluation also generated:

- Precision
- Recall
- F1-score
- Confusion matrix

### Classification Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Tomato Bacterial Spot | 91.34% | 98.60% | 94.83% |
| Tomato Early Blight | 94.03% | 63.00% | 75.45% |
| Tomato Late Blight | 91.50% | 95.31% | 93.37% |
| Tomato Leaf Mold | 86.32% | 85.42% | 85.86% |
| Tomato Septoria Leaf Spot | 90.42% | 84.83% | 87.54% |
| Tomato Spider Mites | 84.83% | 89.35% | 87.03% |
| Tomato Target Spot | 80.00% | 85.11% | 82.47% |
| Tomato Yellow Leaf Curl Virus | 96.89% | 96.89% | 96.89% |
| Tomato Mosaic Virus | 88.10% | 97.37% | 92.50% |
| Tomato Healthy | 98.10% | 96.88% | 97.48% |

**Macro F1-score:** 89.34%  
**Weighted F1-score:** 90.83%

> Note: Performance depends on image quality, dataset characteristics, and similarity between real-world images and training data.

---

# 🛠️ Technology Stack

## Machine Learning

- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Pillow
- Scikit-learn
- Matplotlib
- Seaborn

## Frontend

- HTML5
- CSS3
- JavaScript
- Browser File API
- JavaScript `fetch()` API

## Backend

- Python
- Flask
- Flask-CORS
- REST API

## Development Tools

- VS Code
- Git
- GitHub
- Git Bash
- Python virtual environment / local Python environment

---

# 🏗️ System Architecture

```text
                     🍅 TOMATO GUARD AI
                             |
                 +-----------+-----------+
                 |                       |
                 ▼                       ▼
          FRONTEND                  BACKEND
       HTML + CSS + JS              Flask API
                 |                       |
                 |   POST /predict       |
                 +----------->-----------+
                                         |
                                         ▼
                                  Image Processing
                                         |
                                         ▼
                                  MobileNetV2
                                         |
                                         ▼
                                  10-Class Model
                                         |
                                         ▼
                              Prediction + Confidence
                                         |
                 <-----------------------+
                 |
                 ▼
           JavaScript UI
                 |
                 ▼
        🍅 Result Display
```

---

# ⚙️ How the System Works

## 1. User uploads a leaf image

The user selects a JPG, JPEG, or PNG image from the browser.

## 2. Frontend preview

JavaScript displays the selected image before analysis.

## 3. Image submission

When the user clicks:

```text
🔍 Analyze Tomato Leaf
```

JavaScript creates a `FormData` object and sends the image to:

```text
POST /predict
```

## 4. Flask receives the image

The Flask backend receives the uploaded image and prepares it for the trained model.

## 5. Model inference

The image is resized to:

```text
224 × 224
```

and passed to the trained MobileNetV2-based classification model.

## 6. Prediction

The model generates probabilities for all 10 tomato classes.

The system selects the class with the highest probability.

## 7. Result generation

The backend returns JSON containing:

- Plant
- Prediction
- Confidence
- Healthy status
- Low-confidence status
- Top 3 predictions

Example:

```json
{
  "success": true,
  "plant": "Tomato",
  "prediction": "Late Blight",
  "confidence": 94.32,
  "healthy": false,
  "low_confidence": false,
  "top_predictions": [
    {
      "name": "Late Blight",
      "confidence": 94.32
    },
    {
      "name": "Early Blight",
      "confidence": 3.21
    },
    {
      "name": "Target Spot",
      "confidence": 1.42
    }
  ]
}
```

## 8. Frontend displays the result

JavaScript receives the JSON response and dynamically updates the result section.

---

# 📁 Project Structure

```text
plant-disease-detection/
│
├── data/
│   ├── raw/
│   │   └── PlantVillage/
│   │       ├── Tomato_Bacterial_spot/
│   │       ├── Tomato_Early_blight/
│   │       ├── Tomato_Late_blight/
│   │       ├── Tomato_Leaf_Mold/
│   │       ├── Tomato_Septoria_leaf_spot/
│   │       ├── Tomato_Spider_mites_Two_spotted_spider_mite/
│   │       ├── Tomato__Target_Spot/
│   │       ├── Tomato__Tomato_YellowLeaf__Curl_Virus/
│   │       ├── Tomato__Tomato_mosaic_virus/
│   │       └── Tomato_healthy/
│   │
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
│
├── src/
│   ├── model.py
│   ├── data_prep.py
│   ├── augment.py
│   ├── train.py
│   ├── evaluate.py
│   ├── labels.py
│   ├── test_img.py
│   ├── labels.txt
│   └── plant_disease_model.h5
│
├── backend/
│   ├── app.py
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── script.js
│   └── assets/
│
├── README.md
└── .gitignore
```

---

# 🧪 Dataset Preparation

The dataset preparation pipeline performs:

```text
PlantVillage Dataset
       ↓
Select Tomato Classes
       ↓
Resize Images
       ↓
Train / Validation / Test Split
       ↓
Data Augmentation
       ↓
Model Training
```

The current split uses approximately:

```text
80% → Training
10% → Validation
10% → Testing
```

Data augmentation is applied to the training images only.

The augmentation pipeline includes:

- Rotation
- Horizontal flip
- Vertical flip
- Brightness adjustment
- Contrast adjustment

Validation and test datasets are kept separate from augmentation.

---

# 🚀 Running the Project Locally

## Prerequisites

Install:

- Python 3.10
- Git
- A modern web browser

Verify Python:

```bash
python --version
```

---

# 1️⃣ Install Backend Dependencies

From the project root:

```bash
pip install -r backend/requirements.txt
```

The backend requirements are:

```text
Flask
flask-cors
tensorflow
numpy
Pillow
```

---

# 2️⃣ Start the Flask Backend

From the project root:

```bash
python backend/app.py
```

The backend will run at:

```text
http://127.0.0.1:5000
```

Expected output:

```text
🍅 TOMATO GUARD AI
AI-Powered Tomato Disease Detection

🍅 Plant scope: TOMATO ONLY
🧠 Model: MobileNetV2
📊 Classes: 10

🌐 Backend:
http://127.0.0.1:5000
```

---

# 3️⃣ Test Backend Health

Open:

```text
http://127.0.0.1:5000/health
```

Expected response:

```json
{
  "success": true,
  "application": "Tomato Guard AI",
  "status": "healthy",
  "plant": "Tomato",
  "model": "MobileNetV2",
  "classes": 10
}
```

---

# 4️⃣ Start the Frontend

Open a second terminal.

```bash
cd frontend
```

Run:

```bash
python -m http.server 5500
```

The frontend will be available at:

```text
http://127.0.0.1:5500
```

Open that address in your browser.

---

# 5️⃣ Use the Application

1. Open the Tomato Guard AI website.
2. Read the tomato-only scope notice.
3. Click **Browse Image** or drag and drop an image.
4. Preview the selected leaf.
5. Click **Analyze Tomato Leaf**.
6. Wait for the AI response.
7. View:
   - Predicted condition
   - Confidence score
   - Top 3 predictions
   - Healthy/disease status
   - Preliminary harvester guidance

---

# 🔌 Backend API Integration

## Health Endpoint

```text
GET /health
```

Purpose:

- Verify backend availability.
- Verify model availability.
- Confirm tomato scope.
- Confirm 10 classes.

---

## Prediction Endpoint

```text
POST /predict
```

### Request

Multipart form data:

```text
image = <leaf image>
```

### Example JavaScript

```javascript
const formData = new FormData();

formData.append("image", selectedFile);

const response = await fetch(
    "http://127.0.0.1:5000/predict",
    {
        method: "POST",
        body: formData
    }
);

const result = await response.json();

console.log(result);
```

---

# 🖥️ Frontend Execution

The frontend consists of three primary files:

### `index.html`

Provides:

- Navigation
- Hero section
- About section
- Features
- How-it-works section
- Tomato scope warning
- Upload interface
- Result section
- Supported conditions
- Footer

### `style.css`

Provides:

- Responsive layout
- Cards
- Buttons
- Upload area
- Confidence bar
- Result styling
- Mobile responsiveness
- Animations

### `script.js`

Handles:

- Image selection
- Drag and drop
- Image preview
- File removal
- API requests
- Loading state
- Prediction result
- Confidence display
- Top 3 predictions
- Result rendering

---

# 🐍 Backend Execution

The Flask backend performs the AI inference workflow.

```text
Request
   ↓
Validate uploaded image
   ↓
Open image
   ↓
Convert to RGB
   ↓
Resize to 224 × 224
   ↓
Prepare NumPy input
   ↓
MobileNetV2 model
   ↓
Probability scores
   ↓
Highest probability class
   ↓
Top 3 predictions
   ↓
JSON response
```

The backend loads the trained model once when the application starts:

```python
MODEL = tf.keras.models.load_model(MODEL_PATH)
```

This avoids loading the model for every individual prediction request.

---

# ⭐ Key Features

## 🍅 1. Tomato-Specific AI

The model is trained only on tomato leaf classes.

This makes the application's intended scope clear:

```text
Tomato → Supported
Potato → Not supported
Pepper → Not supported
Other plants → Not supported
```

---

## 🧠 2. MobileNetV2 Transfer Learning

MobileNetV2 provides a relatively lightweight architecture suitable for image classification applications.

---

## 📸 3. Simple Image Upload

Users can:

- Browse for an image.
- Drag and drop an image.
- Preview the image before analysis.
- Remove and select another image.

---

## 📊 4. Confidence Score

The application displays the model's confidence for the selected prediction.

---

## 🔎 5. Top 3 Predictions

Instead of showing only one class, the system displays the three highest-probability model classes.

---

## 🌾 6. Tomato Harvester-Oriented Interface

The interface is designed around a practical use case:

> helping tomato growers and harvesters identify leaves that may need closer inspection.

---

## ⚠️ 7. Low-Confidence Handling

Predictions below the configured confidence threshold can be presented as low-confidence results, encouraging the user to provide a clearer image.

---

## 📱 8. Responsive Web Interface

The frontend is designed to work across desktop and smaller screens.

---

# 💡 Why Tomato Guard AI Is Different

Many general image-classification demonstrations simply provide:

```text
Upload → Prediction
```

Tomato Guard AI focuses on a more specific agricultural use case.

### 1. Narrow application scope

The system is specifically designed around **tomato leaves**, rather than claiming to diagnose every plant.

### 2. End-to-end integration

It combines:

```text
Dataset
   ↓
Preprocessing
   ↓
Augmentation
   ↓
Transfer Learning
   ↓
Model Evaluation
   ↓
Flask API
   ↓
JavaScript Frontend
```

### 3. Practical result presentation

The application provides:

- Prediction
- Confidence
- Top 3 predictions
- Healthy status
- Low-confidence indication
- Harvester-oriented guidance

### 4. Clear limitations

The application explicitly tells users that the model is for tomato leaves only.

### 5. Lightweight model architecture

MobileNetV2 was selected as a relatively lightweight transfer-learning architecture, making it suitable for a practical web inference application.

---

# 🧪 Testing Instructions

## Test 1 — Backend Health

Run:

```bash
python backend/app.py
```

Open:

```text
http://127.0.0.1:5000/health
```

Expected:

```text
success = true
status = healthy
classes = 10
plant = Tomato
```

---

## Test 2 — Tomato Image

Upload a known tomato image from the test dataset.

Expected:

```text
Prediction
+
Confidence
+
Top 3 Predictions
```

---

## Test 3 — Healthy Tomato

Upload an image from:

```text
Tomato_healthy
```

The expected top prediction should be:

```text
Healthy
```

The actual output may vary depending on the image.

---

## Test 4 — Different Tomato Conditions

Test images from:

```text
Tomato_Bacterial_spot
Tomato_Early_blight
Tomato_Late_blight
Tomato_Leaf_Mold
Tomato_Septoria_leaf_spot
Tomato_Spider_mites_Two_spotted_spider_mite
Tomato__Target_Spot
Tomato__Tomato_YellowLeaf__Curl_Virus
Tomato__Tomato_mosaic_virus
Tomato_healthy
```

---

## Test 5 — Invalid File

Try uploading a non-image file.

Expected behavior:

```text
Please upload a valid image.
```

---

## Test 6 — Backend Offline

Stop Flask and try analyzing an image.

The frontend should report that the backend is unavailable.

---

## Test 7 — Non-Tomato Image

A potato, pepper, or other plant image should **not be interpreted as a guaranteed tomato diagnosis**.

Important:

> The current 10-class classifier itself does not contain a separate non-tomato rejection class. Therefore, a non-tomato image can still receive one of the 10 tomato-class predictions.

A dedicated tomato/non-tomato verification model is a planned future improvement.

---

# 🔐 Limitations

The current system has several important limitations:

1. It is trained for tomato leaf classes only.
2. It does not contain a dedicated non-tomato rejection model.
3. Image quality can affect predictions.
4. Real field images may differ from controlled dataset images.
5. The output is a preliminary AI indication, not a confirmed agricultural diagnosis.
6. A high confidence score does not guarantee real-world correctness.
7. Agricultural decisions should consider field conditions and expert/local guidance.

---

# 🚀 Future Enhancements

## 1. 🍅 Tomato vs Non-Tomato Verification

Add a separate classifier:

```text
Uploaded Image
      ↓
Tomato / Non-Tomato
      ↓
   Tomato?
   /     \
 YES      NO
  ↓        ↓
Disease   Reject
Model     Image
```

This would reduce the risk of interpreting pepper or potato leaves using the tomato classifier.

---

## 2. 📱 Mobile Camera Integration

Allow users to capture a tomato leaf directly using a smartphone camera.

---

## 3. 🌦️ Weather Integration

Combine disease predictions with:

- Temperature
- Humidity
- Rainfall
- Weather forecasts

to provide more contextual crop information.

---

## 4. 🧠 Explainable AI

Add Grad-CAM or another explainability technique to highlight image regions that influenced the prediction.

Example:

```text
Original Leaf
     ↓
AI Analysis
     ↓
Highlighted Symptom Regions
```

---

## 5. 🌐 Multilingual Support

Support languages useful to local tomato growers and agricultural communities.

Potential examples include:

- English
- Telugu
- Hindi
- Tamil
- Kannada

---

## 6. 📊 Prediction History

Allow users to save previous analyses and compare observations over time.

---

## 7. 🌾 Field-Level Monitoring

Extend the system from single-leaf analysis to:

```text
Multiple Leaf Images
        ↓
Crop-level Analysis
        ↓
Disease Distribution
        ↓
Field Monitoring Dashboard
```

---

## 8. 🔬 Improved Model Training

Future versions can explore:

- Fine-tuning more MobileNetV2 layers
- EfficientNet architectures
- Data balancing
- More diverse field images
- Higher-quality datasets
- Cross-dataset validation

---

## 9. ☁️ Cloud Deployment

Deploy:

```text
Frontend → Web Hosting
Backend → Cloud Server
Model → Cloud Inference
```

to make the system accessible through a public URL.

---

# 🌐 Application Access

### Local Frontend

```text
http://127.0.0.1:5500
```

### Local Backend

```text
http://127.0.0.1:5000
```

### Backend Health Check

```text
http://127.0.0.1:5000/health
```

### Live Application

```text
LIVE APP: To be added after deployment
```

> Replace the live application placeholder with the final deployed URL after deployment.

---

# 🔗 Project Integration

The system integrates four major layers:

```text
             DATA LAYER
                  ↓
        Tomato Leaf Dataset
                  ↓
             ML LAYER
                  ↓
       MobileNetV2 Classifier
                  ↓
            API LAYER
                  ↓
            Flask REST API
                  ↓
         PRESENTATION LAYER
                  ↓
         HTML + CSS + JavaScript
```

This separation makes the project easier to maintain and deploy.

---

# 📦 Deployment Architecture

The planned production architecture is:

```text
                    INTERNET
                       |
          +------------+------------+
          |                         |
          ▼                         ▼
      FRONTEND                   BACKEND
       Hosting                  Cloud Server
          |                         |
    HTML/CSS/JS                Python Flask
          |                         |
          +------ HTTPS API --------+
                                    |
                                    ▼
                             TensorFlow/Keras
                                    |
                                    ▼
                              MobileNetV2
                                    |
                                    ▼
                         plant_disease_model.h5
```

The frontend's local API URL:

```javascript
const API_URL = "http://127.0.0.1:5000";
```

must be replaced with the deployed backend URL during production deployment.

---

# 🛠️ Development Workflow

```text
1. Collect dataset
        ↓
2. Prepare dataset
        ↓
3. Split train / validation / test
        ↓
4. Augment training data
        ↓
5. Build MobileNetV2 model
        ↓
6. Train model
        ↓
7. Evaluate model
        ↓
8. Save .h5 model
        ↓
9. Create Flask API
        ↓
10. Build HTML/CSS/JS frontend
        ↓
11. Integrate frontend + backend
        ↓
12. Test application
        ↓
13. Deploy
```

---

# 📚 Main Project Files

| File | Purpose |
|---|---|
| `data_prep.py` | Dataset preprocessing and splitting |
| `augment.py` | Training-image augmentation |
| `model.py` | MobileNetV2 model architecture |
| `train.py` | Model training |
| `evaluate.py` | Model evaluation |
| `labels.py` | Generates class labels |
| `test_img.py` | Standalone image prediction testing |
| `labels.txt` | Model class labels |
| `plant_disease_model.h5` | Trained model |
| `backend/app.py` | Flask inference API |
| `frontend/index.html` | Web interface |
| `frontend/css/style.css` | UI styling |
| `frontend/js/script.js` | Frontend interaction and API integration |

---

# 🤝 Integration Summary

The application connects the components as follows:

```text
User
 ↓
HTML Interface
 ↓
JavaScript
 ↓
FormData
 ↓
Flask /predict
 ↓
Image Processing
 ↓
MobileNetV2
 ↓
Prediction Probabilities
 ↓
Top Class + Confidence
 ↓
JSON Response
 ↓
JavaScript
 ↓
Result UI
```

---

# ⚠️ Responsible Use

Tomato Guard AI is intended as a **preliminary screening and educational assistance system**.

The model's prediction should not be considered a definitive diagnosis or a replacement for qualified agricultural advice.

Users should consider:

- Plant condition
- Field environment
- Weather
- Crop history
- Visible symptoms
- Local agricultural recommendations

before taking important crop-management decisions.

---

# 👨‍💻 Project Status

**Current status:** Functional prototype / deployment-ready development version

### Completed

- [x] Tomato-only dataset preparation
- [x] 10-class classification
- [x] Data augmentation
- [x] MobileNetV2 training
- [x] Model evaluation
- [x] 90.99% test accuracy
- [x] `.h5` model export
- [x] Flask backend
- [x] REST prediction API
- [x] HTML frontend
- [x] CSS responsive interface
- [x] JavaScript image upload
- [x] Frontend-backend integration
- [x] Prediction display
- [x] Confidence display
- [x] Top 3 predictions
- [x] Tomato-only scope messaging

### Planned

- [ ] Tomato/non-tomato verification
- [ ] Live deployment
- [ ] Camera capture
- [ ] Explainable AI
- [ ] Weather integration
- [ ] Multilingual support
- [ ] Prediction history
- [ ] Field-level monitoring

---

# 📜 License

This project is intended for educational, academic, and demonstration purposes.

Add an appropriate open-source license here if the project is released publicly.

---

# ❤️ Acknowledgement

This project combines deep learning, transfer learning, image classification, web development, and REST API integration to create a practical tomato leaf screening application.

**Tomato Guard AI — Turning a tomato leaf image into a preliminary AI insight. 🍅**
