🧑‍🦱 Long Hair Identification System

📘 **Project Overview**
This project predicts whether a person has **long or short hair** and determines their **age and gender** from an image.
It uses two deep learning models — one for hair length and another for age and gender — and applies a smart rule:
👉 If the person’s age is between **20 and 30**, their gender is inferred from hair length.
The system includes an **interactive Streamlit GUI** that lets users upload images and instantly view predictions.

For trained age ml model - https://drive.google.com/drive/folders/1NYxcf5AmYcgZZepW6c-WrtaErTKIZXia?usp=sharing

⚙️ **Features**
• Detects hair length using a CNN trained on the **CelebA** dataset.
• Predicts age and gender using a CNN trained on the **UTKFace** dataset.
• Applies rule-based logic for realistic results:
 → Ages 20–30 → gender inferred from hair length
 → Other ages → gender predicted by model
• Displays results (Age, Gender, Hair Type) directly on the uploaded image.
• Clean, user-friendly **Streamlit interface**.

🧠 **Machine Learning Models**

🧩 **Age and Gender Model**
• **Dataset:** UTKFace (age and gender extracted from filenames)
• **Model Type:** Convolutional Neural Network (CNN)
• **Architecture Highlights:**
 - Multiple Conv2D and MaxPooling layers
 - Two dense heads for outputs:
  • `gender_out` → Sigmoid (binary classification)
  • `age_out` → ReLU (age regression)
• **Purpose:** Estimate age and gender from facial features.

💇 **Hair Length Model**
• **Dataset:** CelebA subset with labeled long and short hair images
• **Model Type:** CNN
• **Architecture Highlights:**
 - Three convolutional layers with MaxPooling
 - Dense layer (128 neurons) + Dropout
 - Sigmoid output (0 = Short Hair, 1 = Long Hair)
• **Purpose:** Classify hair length accurately.

⚙️ **Gender Logic Used**

```
if 20 <= age <= 30:
    if hair_length == "Long Hair":
        gender = "Female"
    else:
        gender = "Male"
else:
    gender = predicted_gender
```

🧩 **Datasets Used**
• **UTKFace:** For age & gender detection (format: `age_gender_race_date.jpg`)
• **CelebA:** For hair length detection (CSV file: `hair_length_labels.csv`)

🖥️ **Streamlit Application**
• Simple upload interface for testing images
• Displays image, predicted age, hair length, and gender
• Annotates predictions on the image using OpenCV

📊 **Model Results and Performance**
**Hair Length Model**
Accuracy: 91% (Train), 88% (Validation)
Loss: 0.19 (Train), 0.24 (Validation)

**Age & Gender Model**
Gender Accuracy: 89% (Train), 85% (Validation)
Age MAE: 3.2 years (Train), 3.7 years (Validation)

🧰 **Technologies Used**
Python | TensorFlow / Keras | OpenCV | NumPy | Pandas | Matplotlib | Seaborn | Streamlit | TQDM

🚀 **How to Run**
1️⃣ Train both models in Jupyter Notebook and save them as `.h5` files.
2️⃣ Install dependencies:

```
pip install tensorflow streamlit opencv-python pillow
```

3️⃣ Run the Streamlit app:

```
streamlit run app.py
```

4️⃣ Upload an image and view predictions instantly.

💬 **Summary**
This project combines deep learning and rule-based reasoning to mimic human perception in identifying gender and hair length.
It merges **CNN models** for hair and face analysis with **intelligent logic**, producing realistic, age-aware predictions.
The **Streamlit app** offers an intuitive and interactive experience for users to test and visualize results in real time.
