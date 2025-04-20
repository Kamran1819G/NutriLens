# NutriScan Dashboard 🍎

## Overview

NutriScan Dashboard is a Streamlit application that helps you identify fruits from images and provides detailed nutritional information. By simply uploading an image of a fruit, NutriScan utilizes AI image recognition to identify the fruit and then fetches its nutritional data from the USDA FoodData Central database.  Furthermore, it leverages the power of the Google Gemini API to generate insightful summaries of the health benefits associated with the identified fruit.

This dashboard is designed to be a user-friendly tool for anyone interested in learning more about the fruits they eat, their nutritional content, and health advantages.

## Features

*   **Image-Based Fruit Identification:**  Accurately identifies a wide range of fruits from uploaded images using a Hugging Face pre-trained image classification model.
*   **Comprehensive Nutrition Facts:** Retrieves detailed nutritional information directly from the USDA FoodData Central API, including macronutrients, vitamins, and minerals.
*   **Interactive Visualizations:** Presents nutritional data through engaging Plotly charts, such as macronutrient bar charts and micronutrient radar charts, for easy understanding.
*   **AI-Powered Health Benefit Summaries:** Generates concise and informative summaries of health benefits using the Google Gemini API, based on the fruit's nutritional profile.
*   **Fruit Comparison:** Allows users to compare the nutritional content of the identified fruit with other common fruits like apple, banana, grapes, mango, and strawberry.
*   **User-Friendly Interface:** Built with Streamlit for an intuitive and easy-to-use web application experience.

## Technologies Used

*   **Python:** The primary programming language.
*   **Streamlit:** For creating the interactive web application.
*   **PIL (Pillow):** For image processing.
*   **Plotly:** For creating interactive charts and visualizations.
*   **Hugging Face Transformers:**  For loading and using the pre-trained image classification model (`dima806/fruit_100_types_image_detection`).
*   **PyTorch:**  The deep learning framework used by the Hugging Face model.
*   **Google Gemini API (Generative AI):** For generating health benefit summaries.
*   **USDA FoodData Central API:** For fetching nutritional data.
*   **requests:** For making HTTP requests to APIs.
*   **dotenv:** For managing environment variables securely.
*   **NumPy:** For numerical operations (implicitly used by PIL and other libraries).

## Setup Instructions

To run NutriScan Dashboard locally, follow these steps:

1.  **Prerequisites:**
    *   Python 3.7 or higher is required.
    *   Ensure you have `pip` installed.

2.  **Clone the Repository:**
    ```bash
    git clone [repository URL] # Replace with your repository URL if you have one
    cd NutriScan-Dashboard # Or the name of the cloned directory
    ```

3.  **Install Dependencies:**
    Navigate to the cloned repository directory in your terminal and run:
    ```bash
    pip install -r requirements.txt
    ```
    This command will install all the necessary Python libraries listed in the `requirements.txt` file.

4.  **Set up API Keys:**
    *   **USDA API Key:**
        *   Visit the [USDA FoodData Central API website](https://fdc.nal.usda.gov/api-key-signup.html) to sign up for an API key. It's typically free for a certain usage level.
        *   Once you receive your API key, create a `.env` file in the root directory of the project.
        *   Add the following line to your `.env` file, replacing `YOUR_USDA_API_KEY` with your actual API key:
            ```
            USDA_API_KEY=YOUR_USDA_API_KEY
            ```
    *   **Google Gemini API Key (Optional for Health Benefits):**
        *   To enable AI-powered health benefit summaries, you need a Google Gemini API key.  You can obtain one through [Google AI Studio](https://makersuite.google.com/app/apikey). Please refer to Google AI Studio documentation for details on setup and usage.
        *   Add the following line to your `.env` file, replacing `YOUR_GEMINI_API_KEY` with your actual Gemini API key:
            ```
            GEMINI_API_KEY=YOUR_GEMINI_API_KEY
            ```
        *   If you do not set up a Gemini API key, the health benefit summaries will be displayed as placeholders, and a warning will be shown in the application.

5.  **Run the Streamlit App:**
    In your terminal, within the project directory, run the Streamlit application:
    ```bash
    streamlit run your_script_name.py # Replace 'your_script_name.py' with the actual name of your Python script (e.g., app.py, NutriScan.py)
    ```
    Streamlit will launch the NutriScan Dashboard in your web browser (usually at `http://localhost:8501`).

## Usage Instructions

1.  **Upload an Image:**
    *   In the "📸 Upload Fruit Image" section, click on "Browse files" or drag and drop an image of a fruit. Ensure the image is clear and shows a single fruit for the best detection results. Supported image types are JPG, JPEG, and PNG.

2.  **Analyze Fruit:**
    *   After uploading the image, click the "🔍 Analyze Fruit" button.
    *   The application will process the image using the AI model to identify the fruit.
    *   If successful, you will see a success message indicating the identified fruit.

3.  **Explore Nutritional Information:**
    *   The dashboard will display tabs for different aspects of the fruit's information:
        *   **📊 Nutrition Facts:** Presents a table of basic nutrition information, vitamins, and minerals.
        *   **📈 Visualizations:** Shows interactive charts, including a macronutrient bar chart and a micronutrient radar chart, to visualize the nutritional profile.
        *   **💪 Health Benefits:** Provides an AI-generated summary of the health benefits of the fruit, powered by the Google Gemini API (if API key is configured).

4.  **Compare with Another Fruit:**
    *   In the "💪 Health Benefits" tab, you will find a section "🔄 Compare with another fruit".
    *   Select a fruit from the dropdown menu (apple, banana, grapes, mango, strawberry) to compare with the identified fruit.
    *   Click the "Compare" button to see a side-by-side comparison of key nutritional information.

## Example Images

For testing the application, you can use images of common fruits like:

*   Apples (Red Delicious, Granny Smith, etc.)
*   Bananas
*   Grapes (Red, Green, Black)
*   Mangoes
*   Strawberries
*   Pineapples
*   Kiwis
*   Oranges
*   And many more! Try different types of fruits to explore the detection capabilities.

For best results, use images with good lighting, a clear background, and where the fruit is centered in the frame.

## Disclaimer

NutriScan Dashboard is intended for educational and demonstration purposes. The accuracy of fruit identification depends on the performance of the Hugging Face model, and the nutritional data is sourced from the USDA FoodData Central API. While efforts are made to ensure accuracy, the information provided should not be considered a substitute for professional dietary or health advice. Always consult with qualified professionals for health-related decisions.

---

Enjoy using NutriScan Dashboard to explore the nutritional world of fruits!