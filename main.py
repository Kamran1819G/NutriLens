import google.generativeai as genai
import streamlit as st
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import time
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

USDA_API_KEY = os.getenv("USDA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print(f"USDA_API_KEY from .env: {USDA_API_KEY}")  # Debug print
print(f"GEMINI_API_KEY from .env: {GEMINI_API_KEY}")  # Debug print

# --- USDA API Configuration ---
USDA_API_BASE_URL = "https://api.nal.usda.gov/fdc/v1"

# --- Gemini API Configuration ---
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')
else:
    gemini_model = None
    st.warning(
        "Gemini API key not configured. Health benefit summaries will be placeholders.")

# Set Streamlit page configuration with custom theme
st.set_page_config(
    page_title="NutriScan Dashboard",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        color: #2E7D32;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-weight: bold;
        padding-bottom: 20px;
        border-bottom: 2px solid #E8F5E9;
        margin-bottom: 30px;
    }
    h2, h3 {
        color: #1B5E20;
        font-family: 'Segoe UI', Arial, sans-serif;
        margin-top: 30px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 10px 24px;
        margin: 20px 0px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #388E3C;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    .nutrition-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .health-benefit {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 30px;
    }
    .logo {
        margin-right: 20px;
        font-size: 40px;
    }
    footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        border-top: 1px solid #E8F5E9;
        color: #7c7c7c;
    }
    </style>
""", unsafe_allow_html=True)

# App Header with Logo
st.markdown("""
    <div class="header-container">
        <div class="logo">🍎</div>
        <div>
            <h1>NutriScan Dashboard</h1>
        </div>
    </div>
""", unsafe_allow_html=True)

# Load CNN model


@st.cache_resource
def load_ml_model():
    return load_model(r"model.h5")


try:
    model = load_ml_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Fruit class mapping
class_labels = {
    0: "apple",
    1: "banana",
    2: "grapes",
    3: "mango",
    4: "strawberry"
}

# --- USDA API Functions ---


@st.cache_data(ttl=3600)
def get_fruit_data_from_usda(fruit_name):
    """Fetches fruit data from USDA FoodData Central API."""
    if not USDA_API_KEY:
        st.warning("USDA API key is not set in environment variables.")
        return None

    try:
        search_url = f"{USDA_API_BASE_URL}/foods/search"
        params = {
            "api_key": USDA_API_KEY,
            "query": fruit_name,
            "dataType": ["Survey (FNDDS)", "Foundation", "SR Legacy"],
            "pageSize": 1
        }

        st.info(f"Requesting USDA data for: {fruit_name}")
        response = requests.get(search_url, params=params)

        # Debug response
        st.info(f"USDA API Status Code: {response.status_code}")

        if response.status_code != 200:
            st.error(f"USDA API Error: {response.text}")
            return None

        search_results = response.json()

        if not search_results.get('foods') or len(search_results['foods']) == 0:
            st.warning(
                f"No results found for '{fruit_name}' in USDA database.")
            return None

        # Get FDC ID of the first result
        food_id = search_results['foods'][0]['fdcId']

        food_url = f"{USDA_API_BASE_URL}/food/{food_id}"
        food_params = {
            "api_key": USDA_API_KEY,
            "format": "full"
        }

        st.info(f"Fetching detailed data for food ID: {food_id}")
        food_response = requests.get(food_url, params=food_params)

        # Debug response
        st.info(
            f"USDA Food Detail API Status Code: {food_response.status_code}")

        if food_response.status_code != 200:
            st.error(f"USDA Food Detail API Error: {food_response.text}")
            return None

        food_data = food_response.json()
        return food_data

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from USDA API: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error in USDA API request: {e}")
        return None


def process_usda_data_for_app(usda_food_data):
    """Processes USDA data to match the format expected by the app."""
    if not usda_food_data:
        return None

    # Initialize with default structure and values
    processed_fruit_data = {
        "name": usda_food_data.get('description', 'Unknown Fruit'),
        "calories": "0 kcal",
        "proteins": "0 g",
        "carbohydrates": "0 g",
        "fiber": "0 g",
        "sugar": "0 g",
        "vitamins": {"A": "0 µg", "C": "0 mg"},
        "minerals": {"iron": "0 mg", "calcium": "0 mg"},
        "health_benefits": []
    }

    # Process food nutrients using the correct key structure
    nutrient_mapping = {
        "Energy": {"key": "calories", "group": None},
        "Protein": {"key": "proteins", "group": None},
        "Carbohydrate, by difference": {"key": "carbohydrates", "group": None},
        "Fiber, total dietary": {"key": "fiber", "group": None},
        "Total Sugars": {"key": "sugar", "group": None},
        "Sugars, total including NLEA": {"key": "sugar", "group": None},
        "Sugars, total": {"key": "sugar", "group": None},
        "Vitamin A, RAE": {"key": "A", "group": "vitamins"},
        "Vitamin A, IU": {"key": "A", "group": "vitamins"},
        "Vitamin C, total ascorbic acid": {"key": "C", "group": "vitamins"},
        "Iron, Fe": {"key": "iron", "group": "minerals"},
        "Calcium, Ca": {"key": "calcium", "group": "minerals"}
    }

    # USDA API has different key structures based on the data type
    # Check for different possible structures
    nutrients = usda_food_data.get('foodNutrients', [])

    for nutrient in nutrients:
        # Handle different response formats from USDA API
        nutrient_name = None
        nutrient_value = None
        unit_name = None

        # Format 1: nutrient has nutrient object
        if 'nutrient' in nutrient:
            nutrient_name = nutrient['nutrient'].get('name')
            nutrient_value = nutrient.get('amount')
            unit_name = nutrient['nutrient'].get('unitName')

        # Format 2: direct properties
        else:
            nutrient_name = nutrient.get('nutrientName')
            nutrient_value = nutrient.get('value')
            unit_name = nutrient.get('unitName')

        # Skip if missing critical data
        if not nutrient_name or nutrient_value is None or not unit_name:
            continue

        # Process the nutrient based on the mapping
        if nutrient_name in nutrient_mapping:
            mapping = nutrient_mapping[nutrient_name]
            value_with_unit = f"{nutrient_value} {unit_name}"

            if mapping["group"] is None:
                processed_fruit_data[mapping["key"]] = value_with_unit
            else:
                processed_fruit_data[mapping["group"]
                                     ][mapping["key"]] = value_with_unit

    return processed_fruit_data

# --- Gemini Health Benefit Summarization Function ---


@st.cache_data(ttl=3600)  # Cache Gemini summaries for an hour
def generate_health_benefit_summary_gemini(fruit_data):
    """Generates a health benefit summary using Google Gemini."""
    global gemini_model
    if not gemini_model:
        return "**Gemini Health Benefit Summaries are disabled.** Please configure Gemini API keys to enable this feature."

    if not fruit_data:
        return "No health benefit information available."

    fruit_name = fruit_data['name']
    nutrients_for_prompt = ""
    for nutrient_type in ["vitamins", "minerals"]:
        for nutrient, value in fruit_data.get(nutrient_type, {}).items():
            nutrients_for_prompt += f"{nutrient}: {value}, "
    nutrients_for_prompt = nutrients_for_prompt.rstrip(
        ", ")  # Remove trailing comma

    prompt = f"""Summarize the key health benefits of eating {fruit_name}, based on its nutritional content: {nutrients_for_prompt}. Keep the summary concise (around 3-4 sentences) and focus on benefits relevant to an average person's diet and health. """

    try:
        response = gemini_model.generate_content(prompt)
        summary = response.text
        return summary
    except Exception as e:  # Catch any potential errors
        st.error(
            f"Error generating health benefit summary with Gemini API: {e}")
        return "Error generating AI summary. Please try again later."


# Main content area
st.subheader("📸 Upload Fruit Image")
st.markdown(
    "Take a photo or upload an image of a fruit to analyze its nutritional content.")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a single fruit for best results"
)
st.markdown('</div>', unsafe_allow_html=True)

# CNN detection function


def detect_objects(img):
    with st.spinner('Analyzing image...'):
        img = img.resize((128, 128))  # Match CNN input size
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        time.sleep(1)  # Slight delay for spinner
        predictions = model.predict(img_array)
        confidence_scores = predictions[0] * 100
        predicted_class = np.argmax(predictions)
        fruit_name = class_labels.get(predicted_class, "Unknown")
        return fruit_name, confidence_scores

# Function to create nutrition facts table


def create_nutrition_facts(fruit):
    vitamins = fruit.get('vitamins', {})
    minerals = fruit.get('minerals', {})

    st.subheader("📊 Nutrition Facts")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Basic Information**")
        st.markdown(f"**Serving Size:** 100g")
        st.markdown(f"**Calories:** {fruit.get('calories', 'N/A')}")
        st.markdown(f"**Proteins:** {fruit.get('proteins', 'N/A')}")
        st.markdown(f"**Carbohydrates:** {fruit.get('carbohydrates', 'N/A')}")
        st.markdown(f"**Fiber:** {fruit.get('fiber', 'N/A')}")
        st.markdown(f"**Sugar:** {fruit.get('sugar', 'N/A')}")

    with col2:
        st.markdown("**Vitamins & Minerals**")
        st.markdown(f"**Vitamin A:** {vitamins.get('A', 'N/A')}")
        st.markdown(f"**Vitamin C:** {vitamins.get('C', 'N/A')}")
        st.markdown(f"**Iron:** {minerals.get('iron', 'N/A')}")
        st.markdown(f"**Calcium:** {minerals.get('calcium', 'N/A')}")

    st.markdown('</div>', unsafe_allow_html=True)

# Function to create macronutrient visualization


def create_macro_chart(fruit):
    nutrient_labels = ['Proteins', 'Carbohydrates', 'Fiber', 'Sugar']
    nutrient_values_str = [
        fruit.get('proteins', '0 g').split(' ')[0],
        fruit.get('carbohydrates', '0 g').split(' ')[0],
        fruit.get('fiber', '0 g').split(' ')[0],
        fruit.get('sugar', '0 g').split(' ')[0]
    ]
    nutrient_values = []
    for val_str in nutrient_values_str:
        try:
            nutrient_values.append(float(val_str))
        except ValueError:
            nutrient_values.append(0)

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA62B']

    fig = px.bar(
        x=nutrient_labels,
        y=nutrient_values,
        text=nutrient_values,
        color=nutrient_labels,
        color_discrete_sequence=colors,
        labels={'x': 'Nutrient', 'y': 'Amount (grams)'},
        title=f'Macronutrient Composition (per 100g)'
    )

    fig.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        font_family='Arial',
        title_font_size=18,
        title_x=0.5,
        height=400,
        xaxis=dict(title_font=dict(size=14)),
        yaxis=dict(title_font=dict(size=14))
    )
    fig.update_traces(texttemplate='%{y:.1f}g', textposition='outside')
    return fig

# Function to create vitamins & minerals visualization


def create_vitamins_minerals_chart(fruit):
    vitamins = fruit.get('vitamins', {})
    minerals = fruit.get('minerals', {})

    # Assuming units for radar chart
    labels = ['Vitamin A (μg)', 'Vitamin C (mg)', 'Iron (mg)', 'Calcium (mg)']
    values = []

    # Extract numeric values and handle units
    def extract_numeric(value_str):
        if not value_str or value_str == 'N/A':
            return 0
        try:
            return float(''.join(c for c in value_str.split(' ')[0] if c.isdigit() or c == '.'))
        except:
            return 0

    values.append(extract_numeric(vitamins.get('A', '0')))
    values.append(extract_numeric(vitamins.get('C', '0')))
    values.append(extract_numeric(minerals.get('iron', '0')))
    values.append(extract_numeric(minerals.get('calcium', '0')))

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name=fruit['name'].capitalize(),
        line_color='#4CAF50'
    ))

    max_val = max(values) if values else 1  # Avoid max on empty list
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_val * 1.2 if max_val > 0 else 1.2]
            )
        ),
        title='Micronutrient Profile',
        title_font_size=18,
        title_x=0.5,
        showlegend=False,
        height=400
    )
    return fig

# Function to display health benefits


def display_health_benefits(fruit):
    st.subheader("💪 Health Benefits (AI Powered)")
    gemini_summary = generate_health_benefit_summary_gemini(fruit)
    st.markdown(
        f'<div class="health-benefit">{gemini_summary}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Function to show confidence scores


def show_confidence_scores(scores):
    st.subheader("🔍 Detection Confidence")
    fruits = list(class_labels.values())
    sorted_indices = np.argsort(scores)[::-1]
    sorted_fruits = [fruits[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]

    fig = px.bar(
        x=sorted_scores,
        y=sorted_fruits,
        orientation='h',
        text=[f"{score:.1f}%" for score in sorted_scores],
        color=sorted_scores,
        color_continuous_scale='Viridis',
        labels={'x': 'Confidence (%)', 'y': 'Fruit Type'},
        title='Detection Confidence Scores'
    )
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        height=300,
        margin=dict(l=0, r=10, t=40, b=0),
        coloraxis_showscale=False
    )
    fig.update_traces(textposition='auto')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# Main application logic - now only using USDA API
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=400)
        st.markdown('</div>', unsafe_allow_html=True)

        analyze_button = st.button("🔍 Analyze Fruit", use_container_width=True)

        if analyze_button and model_loaded:
            detected_fruit_name, confidence_scores = detect_objects(image)

            if detected_fruit_name != "Unknown":
                st.success(
                    f"✅ Successfully identified: **{detected_fruit_name.capitalize()}**!")

                # Get fruit data from USDA only
                usda_fruit_data = get_fruit_data_from_usda(detected_fruit_name)

                if usda_fruit_data:
                    detected_fruit = process_usda_data_for_app(usda_fruit_data)

                    if detected_fruit:
                        tab1, tab2, tab3 = st.tabs(
                            ["📊 Nutrition Facts", "📈 Visualizations", "💪 Health Benefits"])

                        with tab1:
                            create_nutrition_facts(detected_fruit)
                            show_confidence_scores(confidence_scores)

                        with tab2:
                            col1, col2 = st.columns(2)
                            with col1:
                                fig1 = create_macro_chart(detected_fruit)
                                st.plotly_chart(fig1, use_container_width=True)
                            with col2:
                                fig2 = create_vitamins_minerals_chart(
                                    detected_fruit)
                                st.plotly_chart(fig2, use_container_width=True)

                        with tab3:
                            display_health_benefits(detected_fruit)

                            st.subheader("🔄 Compare with another fruit")
                            # Create a list of other fruits that can be detected
                            other_fruits = [
                                f for f in class_labels.values() if f != detected_fruit_name]
                            compare_fruit = st.selectbox(
                                "Select a fruit to compare with:",
                                other_fruits
                            )

                            if st.button("Compare"):
                                with st.spinner(f"Fetching USDA data for {compare_fruit}..."):
                                    # Get USDA data for comparison fruit
                                    comparison_usda_data = get_fruit_data_from_usda(
                                        compare_fruit)
                                    if comparison_usda_data:
                                        comparison_fruit_data = process_usda_data_for_app(
                                            comparison_usda_data)

                                        if comparison_fruit_data:
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.markdown(
                                                    f"**{detected_fruit_name.capitalize()}**")
                                                st.markdown(
                                                    f"Calories: {detected_fruit.get('calories', 'N/A')}")
                                                st.markdown(
                                                    f"Proteins: {detected_fruit.get('proteins', 'N/A')}")
                                                st.markdown(
                                                    f"Fiber: {detected_fruit.get('fiber', 'N/A')}")
                                                st.markdown(
                                                    f"Sugar: {detected_fruit.get('sugar', 'N/A')}")

                                            with col2:
                                                st.markdown(
                                                    f"**{compare_fruit.capitalize()}**")
                                                st.markdown(
                                                    f"Calories: {comparison_fruit_data.get('calories', 'N/A')}")
                                                st.markdown(
                                                    f"Proteins: {comparison_fruit_data.get('proteins', 'N/A')}")
                                                st.markdown(
                                                    f"Fiber: {comparison_fruit_data.get('fiber', 'N/A')}")
                                                st.markdown(
                                                    f"Sugar: {comparison_fruit_data.get('sugar', 'N/A')}")
                                        else:
                                            st.error(
                                                f"Could not process USDA data for {compare_fruit}.")
                                    else:
                                        st.error(
                                            f"Could not retrieve USDA data for {compare_fruit}.")
                    else:
                        st.error(
                            f"Could not process USDA data for {detected_fruit_name}.")
                else:
                    st.warning(
                        f"We identified the fruit as {detected_fruit_name}, but could not retrieve data from the USDA database.")
                    st.info(
                        "This could be due to API limitations or network issues. Please try again later.")
            else:
                st.error("No known fruits detected. Please try a different image.")
                st.info("Tips: Good lighting, clear background, fruit centered.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please try uploading a different image.")
else:
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("👋 Welcome to NutriScan!")
        st.markdown("""
        **Get started by uploading a fruit image above.**

        NutriScan identifies fruits and provides nutritional info from the USDA FoodData Central database and AI-powered health benefit summaries.

        Our AI detects:
        - 🍎 Apples
        - 🍌 Bananas
        - 🍇 Grapes
        - 🥭 Mangoes
        - 🍓 Strawberries
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.image("https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
                 caption="Example Image")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <footer>
        <p>NutriScan Dashboard v2.0 | Powered by AI & USDA | Data from USDA FoodData Central</p>
    </footer>
""", unsafe_allow_html=True)
