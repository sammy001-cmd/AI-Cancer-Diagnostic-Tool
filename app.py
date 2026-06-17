import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import os
import time

# 1. Page Configuration
st.set_page_config(page_title="Oncology AI Portal", page_icon="🧬", layout="wide")

# --- ULTIMATE PREMIUM CSS (Tailwind/Inter Font/Glassmorphism) ---
css_styling = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%, #000000 100%); color: #f8fafc; }
[data-testid="stHeader"] { background-color: transparent !important; }
[data-testid="stSidebar"] { background: rgba(15, 23, 42, 0.4) !important; backdrop-filter: blur(16px) !important; -webkit-backdrop-filter: blur(16px) !important; border-right: 1px solid rgba(255, 255, 255, 0.05) !important; }
[data-testid="stImage"] > img { border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4); }
h1 { background: -webkit-linear-gradient(45deg, #38bdf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800 !important; padding-bottom: 10px; letter-spacing: -1px; }

/* Customizing the Streamlit Progress Bar to fit the theme */
.stProgress > div > div > div > div { background-image: linear-gradient(to right, #38bdf8, #c084fc); }
</style>"""
st.markdown(css_styling, unsafe_allow_html=True)

# --- MAIN HEADER ---
st.title("🧬 Diagnostic Intelligence")
st.markdown("<p style='color: #94a3b8; font-size: 1.1em; font-weight: 300;'>Advanced histological pattern recognition powered by Deep Learning.</p>", unsafe_allow_html=True)

# 2. Secure Model Loader
@st.cache_resource
def initialize_diagnostic_engine():
    model_path = 'cancer_diagnostic_model.h5'
    if not os.path.exists(model_path):
        st.error(f"⚠️ Missing Model: Could not find '{model_path}'.")
        return None
    return load_model(model_path)

cnn_engine = initialize_diagnostic_engine()

# --- SIDEBAR LAYOUT ---
with st.sidebar:
    st.markdown("<h2 style='color: #e2e8f0; font-weight: 600;'>📤 Input Parameters</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Select Tissue Scan", type=["jpg", "png", "jpeg"])
    
    st.markdown("---")
    
    # Back to the simple, clean info box!
    st.info("💡 **Note:** This AI was trained only on microscopic tissue scans. Please do not upload human faces, cars, or dogs!")
    
    st.markdown("---")
    
    # Simple, clean disclaimer moved to the sidebar
    st.caption("Disclaimer: This is a student computer science project for educational purposes. It utilizes a prototype Artificial Intelligence model and is strictly not intended for medical diagnosis.")

# --- MAIN DASHBOARD LAYOUT ---
if uploaded_file is None:
    awaiting_img = "<div style='background: rgba(255, 255, 255, 0.02); border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 40px; text-align: center; margin-top: 30px;'><p style='font-size: 3em; margin-bottom: 0;'>🔬</p><h3 style='color: #94a3b8; font-weight: 400; margin-top: 10px;'>Awaiting Image Input</h3><p style='color: #64748b; font-size: 0.9em;'>Please select an image file from the sidebar to initialize the neural network.</p></div>"
    st.markdown(awaiting_img, unsafe_allow_html=True)

else:
    st.write("<br/>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large") 

    with col1:
        st.markdown("<h3 style='color: #e2e8f0; font-weight: 600; font-size: 1.2em;'>Input Vision</h3>", unsafe_allow_html=True)
        input_img = Image.open(uploaded_file)
        st.image(input_img, use_column_width=True) 

    with col2:
        st.markdown("<h3 style='color: #e2e8f0; font-weight: 600; font-size: 1.2em;'>Inference Results</h3>", unsafe_allow_html=True)
        
        # --- THE HOLLYWOOD LOADING ANIMATION ---
        status_text = st.empty()
        loading_bar = st.progress(0)
        
        # Simulate deep scanning process for UI UX
        loading_messages = ["Extracting spatial features...", "Analyzing pixel density...", "Computing neural pathways...", "Finalizing tensor logic..."]
        for i in range(100):
            if i % 25 == 0:
                st_msg = f"<p style='color: #38bdf8; font-family: monospace; margin-bottom: 5px;'>[PROCESS] {loading_messages[i//25]}</p>"
                status_text.markdown(st_msg, unsafe_allow_html=True)
            loading_bar.progress(i + 1)
            time.sleep(0.01) # Tiny pause to create visual effect
            
        status_text.empty()
        loading_bar.empty()

        # Actual Prediction
        processed_img = input_img.resize((128, 128))
        img_matrix = image.img_to_array(processed_img)
        img_matrix = np.expand_dims(img_matrix, axis=0)
        img_matrix /= 255.0
        prediction_score = cnn_engine.predict(img_matrix)[0][0]
        
        # --- DISPLAY RESULTS ---
        if prediction_score > 0.5:
            confidence = prediction_score * 100
            
            malignant_banner = "<div style='background: rgba(239, 68, 68, 0.08); backdrop-filter: blur(16px); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 16px; padding: 25px; box-shadow: 0 4px 30px rgba(239, 68, 68, 0.1); margin-bottom: 20px;'><h2 style='color: #fca5a5; margin-top: 0; font-weight: 800; font-size: 2em; letter-spacing: -1px;'>MALIGNANT</h2><p style='color: #f8fafc; margin-bottom: 0; font-size: 0.95em;'>High probability of cancerous cell proliferation detected.</p></div>"
            st.markdown(malignant_banner, unsafe_allow_html=True)
            
            # Bringing back the clean progress bar you liked
            st.markdown(f"<p style='color: #cbd5e1; font-weight: 600; margin-bottom: 5px;'>Confidence Score: <span style='color: #fca5a5;'>{confidence:.2f}%</span></p>", unsafe_allow_html=True)
            st.progress(int(confidence))
            
        else:
            confidence = (1 - prediction_score) * 100
            
            benign_banner = "<div style='background: rgba(34, 197, 94, 0.08); backdrop-filter: blur(16px); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 16px; padding: 25px; box-shadow: 0 4px 30px rgba(34, 197, 94, 0.1); margin-bottom: 20px;'><h2 style='color: #86efac; margin-top: 0; font-weight: 800; font-size: 2em; letter-spacing: -1px;'>BENIGN</h2><p style='color: #f8fafc; margin-bottom: 0; font-size: 0.95em;'>Cellular patterns align with normal, healthy tissue architecture.</p></div>"
            st.markdown(benign_banner, unsafe_allow_html=True)
            
            # Bringing back the clean progress bar you liked
            st.markdown(f"<p style='color: #cbd5e1; font-weight: 600; margin-bottom: 5px;'>Confidence Score: <span style='color: #86efac;'>{confidence:.2f}%</span></p>", unsafe_allow_html=True)
            st.progress(int(confidence))
            
            
        
# import streamlit as st
# import numpy as np
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import load_model
# from PIL import Image
# import os

# # 1. Page Configuration
# st.set_page_config(page_title="Oncology AI Portal", page_icon="🧬", layout="wide")

# # --- PREMIUM GLASSMORPHISM & TAILWIND-STYLE CSS ---
# st.markdown("""
#     <style>
#     /* 1. Deep Dark Premium Gradient Background */
#     .stApp {
#         background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%, #000000 100%);
#         color: #f8fafc;
#     }
    
#     /* 2. Transparent Header */
#     [data-testid="stHeader"] {
#         background-color: transparent !important;
#     }

#     /* 3. Glassmorphism Sidebar */
#     [data-testid="stSidebar"] {
#         background: rgba(15, 23, 42, 0.4) !important;
#         backdrop-filter: blur(16px) !important;
#         -webkit-backdrop-filter: blur(16px) !important;
#         border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
#     }

#     /* 4. Glassmorphism File Uploader (Dropzone) */
#     [data-testid="stFileUploadDropzone"] {
#         background: rgba(255, 255, 255, 0.03) !important;
#         border: 1px dashed rgba(255, 255, 255, 0.1) !important;
#         border-radius: 16px !important;
#         backdrop-filter: blur(10px) !important;
#         transition: all 0.3s ease;
#     }
#     [data-testid="stFileUploadDropzone"]:hover {
#         background: rgba(255, 255, 255, 0.08) !important;
#         border: 1px dashed rgba(255, 255, 255, 0.3) !important;
#     }

#     /* 5. Glowing Image Frame */
#     [data-testid="stImage"] > img {
#         border-radius: 16px;
#         border: 1px solid rgba(255, 255, 255, 0.1);
#         box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
#     }

#     /* 6. Tailwind-style Gradient Typography */
#     h1 {
#         background: -webkit-linear-gradient(45deg, #38bdf8, #c084fc);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         font-weight: 800 !important;
#         padding-bottom: 10px;
#     }

#     /* 7. Premium Progress Bar */
#     .stProgress > div > div > div > div {
#         background-image: linear-gradient(to right, #38bdf8, #c084fc);
#     }

#     /* 8. Minimalist Footer */
#     .disclaimer {
#         font-size: 0.85em;
#         color: #64748b;
#         font-style: italic;
#         margin-top: 40px;
#         text-align: center;
#         padding-top: 20px;
#         border-top: 1px solid rgba(255,255,255,0.05);
#     }
#     </style>
# """, unsafe_allow_html=True)

# # --- MAIN HEADER ---
# st.title("🧬 Medical AI Diagnostic Assistant")
# st.markdown("<p style='color: #94a3b8; font-size: 1.1em;'>Welcome to the AI Pathology Portal. Upload a <b>cellular tissue scan</b> to receive an instant machine learning analysis.</p>", unsafe_allow_html=True)

# # 2. Secure Model Loader
# @st.cache_resource
# def initialize_diagnostic_engine():
#     model_path = 'cancer_diagnostic_model.h5'
#     if not os.path.exists(model_path):
#         st.error(f"⚠️ Missing Model: Could not find '{model_path}'.")
#         return None
#     return load_model(model_path)

# cnn_engine = initialize_diagnostic_engine()

# # --- SIDEBAR LAYOUT ---
# with st.sidebar:
#     st.markdown("<h2 style='color: #e2e8f0;'>📤 Upload Menu</h2>", unsafe_allow_html=True)
#     uploaded_file = st.file_uploader("Drag & drop a tissue scan here", type=["jpg", "png", "jpeg"])
    
#     st.markdown("---")
    
#     # Custom Glass Info Box in Sidebar
#     st.markdown("""
#         <div style="background: rgba(56, 189, 248, 0.05); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 12px; padding: 15px;">
#             <p style="color: #7dd3fc; margin: 0; font-size: 0.9em;"><b>💡 Note:</b> This AI was trained only on microscopic tissue scans. Please do not upload human faces, cars, or dogs!</p>
#         </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown("---")
#     st.markdown("<p style='color: #cbd5e1; font-size: 0.9em;'><b>Model Architecture:</b><br/>🧠 Convolutional Neural Network<br/>⚙️ 3.3 Million Parameters</p>", unsafe_allow_html=True)

# # --- MAIN DASHBOARD LAYOUT ---
# if uploaded_file is None:
#     # Custom Glass Notification when empty
#     st.markdown("""
#         <div style="background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 30px; text-align: center; margin-top: 20px;">
#             <h3 style="color: #94a3b8; font-weight: 400;">👈 Please upload a tissue image from the sidebar to begin the analysis.</h3>
#         </div>
#     """, unsafe_allow_html=True)
    
#     st.write("") # Spacer

#     with st.expander("How does this AI work?"):
#         st.markdown("<p style='color: #cbd5e1;'>This tool uses a Deep Learning model built with <b>TensorFlow and Keras</b>. It was trained on thousands of medical images to identify the chaotic, irregular cellular structures of malignant tumors versus the uniform structures of healthy benign tissue.</p>", unsafe_allow_html=True)

# else:
#     st.write("<br/>", unsafe_allow_html=True)
#     col1, col2 = st.columns([1, 1], gap="large") 

#     with col1:
#         st.markdown("<h3 style='color: #e2e8f0;'>📷 Uploaded Scan</h3>", unsafe_allow_html=True)
#         input_img = Image.open(uploaded_file)
#         st.image(input_img, use_column_width=True) 

#     with col2:
#         st.markdown("<h3 style='color: #e2e8f0;'>🔬 AI Analysis Results</h3>", unsafe_allow_html=True)
        
#         with st.spinner('Scanning cellular structures and extracting features...'):
#             # Prep the image
#             processed_img = input_img.resize((128, 128))
#             img_matrix = image.img_to_array(processed_img)
#             img_matrix = np.expand_dims(img_matrix, axis=0)
#             img_matrix /= 255.0

#             # Prediction
#             prediction_score = cnn_engine.predict(img_matrix)[0][0]
        
#         st.write("<br/>", unsafe_allow_html=True)
        
#         # --- CUSTOM HTML/CSS GLASSMORPHISM RESULTS CARDS ---
#         if prediction_score > 0.5:
#             confidence = prediction_score * 100
            
#             # Tailwind-style Red Glass Box
#             st.markdown("""
#                 <div style="background: rgba(239, 68, 68, 0.08); backdrop-filter: blur(16px); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 16px; padding: 25px; box-shadow: 0 4px 30px rgba(239, 68, 68, 0.1);">
#                     <h2 style="color: #fca5a5; margin-top: 0; font-weight: 700;">🚨 MALIGNANT</h2>
#                     <h4 style="color: #fecaca; margin-top: -10px;">Cancer Patterns Detected</h4>
#                     <p style="color: #f8fafc; margin-bottom: 0; font-size: 0.95em;">The neural network detected highly irregular cellular structures strongly associated with malignancy.</p>
#                 </div>
#             """, unsafe_allow_html=True)
            
#             st.write("<br/>", unsafe_allow_html=True)
#             st.markdown(f"<p style='color: #cbd5e1; font-weight: 600; margin-bottom: 5px;'>Confidence Score: <span style='color: #f8fafc;'>{confidence:.2f}%</span></p>", unsafe_allow_html=True)
#             st.progress(int(confidence))
            
#         else:
#             confidence = (1 - prediction_score) * 100
            
#             # Tailwind-style Green Glass Box
#             st.markdown("""
#                 <div style="background: rgba(34, 197, 94, 0.08); backdrop-filter: blur(16px); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 16px; padding: 25px; box-shadow: 0 4px 30px rgba(34, 197, 94, 0.1);">
#                     <h2 style="color: #86efac; margin-top: 0; font-weight: 700;">✅ BENIGN</h2>
#                     <h4 style="color: #bbf7d0; margin-top: -10px;">Normal Tissue</h4>
#                     <p style="color: #f8fafc; margin-bottom: 0; font-size: 0.95em;">The neural network detected uniform, healthy cellular structures. No malignancy detected.</p>
#                 </div>
#             """, unsafe_allow_html=True)
            
#             st.write("<br/>", unsafe_allow_html=True)
#             st.markdown(f"<p style='color: #cbd5e1; font-weight: 600; margin-bottom: 5px;'>Confidence Score: <span style='color: #f8fafc;'>{confidence:.2f}%</span></p>", unsafe_allow_html=True)
#             st.progress(int(confidence))
        
#         st.markdown("<p class='disclaimer'>Disclaimer: This is a student computer science project for educational purposes. This AI is not a certified medical device and should not be used for actual medical diagnoses.</p>", unsafe_allow_html=True)


        
        
        
# import streamlit as st
# import numpy as np
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import load_model
# from PIL import Image
# import os

# # 1. Page Configuration
# st.set_page_config(page_title="Oncology AI Portal", page_icon="🔬", layout="centered")

# st.title("🔬 Medical AI Digital Pathology Portal")
# st.write("Upload a raw histological tissue section snapshot to execute an automated screening analysis using the saved Convolutional Neural Network model.")
# # 2. Secure Model Loader
# @st.cache_resource
# def initialize_diagnostic_engine():
#     # Looks for the file in the current working directory
#     model_path = 'cancer_diagnostic_model.h5'
#     if not os.path.exists(model_path):
#         st.error(f"Critical Error: '{model_path}' not found in this folder. Please verify the file location.")
#         return None
#     return load_model(model_path)

# cnn_engine = initialize_diagnostic_engine()

# # 3. Dropzone Upload Target
# uploaded_file = st.file_uploader("Select or Drag & Drop Histology Image File...", type=["jpg", "png", "jpeg"])

# # 4. Processing Pipeline
# if uploaded_file is not None and cnn_engine is not None:
#     # Display preview
#     input_img = Image.open(uploaded_file)
#     st.image(input_img, caption='Stained Tissue Specimen Under Review', width=350)
    
#     with st.spinner('Parsing tissue matrices and calculating pixel boundaries...'):
#         # Match pipeline resizing parameters exactly
#         processed_img = input_img.resize((128, 128))
#         img_matrix = image.img_to_array(processed_img)
#         img_matrix = np.expand_dims(img_matrix, axis=0)
#         img_matrix /= 255.0  # Min-max normalization

#         # Inferencing
#         prediction_score = cnn_engine.predict(img_matrix)[0][0]
        
#     # 5. Output Report Block
#     st.markdown("### 📋 Diagnostic Analysis Report")
    
#     if prediction_score > 0.5:
#         st.error("#### STATUS: MALIGNANT CELL PATTERNS DETECTED")
#         st.write(f"The neural network has flagged high structural deviation. **Model Confidence:** {prediction_score * 100:.2f}%")
#     else:
#         st.success("#### STATUS: BENIGN / NEGATIVE FOR MALIGNANCY")
#         st.write(f"Cellular array structures display uniform configurations. **Model Confidence:** {(1 - prediction_score) * 100:.2f}%")