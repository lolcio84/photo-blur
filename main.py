import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image
import io
import cv2

# Konfiguracja strony
st.set_page_config(page_title="Magic Image Tool", layout="centered")
st.title("🪄 Magic Image Tool (Pro)")
st.write("Wgraj zdjęcie, wykryj twarz i dopasuj granice rozmycia.")

# Inicjalizacja MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=5, min_detection_confidence=0.5)

# --- Funkcje pomocnicze ---

def process_face_detection(img_array):
    """Wykrywa twarze i zwraca punkty (landmarks)."""
    results = face_mesh.process(img_array)
    faces_data = []
    
    if results.multi_face_landmarks:
        height, width, _ = img_array.shape
        for face_landmarks in results.multi_face_landmarks:
            points = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append([x, y])
            faces_data.append(np.array(points, dtype=np.int32))
            
    return faces_data, (img_array.shape[0], img_array.shape[1])

def calculate_blur_kernel(strength_value):
    """
    Konwertuje wartość suwaka (1-100) na kernel Gaussa.
    Używa skali wykładniczej, żeby każda zmiana była odczuwalna.
    """
    min_kernel = 5
    max_kernel = 199
    
    # Normalizacja do 0-1
    normalized = strength_value / 100.0
    
    # Funkcja wykładnicza dla lepszej kontroli
    kernel = min_kernel + (max_kernel - min_kernel) * (normalized ** 1.5)
    kernel = int(kernel)
    
    # Upewnij się, że kernel jest nieparzysty
    if kernel % 2 == 0:
        kernel += 1
    
    return kernel

def create_blur_image(img_array, faces_data, shape, blur_strength, margin_size):
    """Tworzy obraz z blurem na podstawie zapisanych punktów i ustawień suwaków."""
    height, width = shape
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Oblicz kernel rozmycia ze skali wykładniczej
    kernel_size = calculate_blur_kernel(blur_strength)
    
    for points in faces_data:
        # 1. Otoczka wypukła
        hull = cv2.convexHull(points)
        
        # 2. Aplikowanie marginesu
        temp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(temp_mask, [hull], 255)
        
        if margin_size > 0:
            kernel = np.ones((5, 5), np.uint8)
            temp_mask = cv2.dilate(temp_mask, kernel, iterations=int(margin_size))
        elif margin_size < 0:
            kernel = np.ones((5, 5), np.uint8)
            temp_mask = cv2.erode(temp_mask, kernel, iterations=abs(int(margin_size)))
        
        mask = cv2.bitwise_or(mask, temp_mask)
    
    # 3. Wygładzenie krawędzi maski
    mask = cv2.GaussianBlur(mask, (31, 31), 5)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    
    # 4. Rozmycie obrazu z dynamicznym kernelem
    sigma = 30 + (blur_strength / 3)
    blurred_cv = cv2.GaussianBlur(img_cv, (kernel_size, kernel_size), sigma)
    
    # 5. Łączenie obrazów
    result_cv = (blurred_cv * mask_3ch + img_cv * (1 - mask_3ch)).astype(np.uint8)
    result_rgb = cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(result_rgb)

# --- Główna logika aplikacji ---

# 1. Upload pliku
uploaded_file = st.file_uploader("Wybierz zdjęcie...", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"])

if uploaded_file:
    # Wczytanie obrazu
    image_pil = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image_pil)
    
    # Inicjalizacja session state dla tego pliku
    if 'uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.faces_data = None
        st.session_state.shape = None
        st.session_state.img_array = img_array
    
    st.image(image_pil, caption="Oryginalne zdjęcie", use_container_width=True)
    
    # 2. Przycisk wykrywania
    if st.button("🔍 Wykryj twarze"):
        with st.spinner('Analizuję geometrię twarzy...'):
            faces_data, shape = process_face_detection(img_array)
            
            if faces_data:
                st.session_state.faces_data = faces_data
                st.session_state.shape = shape
                st.session_state.img_array = img_array
                st.success(f"✅ Znaleziono {len(faces_data)} twarzy! Dostosuj efekt poniżej.")
            else:
                st.warning("❌ Nie wykryto żadnych twarzy.")
    
    # 3. Panel edycji (pojawia się tylko po wykryciu)
    if st.session_state.faces_data:
        st.divider()
        st.subheader("🎛️ Panel Dostosowania")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Suwak mocy rozmycia (1-100 ze skalą wykładniczą)
            blur_strength = st.slider(
                "💨 Siła rozmycia", 
                min_value=1, 
                max_value=100, 
                value=50, 
                step=1,
                help="1 = lekkie rozmycie, 100 = maksymalne zamazanie. Skala jest nieliniowa dla lepszej kontroli."
            )
            # Podgląd aktualnej wartości kernela
            current_kernel = calculate_blur_kernel(blur_strength)
            st.caption(f"Aktualny kernel: {current_kernel}x{current_kernel}")
        
        with col2:
            # Suwak granic (marginesu)
            margin_size = st.slider(
                "📏 Granice (Margines)", 
                min_value=-10, 
                max_value=30, 
                value=2, 
                step=1,
                help="Dodatnie: powiększa obszar bluru. Ujemne: zmniejsza obszar bluru."
            )
        
        # Podgląd na żywo
        st.write("### 🖼️ Podgląd na żywo:")
        preview_image = create_blur_image(
            st.session_state.img_array, 
            st.session_state.faces_data, 
            st.session_state.shape,
            blur_strength, 
            margin_size
        )
        
        st.image(preview_image, caption="Efekt końcowy", use_container_width=True)
        
        # 4. Opcje pobierania
        st.divider()
        st.subheader("💾 Pobierz zdjęcie")
        
        col_dl1, col_dl2 = st.columns([2, 1])
        
        with col_dl1:
            # Wybór formatu pliku
            download_format = st.selectbox(
                "Wybierz format pliku:",
                options=["PNG", "JPEG", "WEBP", "BMP", "TIFF"],
                index=0,
                help="PNG = bezstratny (większy plik), JPEG = mniejszy plik (kompresja), WEBP = nowoczesny format internetowy"
            )
        
        with col_dl2:
            # Jakość dla formatów stratnych (JPEG, WEBP)
            if download_format in ["JPEG", "WEBP"]:
                quality = st.slider(
                    "Jakość",
                    min_value=10,
                    max_value=100,
                    value=90,
                    step=5
                )
            else:
                quality = 100  # PNG, BMP, TIFF nie używają kompresji stratnej
        
        # Przygotowanie pliku do pobrania
        buf = io.BytesIO()
        
        # Konwersja formatu
        if download_format == "JPEG":
            # JPEG nie obsługuje przezroczystości, więc konwertujemy do RGB
            preview_image = preview_image.convert("RGB")
            preview_image.save(buf, format="JPEG", quality=quality, optimize=True)
            file_ext = "jpg"
            mime_type = "image/jpeg"
        elif download_format == "WEBP":
            preview_image.save(buf, format="WEBP", quality=quality, lossless=False)
            file_ext = "webp"
            mime_type = "image/webp"
        elif download_format == "BMP":
            preview_image.save(buf, format="BMP")
            file_ext = "bmp"
            mime_type = "image/bmp"
        elif download_format == "TIFF":
            preview_image.save(buf, format="TIFF")
            file_ext = "tiff"
            mime_type = "image/tiff"
        else:  # PNG
            preview_image.save(buf, format="PNG", optimize=True)
            file_ext = "png"
            mime_type = "image/png"
        
        byte_im = buf.getvalue()
        file_size_kb = len(byte_im) / 1024
        
        # Informacja o rozmiarze pliku
        st.caption(f"📊 Szacowany rozmiar pliku: {file_size_kb:.1f} KB")
        
        # Przycisk pobierania
        st.download_button(
            label=f"📥 Pobierz jako {download_format}",
            data=byte_im,
            file_name=f"magic_blur.{file_ext}",
            mime=mime_type,
            type="primary",
            use_container_width=True
        )

st.divider()
st.caption("💡 Wskazówka: PNG zachowuje najwyższą jakość. JPEG/WEBP są mniejsze i lepsze do udostępniania.")